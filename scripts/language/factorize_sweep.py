#!/usr/bin/env python
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from balf.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
)
from balf.utils import (
    count_model_flops,
    get_all_convs_and_linears,
    seed_everything,
    make_factorization_cache_location,
    maybe_retrieve_activation_cache,
)


class ContiguousSeqDataset(Dataset):
    def __init__(self, token_ids: torch.LongTensor, seq_len: int):
        n_tokens = (token_ids.size(0) // seq_len) * seq_len
        token_ids = token_ids[:n_tokens]
        self.seq_len = seq_len
        self.samples = token_ids.view(-1, seq_len)

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return {"input_ids": ids.clone(), "attention_mask": torch.ones_like(ids)}


def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}


@torch.no_grad()
def perplexity(model, dataloader, device=None, ignore_inf=True):
    device = device or next(model.parameters()).device
    model.to(device).eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nll_tokens = []
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        logits = model(
            input_ids=input_ids, attention_mask=attn_mask, use_cache=False
        ).logits
        if ignore_inf and not torch.isfinite(logits).all():
            continue
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nll_tokens.append(loss)
        total_tokens += shift_labels.numel()
    if total_tokens == 0:
        raise RuntimeError("No valid tokens processed")
    mean_nll = torch.cat(nll_tokens, dim=0).mean()
    return math.exp(mean_nll.item())


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--dataset", choices=["wikitext2", "ptb"], default="wikitext2")
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--calib_size", type=int, default=128)
parser.add_argument(
    "--mode",
    default="flops_auto",
    choices=["flops_auto", "params_auto", "energy_act_aware", "energy"],
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
model = (
    AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    .to(device)
    .eval()
)

if args.dataset == "wikitext2":
    eval_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    calib_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
else:
    eval_texts = load_dataset("ptb_text_only", "penn_treebank", split="validation", trust_remote_code=True)[
        "sentence"
    ]
    calib_texts = load_dataset("ptb_text_only", "penn_treebank", split="train", trust_remote_code=True)[
        "sentence"
    ]

eval_tok_ids = tok("\n\n".join(eval_texts), return_tensors="pt").input_ids.squeeze(0)
calib_tok_ids = tok("\n\n".join(calib_texts), return_tensors="pt").input_ids.squeeze(0)

eval_ds = ContiguousSeqDataset(eval_tok_ids, args.seq_len)
calib_ds_full = ContiguousSeqDataset(calib_tok_ids, args.seq_len)

args.eval_subset = 8

if args.eval_subset is not None and args.eval_subset < len(eval_ds):
    idx = torch.randperm(len(eval_ds))[: args.eval_subset]
    eval_ds = torch.utils.data.Subset(eval_ds, idx)

if len(calib_ds_full) > args.calib_size:
    idx = torch.randperm(len(calib_ds_full))[: args.calib_size]
    calib_ds = torch.utils.data.Subset(calib_ds_full, idx)
else:
    calib_ds = calib_ds_full

eval_dl = DataLoader(
    eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
)


calib_dl = DataLoader(
    calib_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
)

ppl_orig = perplexity(model, eval_dl, device=device)
params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, (1, args.seq_len), dtype=torch.long)["total"]
print(f"[original] ppl={ppl_orig:.4f} params={params_orig} flops_total={flops_orig}")

all_keys = get_all_convs_and_linears(model)
skip_re = re.compile(r"(embed_tokens|embed_positions|lm_head)")
layer_keys = [k for k in all_keys if not skip_re.search(k)]

activation_cache = maybe_retrieve_activation_cache(
    model_name=args.model_name,
    calib_size=args.calib_size,
    dataset_name=args.dataset,
    script_key="factorize_sweep",
    seed=args.seed,
    model=model,
    dataloader=calib_dl,
    keys=layer_keys,
)

base_dir = Path(args.results_dir)
base_dir.mkdir(parents=True, exist_ok=True)
results = []

ratios_comp = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
ratios_energy = [0.85, 0.9, 0.95, 0.98, 0.99999]

for k in ratios_comp if args.mode in ["flops_auto", "params_auto"] else ratios_energy:
    if args.mode in ["flops_auto", "params_auto"]:
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=k,
            inplace=False,
            keys=layer_keys,
            metric="flops" if args.mode == "flops_auto" else "params",
            save_dir=make_factorization_cache_location(
                args.model_name,
                args.calib_size,
                args.dataset,
                "factorize_sweep",
                args.seed,
            ),
        )
    elif args.mode == "energy_act_aware":
        model_lr = to_low_rank_activation_aware_manual(
            model,
            activation_cache,
            cfg_dict={
                kk: {"name": "svals_energy_ratio_to_keep", "value": k}
                for kk in layer_keys
            },
            inplace=False,
            save_dir=make_factorization_cache_location(
                args.model_name,
                args.calib_size,
                args.dataset,
                "factorize_sweep",
                args.seed,
            ),
        )
    else:
        model_lr = to_low_rank_manual(
            model,
            cfg_dict={
                kk: {"name": "svals_energy_ratio_to_keep", "value": k}
                for kk in layer_keys
            },
            inplace=False,
        )

    model_lr.to(dtype=torch.float16).to(device).eval()
    ppl_lr = perplexity(model_lr, eval_dl, device=device)
    params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_lr = count_model_flops(model_lr, (1, args.seq_len), dtype=torch.long)["total"]

    print(
        f"[ratio={k:.6f}] ppl={ppl_lr:.4f} params_ratio={params_lr/params_orig:.4f} flops_ratio={flops_lr/flops_orig:.4f}"
    )

    results.append(
        {
            "metric_value": float(k),
            "ppl": float(ppl_lr),
            "ppl_orig": float(ppl_orig),
            "params_ratio": float(params_lr / params_orig),
            "flops_ratio": float(flops_lr / flops_orig),
            "mode": args.mode,
            "seq_len": args.seq_len,
        }
    )

results.append(
    {
        "metric_value": "original",
        "ppl": float(ppl_orig),
        "ppl_orig": float(ppl_orig),
        "params_ratio": 1.0,
        "flops_ratio": 1.0,
        "mode": args.mode,
        "seq_len": args.seq_len,
    }
)

with open(base_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
