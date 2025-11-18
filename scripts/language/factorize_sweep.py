import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator

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
import copy
import gc

from lib.dataset_utils_language import test_ppl, get_train
from lm_eval.models.huggingface import HFLM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--calib_size", type=int, default=256)
parser.add_argument(
    "--mode",
    default="flops_auto",
    choices=["flops_auto", "params_auto", "energy_act_aware", "rank"],
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_tasks", required=True)
args = parser.parse_args()

eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]

seed_everything(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
model = (
    AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    .to(device)
    .eval()
)

ppl_orig = test_ppl(
    model,
    tok,
    datasets=[args.dataset],
    ppl_seqlen=args.seq_len,
)
params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, (1, args.seq_len), dtype=torch.long)["total"]
print(f"[original] ppl={ppl_orig} params={params_orig} flops_total={flops_orig}")
eval_raw_orig = evaluator.simple_evaluate(
    HFLM(model, batch_size=16),
    tasks=eval_tasks,
    batch_size=args.batch_size,
    limit=128,
)
eval_results_orig = {t: eval_raw_orig["results"][t]["acc,none"] for t in eval_tasks}
print("[original]", eval_results_orig)

all_keys = get_all_convs_and_linears(model)
skip_re = re.compile(r"(embed_tokens|embed_positions|lm_head)")
layer_keys = [k for k in all_keys if not skip_re.search(k)]

train_ds = get_train(
    args.dataset,
    tok,
    size=args.calib_size,
    seed=args.seed,
    seqlen=args.seq_len,
)

train_ds = list(map(lambda tupl: tupl[0], train_ds))

train_dl = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
)

activation_cache = maybe_retrieve_activation_cache(
    model_name=args.model_name,
    calib_size=args.calib_size,
    dataset_name=args.dataset,
    script_key="factorize_sweep",
    seed=args.seed,
    model=model,
    dataloader=train_dl,
    keys=layer_keys,
)

base_dir = Path(args.results_dir)
base_dir.mkdir(parents=True, exist_ok=True)
results = []

ratios_comp = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
ratios_energy = [0.85, 0.9, 0.95, 0.98, 0.99999]

model = model.cpu()

for k in (
    ratios_comp if args.mode in ["flops_auto", "params_auto", "rank"] else ratios_energy
):
    model_copy = copy.deepcopy(model).half().cuda()
    if args.mode in ["flops_auto", "params_auto"]:
        model_lr = to_low_rank_activation_aware_auto(
            model_copy,
            activation_cache,
            ratio_to_keep=k,
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
            model_copy,
            activation_cache,
            cfg_dict={
                kk: {"name": "svals_energy_ratio_to_keep", "value": k}
                for kk in layer_keys
            },
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
            model_copy,
            cfg_dict={
                kk: {"name": "rank_ratio_to_keep", "value": k} for kk in layer_keys
            },
        )

    model_lr.to(dtype=torch.float16).to(device).eval()
    ppl_lr = test_ppl(
        model_lr,
        tok,
        datasets=[args.dataset],
        ppl_seqlen=args.seq_len,
    )
    params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_lr = count_model_flops(model_lr, (1, args.seq_len), dtype=torch.long)["total"]

    print(
        f"[ratio={k:.6f}] ppl={ppl_lr:} params_ratio={params_lr/params_orig:.4f} flops_ratio={flops_lr/flops_orig:.4f}"
    )
    lm = HFLM(model_lr, batch_size=16)
    eval_raw_lr = evaluator.simple_evaluate(
        lm,
        tasks=eval_tasks,
        batch_size=args.batch_size,
        limit=128,
    )
    eval_results_lr = {t: eval_raw_lr["results"][t]["acc,none"] for t in eval_tasks}
    print(eval_results_lr)

    results.append(
        {
            "metric_value": float(k),
            "ppl": ppl_lr,
            "params_ratio": float(params_lr / params_orig),
            "flops_ratio": float(flops_lr / flops_orig),
            "mode": args.mode,
            "seq_len": args.seq_len,
            "eval_results": eval_results_lr,
        }
    )
    model_lr.cpu()

    # Drop all big references
    del lm
    del eval_raw_lr
    del model_lr
    del model_copy

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "After cleanup: allocated",
        torch.cuda.memory_allocated() / 1024**2,
        "MB, reserved",
        torch.cuda.memory_reserved() / 1024**2,
        "MB",
    )
results.append(
    {
        "metric_value": "original",
        "ppl": ppl_orig,
        "params_ratio": 1.0,
        "flops_ratio": 1.0,
        "mode": args.mode,
        "seq_len": args.seq_len,
        "eval_results": eval_results_orig,
    }
)

with open(base_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=lambda o: float(o))
