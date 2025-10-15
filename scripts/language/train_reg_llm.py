#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import copy
import shutil
import tempfile

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from peft import LoraConfig, get_peft_model

from lib.polar_express import polar_express


def tokenize_function_factory(tokenizer, max_length):
    def fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return fn


@torch.no_grad()
def add_lora_nuclear_reg_grads(model, reg_lambda: float, eps: float = 1e-12):
    """
    Adds gradients corresponding to nuclear-norm regularization for layers that have LoRA adapters.
    Assumes modules may have attributes: weight, lora_A.default, lora_B.default, and optional 'scaling'.
    """
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None or not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue

        lora_A = getattr(getattr(m, "lora_A", None), "default", None)
        lora_B = getattr(getattr(m, "lora_B", None), "default", None)
        if lora_A is None or lora_B is None:
            continue

        A = lora_A.weight.bfloat16()
        B = lora_B.weight.bfloat16()

        scaling = 1.0
        if hasattr(m, "scaling"):
            s = getattr(m, "scaling")
            scaling = float(s["default"] if isinstance(s, dict) else s)

        W = w + (B @ A) * scaling
        f = torch.linalg.norm(W, ord="fro") + eps
        UV = polar_express(W)
        nuc = torch.trace(W @ UV.T.bfloat16())
        G = (((UV / f) - (nuc / (f**3)) * W) * scaling).bfloat16()

        dW = G
        dB = G @ A.T
        dA = B.T @ G

        if w.requires_grad:
            if w.grad is None:
                w.grad = torch.zeros_like(w)
            w.grad.add_(reg_lambda * dW)
        if A.requires_grad:
            if A.grad is None:
                A.grad = torch.zeros_like(A)
            A.grad.add_(reg_lambda * dA)
        if B.requires_grad:
            if B.grad is None:
                B.grad = torch.zeros_like(B)
            B.grad.add_(reg_lambda * dB)


class LoRaNuclearRegCallback(TrainerCallback):
    def __init__(self, reg_lambda: float = 1e-1, eps: float = 1e-12):
        self.reg_lambda = reg_lambda
        self.eps = eps

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        add_lora_nuclear_reg_grads(model, self.reg_lambda, self.eps)
        return control


class SaveLatestMergedCallback(TrainerCallback):
    """
    Overwrite `merged_output_dir` with the latest merged model AND tokenizer on each save trigger.
    No step subfolders. Uses temp dir + atomic replace to avoid partial writes.
    """

    def __init__(
        self,
        merged_output_dir: str,
        save_steps: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.merged_output_dir = merged_output_dir
        self.save_steps = save_steps
        self.tokenizer = tokenizer
        os.makedirs(self.merged_output_dir, exist_ok=True)

    def _save_merged_copy(self, peft_model):
        # Work on a CPU clone so we don't mutate the live training model
        clone = copy.deepcopy(peft_model).to("cpu")
        merged = clone.merge_and_unload()  # merges LoRA into base weights on the clone

        parent = os.path.dirname(os.path.abspath(self.merged_output_dir)) or "."
        with tempfile.TemporaryDirectory(dir=parent) as tmpdir:
            # Write model + tokenizer into the same temp dir
            merged.save_pretrained(tmpdir)
            self.tokenizer.save_pretrained(tmpdir)

            # Replace target dir atomically (on same filesystem)
            if os.path.exists(self.merged_output_dir):
                shutil.rmtree(self.merged_output_dir)
            os.rename(tmpdir, self.merged_output_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if (
            self.save_steps > 0
            and state.global_step > 0
            and state.global_step % self.save_steps == 0
        ):
            self._save_merged_copy(kwargs["model"])
            print(
                f"[save] wrote latest merged model + tokenizer to {self.merged_output_dir}"
            )
        return control


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3-1b")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="./lora-opt-reg")
    p.add_argument("--merged_output_dir", type=str, default="./lora-merged-final")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--reg_lambda", type=float, default=1e-1)
    p.add_argument("--reg_eps", type=float, default=1e-12)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    # Tokenizer: use fast + explicit legacy=False to avoid legacy warning
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        legacy=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = dataset.map(
        tokenize_function_factory(tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )

    model = get_peft_model(model, peft_config)

    # Disable HF's own checkpoint saving (we'll save only merged via our callback)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_strategy="no",  # no adapter checkpoints
        save_steps=args.save_steps,  # used only by our callback
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        push_to_hub=args.push_to_hub,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[
            LoRaNuclearRegCallback(reg_lambda=args.reg_lambda, eps=args.reg_eps),
            SaveLatestMergedCallback(
                merged_output_dir=args.merged_output_dir,
                save_steps=args.save_steps,
                tokenizer=tokenizer,
            ),
        ],
    )

    trainer.train()

    # Final merged save (overwrite same dir, no step suffix)
    merged = copy.deepcopy(model).merge_and_unload()
    merged.save_pretrained(args.merged_output_dir)
    tokenizer.save_pretrained(args.merged_output_dir)
    print(f"[final] wrote final merged model + tokenizer to {args.merged_output_dir}")


if __name__ == "__main__":
    main()
