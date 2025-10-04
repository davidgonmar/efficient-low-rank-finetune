"""
This script evaluates low-rank factorization of ResNet models on CIFAR-10.
It sweeps over different compression ratios and records accuracy.
It can use BALF (our proposed method) or other baselines.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from balf.utils import (
    cifar10_mean,
    cifar10_std,
    evaluate_vision_model,
    seed_everything,
    count_model_flops,
    get_all_convs_and_linears,
    make_factorization_cache_location,
)
from balf.factorization.factorize import (
    to_low_rank_activation_aware_auto,
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
    collect_activation_cache,
)
from lib.models import load_model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_name",
    required=True,
    choices=["resnet20", "resnet56", "vit_tiny_patch16_224"],
)
parser.add_argument("--pretrained_path", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--calib_size", type=int, default=1024)
parser.add_argument(
    "--mode",
    default="flops_auto",
    choices=["flops_auto", "params_auto", "energy_act_aware", "energy"],
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = load_model(
    args.model_name,
    pretrained_path=args.pretrained_path,
).to(device)


if args.model_name != "vit_tiny_patch16_224":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
    )

    eval_ds = datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )
    eval_dl = DataLoader(eval_ds, batch_size=512)

    train_ds = datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    subset = torch.utils.data.Subset(
        train_ds, torch.randperm(len(train_ds))[: args.calib_size]
    )
    train_dl = DataLoader(subset, batch_size=256)
    INPUT_SIZE = (1, 3, 32, 32)
else:  # vit_tiny_patch16_224
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    eval_ds = datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )
    eval_dl = DataLoader(eval_ds, batch_size=512)
    train_ds = datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    subset = torch.utils.data.Subset(
        train_ds, torch.randperm(len(train_ds))[: args.calib_size]
    )
    train_dl = DataLoader(subset, batch_size=256)
    INPUT_SIZE = (1, 3, 224, 224)


baseline_metrics = evaluate_vision_model(model, eval_dl)
params_orig = sum(p.numel() for p in model.parameters())
flops_orig = count_model_flops(model, INPUT_SIZE)

print(
    f"[original] loss={baseline_metrics['loss']:.4f} acc={baseline_metrics['accuracy']:.4f} "
    f"params={params_orig} flops_total={flops_orig['total']}"
)


layer_keys = [k for k in get_all_convs_and_linears(model)]


activation_cache = collect_activation_cache(model, train_dl, keys=layer_keys)

base_dir = Path(args.results_dir)
base_dir.mkdir(parents=True, exist_ok=True)

results = []


ratios_comp = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    1.00,
]

#
ratios_energy = [
    0.85,
    0.875,
    0.9,
    0.915,
    0.93,
    0.95,
    0.96,
    0.98,
    0.99999,
]


for k in ratios_comp if args.mode in ["flops_auto", "params_auto"] else ratios_energy:
    if args.mode == "flops_auto" or args.mode == "params_auto":
        model_lr = to_low_rank_activation_aware_auto(
            model,
            activation_cache,
            ratio_to_keep=k,
            inplace=False,
            keys=layer_keys,
            metric="flops" if args.mode == "flops_auto" else "params",
            save_dir=make_factorization_cache_location(
                args.model_name
                + "_"
                + args.pretrained_path.split("/")[-1].split(".")[0],
                args.calib_size,
                "cifar10",
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
                args.model_name
                + "_"
                + args.pretrained_path.split("/")[-1].split(".")[0],
                args.calib_size,
                "cifar10",
                "factorize_sweep",
                args.seed,
            ),
        )
    elif args.mode == "energy":
        model_lr = to_low_rank_manual(
            model,
            cfg_dict={
                kk: {"name": "svals_energy_ratio_to_keep", "value": k}
                for kk in layer_keys
            },
            inplace=False,
        )

    params_lr = sum(p.numel() for p in model_lr.parameters())
    flops_raw_lr = count_model_flops(model_lr, INPUT_SIZE)
    eval_lr = evaluate_vision_model(model_lr.to(device), eval_dl)

    print(
        f"[ratio={k:.6f}] loss={eval_lr['loss']:.4f} acc={eval_lr['accuracy']:.4f} "
        f"params_ratio={params_lr/params_orig:.4f} flops_ratio={flops_raw_lr['total']/flops_orig['total']:.4f}"
    )

    results.append(
        {
            "metric_value": k,
            "loss": float(eval_lr["loss"]),
            "accuracy": float(eval_lr["accuracy"]),
            "params_ratio": float(params_lr / params_orig),
            "flops_ratio": float(flops_raw_lr["total"] / flops_orig["total"]),
            "mode": args.mode,
        }
    )

results.append(
    {
        "metric_value": "original",
        "loss": float(baseline_metrics["loss"]),
        "accuracy": float(baseline_metrics["accuracy"]),
        "params_ratio": 1.0,
        "flops_ratio": 1.0,
        "mode": args.mode,
    }
)
output_file = base_dir / "results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
