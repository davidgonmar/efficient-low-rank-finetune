"""
This script plots, for several layers, the normalized cumulative spectral energy
as a function of rank for an original and a regularized model.
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from balf.utils import seed_everything, get_all_convs_and_linears
from lib.models import load_model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_name",
    required=True,
    choices=["resnet20", "resnet56", "vit_tiny_patch16_224"],
)
parser.add_argument("--pretrained_orig", required=True)
parser.add_argument("--pretrained_reg", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--indices",
    type=int,
    nargs="*",
    help="optional layer indices; if omitted, layers are chosen evenly across depth",
)
args = parser.parse_args()
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_orig = load_model(
    args.model_name,
    pretrained_path=args.pretrained_orig,
).to(device)
model_orig.eval()

model_reg = load_model(
    args.model_name,
    pretrained_path=args.pretrained_reg,
).to(device)
model_reg.eval()

raw_layer_names_orig = get_all_convs_and_linears(model_orig)
raw_layer_names_reg = get_all_convs_and_linears(model_reg)

layer_names_orig = [n for n in raw_layer_names_orig if "head" not in n]
layer_names_reg = [n for n in raw_layer_names_reg if "head" not in n]

if len(layer_names_orig) != len(layer_names_reg):
    raise RuntimeError(
        "mismatch in number of conv/linear layers between models after filtering"
    )

named_modules_orig = dict(model_orig.named_modules())
named_modules_reg = dict(model_reg.named_modules())

n_layers_total = len(layer_names_orig)

if args.indices and len(args.indices) > 0:
    indices = [i for i in args.indices if 0 <= i < n_layers_total]
else:
    num_to_pick = min(8, n_layers_total)
    positions = torch.linspace(0, n_layers_total - 1, steps=num_to_pick)
    indices = []
    for p in positions:
        idx = int(round(p.item()))
        if idx not in indices:
            indices.append(idx)
    indices = indices[:num_to_pick]

if len(indices) == 0:
    raise RuntimeError("no valid layer indices selected")

base_dir = Path(args.results_dir)
base_dir.mkdir(parents=True, exist_ok=True)


def energy_curve(layer):
    w = layer.weight.detach().cpu()
    w = w.view(w.shape[0], -1)
    s = torch.linalg.svdvals(w.to(torch.float32))
    energy = s**2
    total = energy.sum()
    cumulative = torch.cumsum(energy, dim=0) / total
    ranks = torch.arange(1, s.shape[0] + 1)
    return ranks.numpy(), cumulative.numpy()


rows = 2
cols = 4

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)

fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
axes_flat = list(axes.reshape(-1))

last_plot_idx = -1

for plot_idx, layer_idx in enumerate(indices):
    if plot_idx >= len(axes_flat):
        break
    name = layer_names_orig[layer_idx]
    if name not in named_modules_orig or name not in named_modules_reg:
        raise RuntimeError(f"layer name {name} not found in model modules")
    layer_orig = named_modules_orig[name]
    layer_reg = named_modules_reg[name]

    r_orig, e_orig = energy_curve(layer_orig)
    r_reg, e_reg = energy_curve(layer_reg)

    m = min(len(r_orig), len(r_reg))
    r = r_orig[:m]

    ax = axes_flat[plot_idx]
    ax.plot(r, e_orig[:m], label="Original")
    ax.plot(r, e_reg[:m], label="Regularized")
    ax.set_title(name)
    if plot_idx % cols == 0:
        ax.set_ylabel("Energy retained")
    if plot_idx >= (rows - 1) * cols:
        ax.set_xlabel("Rank")
    ax.set_xlim(1, r[m - 1])
    ax.set_ylim(0.0, 1.01)
    last_plot_idx = plot_idx

for j in range(last_plot_idx + 1, len(axes_flat)):
    fig.delaxes(axes_flat[j])

handles, labels = axes_flat[0].get_legend_handles_labels()
if handles:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )

fig.suptitle(f"{args.model_name}: singular value spectra (original vs regularized)")
fig.tight_layout(rect=[0, 0.05, 1, 0.96])

out_path = base_dir / f"svals_{args.model_name}.pdf"
fig.savefig(out_path, bbox_inches="tight")

print(f"saved plot to {out_path}")
