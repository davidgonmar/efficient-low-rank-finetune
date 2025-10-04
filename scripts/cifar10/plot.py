# plot_compressibility.py
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_run(path, label):
    with open(path, "r") as f:
        data = json.load(f)
    pts = [d for d in data if d.get("metric_value") != "original"]
    orig = next((d for d in data if d.get("metric_value") == "original"), None)
    pts.sort(key=lambda x: x["flops_ratio"])
    return {
        "label": label,
        "flops_x": [p["flops_ratio"] for p in pts] + ([1.0] if orig else []),
        "params_x": [p["params_ratio"] for p in pts] + ([1.0] if orig else []),
        "acc_y": [p["accuracy"] for p in pts] + ([orig["accuracy"]] if orig else []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", required=True, help="format: /path/to/results.json:Label"
    )
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--title", default="ViT compressibility", help="Plot title prefix")
    ap.add_argument("--name", help="Plot name")
    args = ap.parse_args()

    series = []
    for r in args.runs:
        if ":" in r:
            p, lbl = r.split(":", 1)
        else:
            p, lbl = r, Path(r).parent.name
        series.append(load_run(p, lbl))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render to PDF instead of PNG
    plt.figure()
    for s in series:
        plt.plot(s["flops_x"], s["acc_y"], marker="o", label=s["label"])
    plt.xlabel("FLOPs ratio")
    plt.ylabel("Top-1 accuracy")
    plt.title(f"{args.title} vs FLOPs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    flops_path = out_dir / f"compressibility_flops_{args.name}.pdf"
    plt.savefig(flops_path, dpi=200)

    plt.figure()
    for s in series:
        plt.plot(s["params_x"], s["acc_y"], marker="o", label=s["label"])
    plt.xlabel("Parameters ratio")
    plt.ylabel("Top-1 accuracy")
    plt.title(f"{args.title} vs Parameters")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    params_path = out_dir / f"compressibility_params_{args.name}.pdf"
    plt.savefig(params_path, dpi=200)


if __name__ == "__main__":
    main()
