import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from lib.polar_express import polar_express


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--reg_lambda", type=float, default=1e-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="vit_tiny_cifar10_nuclear_last.pt")
    p.add_argument(
        "--pretrained_path", type=str, default="vit_tiny_cifar10_pretrain.pt"
    )
    p.add_argument("--samples_per_epoch", type=int, default=1000)
    return p.parse_args()


@torch.no_grad()
def add_nuclear_reg_grads(m: nn.Module, reg_lambda: float, eps: float = 1e-12):
    ret = {}
    for name, mod in m.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue
        W = w.float()
        orig_shape = W.shape
        if W.dim() == 4:
            W = W.view(W.size(0), -1)
        f = torch.linalg.norm(W, ord="fro") + eps
        UV = polar_express(W)
        nuc = torch.trace(W @ UV.T)
        G = (UV / f) - (nuc / (f**3)) * W
        if w.requires_grad:
            if w.grad is None:
                w.grad = torch.zeros_like(w)
            if len(orig_shape) == 4:
                G = G.view(orig_shape)
            w.grad.add_(reg_lambda * G.to(w.dtype))
        ret[name] = (nuc / f).item()
    return ret


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.float32

    train_tf = transforms.Compose(
        [
            transforms.Resize(
                args.image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(
                args.image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_tf
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=10).to(
        device
    )
    model.load_state_dict(torch.load(args.pretrained_path))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    val_loss, val_acc = evaluate(model, test_loader, device, amp_dtype)
    print(f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        seen = 0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            ret = add_nuclear_reg_grads(model, args.reg_lambda)
            scaler.step(optimizer)
            scaler.update()
            run += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch} iter {i} loss {run/50:.4f}")
                run = 0.0
                print(sum(ret.values()) / len(ret))
            seen += x.size(0)
            if args.samples_per_epoch > 0 and seen > args.samples_per_epoch:
                seen = 0
                break
        val_loss, val_acc = evaluate(model, test_loader, device, amp_dtype)
        print(
            f"epoch {epoch} val_loss {val_loss:.4f} val_acc {val_acc:.4f} time {time.time()-t0:.1f}s regloss {sum(ret.values()) / len(ret):.4f}"
        )
    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
