import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import wandb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="vit_tiny_cifar10_pretrain.pt")
    p.add_argument("--wandb_project", type=str, default="cifar10-vit-pretrain")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        total = 0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)
        train_loss = loss_sum / total
        val_loss, val_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch} train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} time {elapsed:.1f}s"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time": elapsed,
            }
        )
    torch.save(model.state_dict(), args.save_path)
    wandb.finish()


if __name__ == "__main__":
    main()
