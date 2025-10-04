import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="vit_tiny_cifar10_pretrain.pt")
    return p.parse_args()


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

    train_ds = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_tf
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=10).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run += loss.item()
            if i % 50 == 0:
                print(f"epoch {epoch} iter {i} loss {run/50:.4f}")
                run = 0.0
        print(f"epoch {epoch} time {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
