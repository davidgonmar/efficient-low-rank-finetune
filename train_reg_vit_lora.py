import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from peft import LoraConfig, get_peft_model
from lib.polar_express import polar_express


class Cfg:
    model_name = "vit_tiny_patch16_224"
    image_size = 32
    batch_size = 128
    num_workers = 4
    epochs = 20
    lr = 5e-4
    wd = 0.05
    reg_lambda = 1e-1
    seed = 42


cfg = Cfg()
torch.manual_seed(cfg.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float16 if device == "cuda" else torch.float32

train_tf = transforms.Compose(
    [
        transforms.Resize(
            cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomCrop(cfg.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
test_tf = transforms.Compose(
    [
        transforms.Resize(
            cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

train_ds = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_tf
)
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=cfg.batch_size * 2,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

base = timm.create_model(cfg.model_name, pretrained=True, num_classes=10)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["qkv", "proj", "fc1", "fc2"],
    task_type="FEATURE_EXTRACTION",
)
model = get_peft_model(base, peft_config).to(device)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.wd
)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


@torch.no_grad()
def add_lora_nuclear_reg_grads(m: nn.Module, reg_lambda: float, eps: float = 1e-12):
    for mod in m.modules():
        w = getattr(mod, "weight", None)
        if w is None or not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue
        lora_A = getattr(getattr(mod, "lora_A", None), "default", None)
        lora_B = getattr(getattr(mod, "lora_B", None), "default", None)
        if lora_A is None or lora_B is None:
            continue
        A = lora_A.weight.float()
        B = lora_B.weight.float()
        scaling = 1.0
        if hasattr(mod, "scaling"):
            s = getattr(mod, "scaling")
            scaling = float(s["default"] if isinstance(s, dict) else s)
        W = w.float() + (B @ A) * scaling
        f = torch.linalg.norm(W, ord="fro") + eps
        UV = polar_express(W)
        nuc = torch.trace(W @ UV.T)
        G = ((UV / f) - (nuc / (f**3)) * W) * scaling
        dB = G @ A.T
        dA = B.T @ G
        if w.requires_grad:
            if w.grad is None:
                w.grad = torch.zeros_like(w)
            w.grad.add_(reg_lambda * G.to(w.dtype))
        if lora_A.weight.requires_grad:
            if lora_A.weight.grad is None:
                lora_A.weight.grad = torch.zeros_like(lora_A.weight)
            lora_A.weight.grad.add_(reg_lambda * dA.to(lora_A.weight.dtype))
        if lora_B.weight.requires_grad:
            if lora_B.weight.grad is None:
                lora_B.weight.grad = torch.zeros_like(lora_B.weight)
            lora_B.weight.grad.add_(reg_lambda * dB.to(lora_B.weight.dtype))


def evaluate(model, loader):
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


best = 0.0
for epoch in range(1, cfg.epochs + 1):
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
        scaler.unscale_(optimizer)
        add_lora_nuclear_reg_grads(model, cfg.reg_lambda)
        scaler.step(optimizer)
        scaler.update()
        run += loss.item()
        if i % 50 == 0:
            print(f"epoch {epoch} iter {i} loss {run/50:.4f}")
            run = 0.0
    val_loss, val_acc = evaluate(model, test_loader)
    print(
        f"epoch {epoch} val_loss {val_loss:.4f} val_acc {val_acc:.4f} time {time.time()-t0:.1f}s"
    )
    if val_acc > best:
        best = val_acc
        torch.save(model.state_dict(), "timm_vit_tiny_cifar10_peftlora_best.pt")
torch.save(model.state_dict(), "timm_vit_tiny_cifar10_peftlora_last.pt")
