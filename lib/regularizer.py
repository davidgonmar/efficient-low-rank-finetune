import torch
from lib.polar_express import polar_express


# autograd
class LowRankRegularizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, eps):
        frob = torch.linalg.norm(W, ord="fro") + eps
        UV = polar_express(W)
        nuc = torch.trace(W @ UV.T)
        reg = nuc / frob
        ctx.save_for_backward(W, UV, frob, nuc)
        return reg

    @staticmethod
    def backward(ctx, grad_output):
        W, UV, frob, nuc = ctx.saved_tensors
        G = (UV / frob) - (nuc / (frob**3)) * W
        return grad_output * G, None


def low_rank_reg(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return LowRankRegularizer.apply(W, eps)


@torch.no_grad()
def low_rank_reg_loss(model, eps: float = 1e-12):
    loss = 0.0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None or not isinstance(w, torch.Tensor) or w.dim() < 2:
            continue

        lora_A = getattr(getattr(m, "lora_A", None), "default", None)
        lora_B = getattr(getattr(m, "lora_B", None), "default", None)
        if lora_A is None or lora_B is None:
            continue

        # Actual trainable params (no dtype/device changes)
        A = lora_A.weight
        B = lora_B.weight

        # LoRA scaling
        scaling = 1.0
        if hasattr(m, "scaling"):
            s = getattr(m, "scaling")
            scaling = float(s["default"] if isinstance(s, dict) else s)

        # Effective weight
        W = w + (B @ A) * scaling

        loss = loss + low_rank_reg(W, eps=eps)

    return loss
