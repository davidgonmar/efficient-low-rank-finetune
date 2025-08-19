import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from transformers import Trainer

model_name = "facebook/opt-350m"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=256
    )


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./lora-opt-reg",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=50,
    bf16=torch.cuda.is_available(),
    push_to_hub=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def uv_fn(M):
    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / torch.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = torch.eye(A.shape[0], device=M.device, dtype=M.dtype)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


class LoRaRegFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W_hat, A, B, scale, eps=1e-12):
        A, B = A.T, B.T
        W = W_hat + A.matmul(B) * scale
        frob = W.norm(p="fro")
        UV = uv_fn(W)
        nuc = torch.trace(W @ UV.T)
        ctx.save_for_backward(W, A, B, UV, frob, nuc)
        ctx.scale = scale
        ctx.eps = eps
        return nuc / (frob)

    @staticmethod
    def backward(ctx, grad_output):
        W, A, B, UV, frob, nuc = ctx.saved_tensors
        scale = ctx.scale
        f = frob + ctx.eps
        G = (UV) / f - (nuc / (f**3)) * W
        G = grad_output * G
        dW_hat = G
        dA = G @ B.t() * scale
        dB = A.t() @ G * scale
        return dW_hat, dA.T, dB.T, None, None, None


class PenaltyTrainer(Trainer):
    def __init__(self, *args, reg_lambda: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_lambda = reg_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        penalty = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

        for m in model.modules():
            # break
            w = getattr(m, "weight", None)
            if w is None or not isinstance(w, torch.Tensor) or w.dim() < 2:
                continue
            merged = w
            A = B = None
            scaling = 1.0
            # print(m)
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                adapter = getattr(m, "active_adapter", None)
                if adapter is None and hasattr(model, "active_adapter"):
                    adapter = getattr(model, "active_adapter", None)
                if adapter is not None:
                    A = m.lora_A.default.weight
                    B = m.lora_B.default.weight
                    scaling = float(getattr(m, "scaling", 1.0)["default"])

            if A is not None and B is not None:
                # print(w.dtype, A.dtype, B.dtype)
                penalty = penalty + LoRaRegFn.apply(merged, A, B, scaling)

        total_loss = base_loss + self.reg_lambda * penalty

        if self.state is not None and self.is_world_process_zero():
            self.log(
                {
                    "ce_loss": base_loss.detach().float().item(),
                    "reg_loss": penalty.detach().float().item(),
                    "total_loss": total_loss.detach().float().item(),
                }
            )

        if return_outputs:
            outputs = {"ce_loss": base_loss.detach(), "reg_loss": penalty.detach()}
            return total_loss, outputs

        return total_loss


trainer = PenaltyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    reg_lambda=1e-1,
)

trainer.train()
trainer.save_model("./lora-opt-reg-final")
