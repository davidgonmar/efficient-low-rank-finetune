import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from lib.polar_express import polar_express


model_name = "meta-llama/Llama-2-7b-hf"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=256
    )


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./lora-opt-reg",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    logging_steps=50,
    bf16=torch.cuda.is_available(),
    push_to_hub=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


@torch.no_grad()
def add_lora_nuclear_reg_grads(model, reg_lambda: float, eps: float = 1e-12):
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
        nuc = torch.trace(W @ UV.T)
        G = ((UV / f) - (nuc / (f**3)) * W) * scaling
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


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    callbacks=[LoRaNuclearRegCallback(reg_lambda=1e-1)],
)

trainer.train()
trainer.save_model("./lora-opt-reg-final")
