import argparse
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description="LoRA SFT Training")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="qwen")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "allenai/tulu-3-sft-mixture"
OUTPUT_DIR = args.output_dir
MAX_LENGTH = 256

LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    seed=args.seed,
    learning_rate=args.lr,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    logging_steps=100,
    warmup_steps=100,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=5,
    save_only_model=True,
    report_to="wandb",
    bf16=True
)

# ===============================================
# Model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)
model = get_peft_model(model, LORA_CONFIG)

def count_parameters(model):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")

# ===============================================
# Dataset 

def preprocess(example):
    text = text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
    )
    tok = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok


train_raw = load_dataset(DATASET_NAME, split="train[:5%]")
test_raw = load_dataset(DATASET_NAME, split="train[5%:6%]")
tokenized_tr = train_raw.map(
    preprocess,
    remove_columns=train_raw.column_names,
    num_proc=8,
    batched=True
)
tokenized_ts = test_raw.map(
    preprocess,
    remove_columns=test_raw.column_names,
    num_proc=8,
    batched=True
)

# ===============================================
# Trainer

trainer = Trainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=tokenized_tr,
    eval_dataset=tokenized_ts,
    data_collator=None,
)

trainer.train()
trainer.evaluate()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
