import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import PeftModel
from tqdm import tqdm

parser = argparse.ArgumentParser(description="LoRA Model Soup Evaluation")
parser.add_argument("--path1", type=str, required=True)
parser.add_argument("--path2", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="merged_model")
args = parser.parse_args()

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "allenai/tulu-3-sft-mixture"
MAX_LENGTH = 256

# Load base model and tokenizer
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    trust_remote_code=True
)
base_model2 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    trust_remote_code=True
)
model1 = PeftModel.from_pretrained(base_model, args.path1)
model2 = PeftModel.from_pretrained(base_model2, args.path2)

# Merge LoRA weights
print(f"Merging models")
merged_state_dict = {}

state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()

for key in tqdm(state_dict1.keys()):
    if "lora" in key:
        merged_state_dict[key] = 0.5 * (state_dict1[key] + state_dict2[key])

# Load merged weights into model1
model1.load_state_dict(merged_state_dict, strict=False)
merged_model = model1
merged_model = merged_model.merge_and_unload()  # Merge LoRA weights into base model

# Evaluate
def preprocess(example):
    text = tokenizer.apply_chat_template(
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

test_raw = load_dataset(DATASET_NAME, split="train[5%:6%]")
tokenized_ts = test_raw.map(
    preprocess,
    remove_columns=test_raw.column_names,
    num_proc=8,
    batched=True
)
eval_args = TrainingArguments(
    output_dir=None,
    per_device_eval_batch_size=4,
    bf16=True,
    report_to="none"
)
trainer = Trainer(
    model=merged_model,
    args=eval_args,
    eval_dataset=tokenized_ts,
    data_collator=None,
)

print("Evaluating merged model...")
results = trainer.evaluate()
print(results)
for key, value in results.items():
    print(f"{key}: {value}")

savename = "merge_loss.json"
with open(os.path.join(args.path1, savename), "w") as f:
    json.dump(results, f)
