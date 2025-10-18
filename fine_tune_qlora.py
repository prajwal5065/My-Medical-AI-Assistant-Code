import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
DATA_FILE = "dataset.json"
OUTPUT_DIR = "./results"
# This is the folder that will be created
SAVE_MODEL_PATH = "./fine_tuned_model"

# --- 0. Initial Checks ---
if not os.path.exists(DATA_FILE):
    print(f"Error: Dataset file '{DATA_FILE}' not found.")
    sys.exit(1)

# --- 1. Load Dataset ---
print(f"Loading dataset from {DATA_FILE}...")
dataset = load_dataset("json", data_files=DATA_FILE)

# --- 2. Load Model and Tokenizer (with QLoRA Config) ---
print(f"Loading model {MODEL_NAME} with 4-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# --- 3. Tokenize Dataset ---
def tokenize(batch):
    inputs = [f"User: {i}\nAssistant: {r}" for i, r in zip(batch["instruction"], batch["response"])]
    tokens = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# --- 4. PEFT (LoRA) Configuration ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- 5. Training Setup ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    save_strategy="steps",
    save_total_limit=1,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# --- 6. Run Training and Save ---
print("🚀 Starting training...")
trainer.train()

print("✅ Training finished. Saving model...")
model.save_pretrained(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)
print(f"✅ Fine-tuning complete and model saved to {SAVE_MODEL_PATH}!")