"""
Created on 30/03/2025

@author: Aryan

Filename: train.py
Relative Path: train.py
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from training.trainmodel import train_model

# Check if MPS is available (macOS with Apple Silicon or AMD GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
print(f"Using device: {device}")

MODEL_NAME = "EleutherAI/gpt-neo-125m"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Ensure we have a valid pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure LoRA
lora_config = LoraConfig(
    r=8,                   # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # typical modules to apply LoRA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the original model with LoRA adapters
model = get_peft_model(model, lora_config)

if __name__ == "__main__":
    # Train the model (passes tokenizer, device, and model into train_model)
    train_model(tokenizer=tokenizer, device=device, model=model)
