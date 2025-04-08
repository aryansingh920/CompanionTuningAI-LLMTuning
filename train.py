"""
Created on 30/03/2025

@author: Aryan

Filename: train.py

Relative Path: train.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from training.trainmodel import train_model

# Check if MPS is available (macOS with Apple Silicon or AMD GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "openchat/openchat_3.5"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# After loading the tokenizer, add these lines:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


if __name__ == "__main__":
    train_model(tokenizer=tokenizer, device=device, model=model)
