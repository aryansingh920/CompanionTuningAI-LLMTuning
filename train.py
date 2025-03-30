"""
Created on 16/03/2025

@author: Aryan

Filename: main.py

Relative Path: main.py
"""

import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Check if MPS is available (macOS with Apple Silicon or AMD GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Choose a smaller pretrained model
# Using a smaller model (125M instead of 1.3B)
MODEL_NAME = "EleutherAI/gpt-neo-125M"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# After loading the tokenizer, add these lines:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Load therapy datasets
datasets_list = [
    "nbertagnolli/counsel-chat",
    # "facebook/EmpatheticDialogues",
]


def preprocess_function(examples):
    """ Tokenize text data for GPT fine-tuning """
    # Ensure lists are converted into strings
    questions = [q if isinstance(
        q, str) else "" for q in examples["questionText"]]
    answers = [a if isinstance(a, str) else "" for a in examples["answerText"]]

    # Concatenate question and answer pairs
    inputs = [q + " " + a for q, a in zip(questions, answers)]

    # Reduced max_length
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=256)

# Load and preprocess datasets


def load_and_tokenize_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    print(
        f"Dataset: {dataset_name} - Columns: {dataset['train'].column_names}")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset


# Process all datasets
train_datasets = [load_and_tokenize_dataset(ds) for ds in datasets_list]
train_dataset = datasets.concatenate_datasets(
    [ds["train"] for ds in train_datasets])

# Data collator for efficient batch processing
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM does not use masked language modeling
)

# Define training arguments with reduced batch size and gradient accumulation
training_args = TrainingArguments(
    output_dir="./therapy_gpt",
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=16,  # Increased gradient accumulation
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_total_limit=2,
    report_to="none",
    push_to_hub=False,
    fp16=False,  # Disable mixed precision training if causing issues
    use_mps_device=True if device == "mps" else False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Start fine-tuning
trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_therapy_gpt")
tokenizer.save_pretrained("./fine_tuned_therapy_gpt")

print("Model fine-tuning complete. Saved to './fine_tuned_therapy_gpt'")
