"""
trainmodel.py

Trains the model on the processed dataset via load_and_preprocess_datasets.
"""

import torch
from training.loadpreprocess import load_and_preprocess_datasets
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers import EarlyStoppingCallback


def train_model(tokenizer, device, model):
    """Train the model on the processed datasets."""
    # Load and preprocess all datasets using your custom function
    combined_dataset = load_and_preprocess_datasets(tokenizer)
    print(f"Combined dataset size: {len(combined_dataset)} examples")

    # Split the dataset into train and eval
    # You can adjust the split ratio as needed
    train_size = int(0.9 * len(combined_dataset))
    eval_size = len(combined_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, eval_size]
    )

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Evaluation dataset size: {len(eval_dataset)} examples")

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM does not use masked LM
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./tone_adaptive_chatbot",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,  # Add this line
        gradient_accumulation_steps=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=5,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        fp16=False,
        use_mps_device=(device.type == "mps"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create a Trainer instance with both train and eval datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add this line
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Start fine-tuning
    print("Starting model training...")
    trainer.train()

    # Save fine-tuned model & tokenizer
    model.save_pretrained("./fine_tuned_tone_adaptive_model")
    tokenizer.save_pretrained("./fine_tuned_tone_adaptive_model")

    print("Model fine-tuning complete. Saved to './fine_tuned_tone_adaptive_model'")
