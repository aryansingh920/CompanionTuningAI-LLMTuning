from training.loadpreprocess import load_and_preprocess_datasets

from transformers import (

    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)



def train_model(tokenizer,device,model):
    """Train the model on the processed datasets"""
    # Load and preprocess all datasets
    combined_dataset = load_and_preprocess_datasets(tokenizer)
    print(f"Combined dataset size: {len(combined_dataset)} examples")

    # Data collator for efficient batch processing
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM does not use masked language modeling
    )

    # Define training arguments with appropriate settings for your hardware
    training_args = TrainingArguments(
        output_dir="./tone_adaptive_chatbot",
        per_device_train_batch_size=2,  # Adjust based on your GPU/MPS memory
        gradient_accumulation_steps=16,  # Increase for lower memory usage
        evaluation_strategy="no",  # No evaluation during training
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=1,
        save_total_limit=2,  # Only keep the 2 most recent checkpoints
        report_to="none",  # Disable wandb and other reporting
        push_to_hub=False,
        fp16=False,  # Disable mixed precision if causing issues on MPS
        use_mps_device=True if device == "mps" else False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=data_collator
    )

    # Start fine-tuning
    print("Starting model training...")
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("./fine_tuned_tone_adaptive_model")
    tokenizer.save_pretrained("./fine_tuned_tone_adaptive_model")

    print("Model fine-tuning complete. Saved to './fine_tuned_tone_adaptive_model'")
