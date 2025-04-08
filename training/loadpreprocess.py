
from datasets import load_dataset, concatenate_datasets

from training.dataset_config import (
    load_all_therapeutic_datasets,
    balanced_datasets,
    casual_datasets
)

# Import preprocessing functions
from training.preprocessing import (
    preprocess_and_tokenize_dataset,
)


def load_and_preprocess_datasets(tokenizer):
    """Load all datasets and preprocess them based on their type"""
    print("Loading and preprocessing datasets...")

    # Load therapeutic datasets
    print("Processing therapeutic datasets...")
    therapeutic_datasets = load_all_therapeutic_datasets()
    tokenized_therapeutic = []

    for i, dataset in enumerate(therapeutic_datasets):
        first_split = list(dataset.keys())[0]
        print(
            f"Processing therapeutic dataset {i+1}/{len(therapeutic_datasets)}, split: {first_split}, size: {len(dataset[first_split])}")
        try:
            tokenized_dataset = preprocess_and_tokenize_dataset(
                dataset[first_split],
                "therapeutic",
                tokenizer
            )
            print(
                f"Successfully processed therapeutic dataset with {len(tokenized_dataset)} examples")
            tokenized_therapeutic.append(tokenized_dataset)
        except Exception as e:
            print(f"Error processing therapeutic dataset {i+1}: {e}")
            import traceback
            traceback.print_exc()

    # Load and process balanced datasets
    print("Processing balanced datasets...")
    tokenized_balanced = []
    for i, dataset_name in enumerate(balanced_datasets):
        try:
            dataset = load_dataset(dataset_name)
            first_split = list(dataset.keys())[0]
            print(
                f"Processing balanced dataset {i+1}/{len(balanced_datasets)}: {dataset_name}, split: {first_split}, size: {len(dataset[first_split])}")
            tokenized_dataset = preprocess_and_tokenize_dataset(
                dataset[first_split],
                "balanced",
                tokenizer
            )
            print(
                f"Successfully processed balanced dataset with {len(tokenized_dataset)} examples")
            tokenized_balanced.append(tokenized_dataset)
        except Exception as e:
            print(f"Error processing balanced dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Load and process casual datasets
    print("Processing casual datasets...")
    tokenized_casual = []
    for i, dataset_name in enumerate(casual_datasets):
        try:
            dataset = load_dataset(dataset_name)
            first_split = list(dataset.keys())[0]
            print(
                f"Processing casual dataset {i+1}/{len(casual_datasets)}: {dataset_name}, split: {first_split}, size: {len(dataset[first_split])}")
            tokenized_dataset = preprocess_and_tokenize_dataset(
                dataset[first_split],
                "casual",
                tokenizer
            )
            print(
                f"Successfully processed casual dataset with {len(tokenized_dataset)} examples")
            tokenized_casual.append(tokenized_dataset)
        except Exception as e:
            print(f"Error processing casual dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Combine all tokenized datasets
    all_tokenized_datasets = tokenized_therapeutic + \
        tokenized_balanced + tokenized_casual
    all_tokenized_datasets = [
        ds for ds in all_tokenized_datasets if len(ds) > 0]

    if not all_tokenized_datasets:
        raise ValueError("No valid datasets were processed. Check for errors.")

    print("Concatenating all datasets...")
    combined_dataset = concatenate_datasets(all_tokenized_datasets)

    return combined_dataset
