from datasets import load_dataset
import os
import pandas as pd


def clean_and_load_empathetic_dialogues(data_dir):
    # Clean the CSV files
    for split in ["train", "valid", "test"]:
        file_path = f"{data_dir}/{split}.csv"
        clean_path = f"{data_dir}/{split}_clean.csv"

        try:
            # Read with error handling - using newer pandas parameter
            df = pd.read_csv(file_path, on_bad_lines='skip')
            # Write back clean version
            df.to_csv(clean_path, index=False)
            print(f"Successfully cleaned {split}.csv")
        except Exception as e:
            print(f"Error cleaning {split}.csv: {e}")

    # Check if cleaned files exist before loading
    if all(os.path.exists(f"{data_dir}/{split}_clean.csv") for split in ["train", "valid", "test"]):
        # Now load the cleaned files
        return load_dataset(
            "csv",
            data_files={
                "train": f"{data_dir}/train_clean.csv",
                "validation": f"{data_dir}/valid_clean.csv",
                "test": f"{data_dir}/test_clean.csv"
            }
        )
    else:
        raise FileNotFoundError("Cleaned files were not created successfully")
    
def load_local_empathetic_dialogues(data_dir):
    """Load the manually downloaded EmpatheticDialogues dataset from local CSVs."""
    return load_dataset(
        "csv",
        data_files={
            "train": f"{data_dir}/train.csv",
            "validation": f"{data_dir}/valid.csv",
            "test": f"{data_dir}/test.csv"
        }
    )


# Therapeutic Hugging Face dataset names
therapeutic_dataset_names = [
    "nbertagnolli/counsel-chat",
    "bdotloh/empathetic-dialogues-contexts",
    "Estwld/empathetic_dialogues_llm"
]


def load_all_therapeutic_datasets():
    """Load all therapeutic datasets including local and HF-hosted ones."""
    datasets_list = []

    # Load Hugging Face hosted datasets
    for name in therapeutic_dataset_names:
        try:
            datasets_list.append(load_dataset(name))
        except Exception as e:
            print(f"Error loading {name}: {e}")

    # Load local dataset
    try:
        local = clean_and_load_empathetic_dialogues(
            "datasets/empatheticdialogues")
        datasets_list.append(local)
    except Exception as e:
        print(f"Error loading local EmpatheticDialogues: {e}")

    return datasets_list


# Balanced and casual datasets (names only — you load them in train.py)
balanced_datasets = [
    "ParlAI/blended_skill_talk"
]

casual_datasets = [
    "roskoN/dailydialog",
    "AlekseyKorshuk/persona-chat"
]
