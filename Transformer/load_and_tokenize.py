from datasets import Dataset
import os
DATASET_PATH = "parallel_en_fr_corpus"

# Directly load from the .arrow files (bypass broken state.json)
train_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "train", "dataset.arrow"))
val_dataset   = Dataset.from_file(os.path.join(DATASET_PATH, "validation", "dataset.arrow"))
test_dataset  = Dataset.from_file(os.path.join(DATASET_PATH, "test", "dataset.arrow"))

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print("Train columns:", train_dataset.column_names)  # Should show ['text_en', 'text_fr']