from pathlib import Path
from datasets import DatasetDict, load_from_disk
import shutil

STORAGE_DIR = Path("/Volumes/workspace/default/numreason")

DATA_DIR = STORAGE_DIR / "data"
OUTPUT_DIR = STORAGE_DIR / "data_split"

VALIDATION_SIZE = 0.1
SEED = 42


def main():
    dataset = load_from_disk(str(DATA_DIR))

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    splits = dataset["train"].train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SEED,
    )

    processed = DatasetDict({
        "train": splits["train"],
        "validation": splits["test"],
        "test": dataset["test"],
    })

    processed.save_to_disk(str(OUTPUT_DIR))

    print(processed)
    print(f"Processed dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
