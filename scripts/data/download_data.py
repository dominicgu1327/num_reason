from pathlib import Path
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download

STORAGE_DIR = Path("/Volumes/workspace/default/numreason")

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
MODEL_ID = "Qwen/Qwen3.5-0.8B"

DATA_DIR = STORAGE_DIR / "data"
MODEL_DIR = STORAGE_DIR / "models" / "Qwen3.5-0.8B"


def main():
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if (DATA_DIR / "dataset_dict.json").is_file():
        dataset = load_from_disk(str(DATA_DIR))
        print("Dataset already exists.")
    else:
        print(f"Downloading dataset: {DATASET_ID} ({DATASET_CONFIG})")
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
        dataset.save_to_disk(str(DATA_DIR))

    print(dataset)
    print(f"Dataset saved to: {DATA_DIR}")

    print(f"\nDownloading model: {MODEL_ID}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
    )

    print(f"Model saved to: {MODEL_DIR}")
    print("\nAll downloads completed successfully.")


if __name__ == "__main__":
    main()