from datasets import load_dataset, DatasetDict
from src.config import Config

def load_and_prepare_dataset():
    raw = load_dataset(Config.DATASET_NAME, split="train")

    def extract(example):
        return {
            "en": example["translation"]["en"],
            "pt": example["translation"]["pt"]
        }

    processed = raw.map(extract, remove_columns=["translation"])

    split = processed.train_test_split(
        test_size=Config.VAL_SIZE,
        seed=Config.SEED,
        shuffle=True
    )

    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })
