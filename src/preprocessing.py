from src.tokenizer import tokenizer
from transformers import DataCollatorForSeq2Seq
from src.config import Config

def preprocess(batch):
    inputs = tokenizer(
        batch["en"],
        max_length=Config.MAX_LENGTH,
        truncation=True
    )
    targets = tokenizer(
        text_target=batch["pt"],
        max_length=Config.MAX_LENGTH,
        truncation=True
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def get_collator(model):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt"
    )
