import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.model_loader import load_model
from src.data_loader import load_and_prepare_dataset
from src.preprocessing import preprocess, get_collator
from src.tokenizer import tokenizer
from src.utils import set_seed
from src.config import Config


# Auxiliares: verificação BF16/FP16
def _gpu_supports_bf16():
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

def _gpu_supports_fp16():
    return torch.cuda.is_available()


def train():
    set_seed()

    # 1. Carregar dataset
    ds = load_and_prepare_dataset()
    ds = ds.map(preprocess, batched=True)

    # 2. Determinar dtype ideal
    use_bf16 = Config.USE_BF16 and _gpu_supports_bf16()
    use_fp16 = False if use_bf16 else (Config.USE_FP16 and _gpu_supports_fp16())

    print(f"Precision flags -> use_bf16={use_bf16}, use_fp16={use_fp16}")

    # 3. Carregar modelo com dtype consistente
    model = load_model(use_bf16=use_bf16, use_fp16=use_fp16)

    # 4. Data collator
    collator = get_collator(model)

    # 5. TrainingArguments
    args = Seq2SeqTrainingArguments(
        output_dir=str(Config.CHECKPOINT_DIR),
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM,
        learning_rate=Config.LR,
        warmup_ratio=Config.WARMUP,
        num_train_epochs=Config.EPOCHS,

        fp16=use_fp16,
        bf16=use_bf16,

        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        logging_steps=200,
        save_total_limit=3,
        report_to="none",
    )

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator
    )

    # 7. Treinamento
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(Config.FINAL_MODEL_DIR)


if __name__ == "__main__":
    train()
