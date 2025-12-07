from pathlib import Path

class Config:
    # Diretórios principais
    PROJECT_DIR = Path(__file__).parent.parent
    CHECKPOINT_DIR = PROJECT_DIR / "models" / "checkpoints"
    FINAL_MODEL_DIR = PROJECT_DIR / "models" / "final" / "mbart-en-pt-europarl"

    # Dataset e modelo
    DATASET_NAME = "Nadas31/europarl-en-pt-translation"
    MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

    # Línguas
    SRC_LANG = "en_XX"
    TGT_LANG = "pt_XX"

    # Hiperparâmetros
    MAX_LENGTH = 192
    BATCH_SIZE = 8
    GRAD_ACCUM = 2
    LR = 3e-5
    WARMUP = 0.1
    EPOCHS = 3
    SEED = 42
    VAL_SIZE = 10_000
    USE_FP16 = False
    USE_BF16 = True
