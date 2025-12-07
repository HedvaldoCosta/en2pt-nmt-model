import torch
from transformers import MBartForConditionalGeneration
from src.config import Config

def load_model(device=None, use_bf16=False, use_fp16=False):
    # Escolhe dtype de acordo com flags
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Carrega com dtype quando apropriado (torch_dtype aceita torch.dtype)
    model = MBartForConditionalGeneration.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=dtype if dtype is not None else None
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
