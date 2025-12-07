from src.model_loader import load_model
from src.tokenizer import tokenizer
from src.config import Config
import torch

def translate(text):
    model = load_model(fp16=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[Config.TGT_LANG]
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
