from transformers import MBart50TokenizerFast
from src.config import Config

tokenizer = MBart50TokenizerFast.from_pretrained(Config.MODEL_NAME)
tokenizer.src_lang = Config.SRC_LANG
tokenizer.tgt_lang = Config.TGT_LANG
