from transformers import AutoTokenizer
from loguru import logger

def get_tokenizer(model_name, use_fast=True):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        return tokenizer
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {e}")
        raise

def tokenize_batch(tokenizer, texts, max_length=256):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

def get_tokenizer_local(tokenizer_dir, use_fast=True):
    # Always load a tokenizer from a local directory, never from HuggingFace Hub
    from pathlib import Path
    if not Path(tokenizer_dir).exists():
        logger.error(f"Local tokenizer directory does not exist: {tokenizer_dir}")
        raise FileNotFoundError(f"Local tokenizer directory does not exist: {tokenizer_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast, local_files_only=True)
        return tokenizer
    except Exception as e:
        logger.error(f"Local tokenizer loading failed: {e}")
        raise
