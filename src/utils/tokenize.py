"""Tokenization utilities."""

from typing import List, Optional
from transformers import AutoTokenizer


_tokenizer_cache = {}


def get_tokenizer(model_id: str, cache_dir: Optional[str] = None):
    """Get or create tokenizer (cached)."""
    if model_id not in _tokenizer_cache:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        _tokenizer_cache[model_id] = tokenizer
    return _tokenizer_cache[model_id]


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=True))
