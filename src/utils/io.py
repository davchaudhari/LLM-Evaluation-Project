"""I/O utilities for saving results."""

import json
import jsonlines
import math
from pathlib import Path
from typing import List, Dict, Any


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data as JSONL."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(filepath, mode='w') as writer:
        for item in data:
            writer.write(item)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    with jsonlines.open(filepath, mode='r') as reader:
        return list(reader)


def _sanitize_for_json(obj):
    """Recursively sanitize data for JSON serialization.

    Converts NaN/Infinity to None or strings, and unwraps numpy scalars
    (bool_, integer, floating) and ndarrays into native Python types.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_sanitize_for_json(item) for item in obj]

    # Handle numpy scalars and arrays without requiring numpy at import time.
    if hasattr(obj, "item") and callable(getattr(obj, "item")) and hasattr(obj, "dtype"):
        try:
            return _sanitize_for_json(obj.item())
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")) and hasattr(obj, "dtype"):
        try:
            return _sanitize_for_json(obj.tolist())
        except (ValueError, TypeError):
            pass

    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, (int, str)) or obj is None:
        return obj

    if hasattr(obj, "__dict__"):
        try:
            return float(obj)
        except (ValueError, TypeError):
            return str(obj)
    return obj


def save_json(data: Dict[str, Any], filepath: str):
    """Save data as JSON, handling NaN/Infinity values."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # Sanitize data to handle NaN/Infinity
    sanitized_data = _sanitize_for_json(data)
    with open(filepath, 'w') as f:
        json.dump(sanitized_data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
