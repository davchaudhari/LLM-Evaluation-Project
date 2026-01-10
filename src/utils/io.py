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
    
    Converts NaN, Infinity, and -Infinity to None or strings.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (like numpy scalars)
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
