"""File and data loading utilities."""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json_directory(directory_path: str) -> List[Dict[str, Any]]:
    """Load all JSON files from a directory."""
    data = []
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    for json_file in directory.glob("*.json"):
        try:
            data.append(load_json(str(json_file)))
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return data


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)
