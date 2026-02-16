#usr/bin/env python3
"""
Module for loading JSON files or JSON strings and converting their contents to a
dictionary
"""

import json
import numpy as np

def load_json(json_file: str) -> dict:
    """
    Load a JSON file or a JSON string and return its contents as a dictionary.

    str json_file: Path to the JSON file or JSON string
    return: Dictionary containing the JSON data
    """
    data = {}
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            for key, value in json.load(f).items():
                data[key] = value
        return data
    except FileNotFoundError:
        try:
            for key, value in json.loads(json_file).items():
                data[key] = value
            return data
        except Exception as e:
            raise FileNotFoundError(f"JSON file \"{json_file}\" not found and "
                                    "input is not valid JSON string."
                                    ) from e