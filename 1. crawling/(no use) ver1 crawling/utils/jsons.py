import json
import os

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    directory = os.path.dirname(path)
    if directory:                       
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)