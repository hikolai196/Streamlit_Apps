import json
import os
from pathlib import Path

from eda_agent.config.loader import PROJECT_ROOT

SESSION_DIR = PROJECT_ROOT / "sessions"


def list_sessions():
    if not SESSION_DIR.exists():
        SESSION_DIR.mkdir(parents=True)
    return [f.replace(".json", "") for f in os.listdir(SESSION_DIR)]

def load_session(name):
    path = SESSION_DIR / f"{name}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_session(name, messages):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSION_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
