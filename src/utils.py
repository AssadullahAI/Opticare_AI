# src/utils.py

import os
from config.settings import Config

def read_text_data():
    texts = {}
    for file in os.listdir(Config.DATA_DIR):
        if file.endswith(".txt"):
            key = file.replace(".txt", "")
            with open(Config.DATA_DIR / file, "r", encoding="utf-8") as f:
                texts[key] = f.read()
    return texts
from config.settings import Config

class EmergencyDetector:
    @staticmethod
    def is_emergency(text: str):
        text = text.lower()
        for keyword in Config.EMERGENCY_KEYWORDS:
            if keyword in text:
                return True, keyword
        return False, None
