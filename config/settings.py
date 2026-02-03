"""
═══════════════════════════════════════════════════════════════════════════
                    OPTICARE AI - CONFIGURATION SETTINGS
═══════════════════════════════════════════════════════════════════════════

Advanced Configuration Management for Eye AI Application

This file contains ALL configuration settings for the entire application.
To customize the app, simply edit the values below.

Author: Assadullah Asif
Version: 2.0.0
Last Updated: 2026-02-01
═══════════════════════════════════════════════════════════════════════════
"""

import torch
import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """
    Centralized configuration management for OptiCare AI
    """

    # ═══════════════════════════════════════════════════════════════════════
    # APPLICATION INFORMATION
    # ═══════════════════════════════════════════════════════════════════════
    APP_NAME = "👁️ OptiCare AI - Advanced Medical Eye Analysis Platform"
    VERSION = "2.0.0"
    ENVIRONMENT = os.getenv("ENV", "production")

    # ═══════════════════════════════════════════════════════════════════════
    # DIRECTORY PATHS
    # ═══════════════════════════════════════════════════════════════════════
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    ASSETS_DIR = BASE_DIR / "assets"
    LOGS_DIR = BASE_DIR / "logs"

    # ═══════════════════════════════════════════════════════════════════════
    # AI MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    IMAGE_CLASSES = [
        "central serous chorioretinopathy",
        "diabetic retinopathy",
        "disc edema",
        "glaucoma",
        "healthy",
        "macular star",
        "myopia",
        "pterygium",
        "retinal detachment",
        "retinitis pigmentosa"
    ]

    SIMILARITY_THRESHOLD = 0.7
    TOP_K_RESULTS = 5
    MAX_RESPONSE_LENGTH = 500

    # ═══════════════════════════════════════════════════════════════════════
    # IMAGE CLASSIFICATION CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.6

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING / CHECKPOINT CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    MODEL_PATH = MODELS_DIR / "eye_disease_model.pth"
    CHECKPOINT_PATH = MODEL_PATH
    RESUME_TRAINING = True

    # ═══════════════════════════════════════════════════════════════════════
    # DEVICE (GPU/CPU)
    # ═══════════════════════════════════════════════════════════════════════
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ═══════════════════════════════════════════════════════════════════════
    # DISEASE INFORMATION DATABASE
    # ═══════════════════════════════════════════════════════════════════════
    DISEASES = {
        "central serous chorioretinopathy": {
            "name": "Central Serous Chorioretinopathy",
            "severity": "high",
            "emoji": "🔍",
            "color": "#FFA500"
        },
        "diabetic retinopathy": {
            "name": "Diabetic Retinopathy",
            "severity": "high",
            "emoji": "🩺",
            "color": "#DC143C"
        },
        "disc edema": {
            "name": "Disc Edema",
            "severity": "high",
            "emoji": "⚠️",
            "color": "#FF6347"
        },
        "glaucoma": {
            "name": "Glaucoma",
            "severity": "high",
            "emoji": "⚠️",
            "color": "#FF6347"
        },
        "healthy": {
            "name": "Healthy",
            "severity": "low",
            "emoji": "✅",
            "color": "#00FF00"
        },
        "macular star": {
            "name": "Macular Star",
            "severity": "moderate",
            "emoji": "⭐",
            "color": "#FFA500"
        },
        "myopia": {
            "name": "Myopia",
            "severity": "low",
            "emoji": "👓",
            "color": "#87CEEB"
        },
        "pterygium": {
            "name": "Pterygium",
            "severity": "low",
            "emoji": "🌿",
            "color": "#FFB6C1"
        },
        "retinal detachment": {
            "name": "Retinal Detachment",
            "severity": "critical",
            "emoji": "🚨",
            "color": "#FF0000"
        },
        "retinitis pigmentosa": {
            "name": "Retinitis Pigmentosa",
            "severity": "high",
            "emoji": "🧬",
            "color": "#DC143C"
        }
    }

    # ═══════════════════════════════════════════════════════════════════════
    # EMERGENCY DETECTION KEYWORDS
    # ═══════════════════════════════════════════════════════════════════════
    EMERGENCY_KEYWORDS = [
        "sudden vision loss", "sudden blindness", "complete vision loss",
        "flashing lights", "many floaters", "floaters and flashes",
        "curtain over vision", "shadow over vision", "veil over vision",
        "eye injury", "trauma to eye", "chemical in eye",
        "chemical exposure", "chemical burn", "severe eye pain",
        "excruciating pain", "eye swelling shut", "cannot open eye",
        "pus discharge", "foreign object stuck", "penetrating injury",
        "eye laceration", "double vision suddenly", "pupil irregular",
        "blood in eye", "severe headache with vision", "nausea with vision loss"
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # MEDICAL DISCLAIMER TEXT
    # ═══════════════════════════════════════════════════════════════════════
    MEDICAL_DISCLAIMER = """
    ⚠️ IMPORTANT MEDICAL DISCLAIMER

    This application is for educational and informational purposes only.
    This is NOT a substitute for professional medical advice, diagnosis, or treatment.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    MAX_CONVERSATION_HISTORY = 10
    SESSION_TIMEOUT = 3600

    # ═══════════════════════════════════════════════════════════════
    # SAFETY & PRIVACY SETTINGS
    # ═══════════════════════════════════════════════════════════════
    ENABLE_CONTENT_FILTERING = True
    LOG_QUERIES = True
    ANONYMIZE_LOGS = True

    # ═══════════════════════════════════════════════════════════════════════
    # UI THEME CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    THEME = {
        "primaryColor": "#1E88E5",
        "backgroundColor": "#F5F7FA",
        "secondaryBackgroundColor": "#FFFFFF",
        "textColor": "#262730",
        "font": "sans-serif"
    }

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════
    @classmethod
    def get_disease_info(cls, disease_key: str) -> Dict[str, Any]:
        return cls.DISEASES.get(disease_key, {
            "name": "Unknown",
            "severity": "unknown",
            "emoji": "❓",
            "color": "#808080"
        })

    @classmethod
    def initialize_directories(cls):
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.ASSETS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Auto initialize directories
Config.initialize_directories()
