"""
═══════════════════════════════════════════════════════════════
            OPTICARE AI – IMAGE CLASSIFICATION MODULE
═══════════════════════════════════════════════════════════════

Eye Disease Image Analysis with Quality Checking & Safety Logic

⚠️ IMPORTANT:
- This is a DEMONSTRATION classifier (transfer learning backbone)
- NOT a medical diagnostic device
- Always advise professional evaluation

Author: OptiCare AI Team
Version: 2.0.0
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from config.settings import Config


# ============================================================
# IMAGE QUALITY CHECKER
# ============================================================

class ImageQualityChecker:
    """
    Performs basic image quality checks before analysis
    """

    def check_quality(self, image: Image.Image) -> dict:
        width, height = image.size
        mean_brightness = np.array(image).mean()

        is_acceptable = (
            width >= Config.IMAGE_SIZE[0]
            and height >= Config.IMAGE_SIZE[1]
            and mean_brightness > 40
        )

        return {
            "is_acceptable": is_acceptable,
            "resolution": (width, height),
            "mean_brightness": round(mean_brightness, 2),
            "aspect_ratio": round(width / height, 2),
            "recommendation": (
                "Image quality acceptable"
                if is_acceptable
                else "Low quality image. Use a clearer, well-lit eye photo."
            )
        }


# ============================================================
# EYE DISEASE CLASSIFIER
# ============================================================

class EyeDiseaseClassifier:
    """
    Eye disease image classifier using a CNN backbone
    """

    def __init__(self):
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Using device: {self.device}")

        # Class names (MUST match training order)
        self.class_names = Config.IMAGE_CLASSES
        self.num_classes = len(self.class_names)

        # Build model
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes
        )

        # Load trained weights if available
        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # --------------------------------------------------------
    # CHECKPOINT LOADER (ROBUST)
    # --------------------------------------------------------

    def _load_checkpoint(self):
        """
        Load model checkpoint if exists.
        Supports:
        - full checkpoint dict
        - weights-only state_dict
        """
        if not os.path.exists(Config.MODEL_PATH):
            print("⚠️ Model checkpoint not found. Using ImageNet weights.")
            return

        checkpoint = torch.load(
            Config.MODEL_PATH,
            map_location=self.device
        )

        # Full checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Weights-only checkpoint
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)

        else:
            raise ValueError("❌ Invalid model checkpoint format")

        print("✅ Model checkpoint loaded successfully")

    # --------------------------------------------------------
    # IMAGE ANALYSIS
    # --------------------------------------------------------

    def analyze_with_confidence(
        self,
        image: Image.Image,
        threshold: float = None
    ) -> dict:
        """
        Analyze eye image and return prediction with confidence
        """
        threshold = threshold or Config.CONFIDENCE_THRESHOLD

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)

        probs = torch.softmax(outputs[0], dim=0)
        confidence, idx = torch.max(probs, dim=0)

        predicted_class_key = self.class_names[idx.item()]
        disease_info = Config.get_disease_info(predicted_class_key)

        confidence_level = (
            "High" if confidence.item() >= threshold else "Low"
        )

        return {
            "predicted_class": disease_info["name"],
            "confidence": round(confidence.item(), 3),
            "severity": disease_info["severity"],
            "confidence_level": confidence_level,
            "all_probabilities": {
                Config.get_disease_info(self.class_names[i])["name"]:
                round(probs[i].item(), 3)
                for i in range(self.num_classes)
            }
        }

    # --------------------------------------------------------
    # MEDICAL RECOMMENDATION
    # --------------------------------------------------------

    def get_recommendation(self, result: dict) -> str:
        """
        Generate medical recommendation based on severity
        """
        severity = result.get("severity", "unknown")

        if severity == "critical":
            return (
                "🚨 This may indicate a medical emergency. "
                "Seek immediate ophthalmological or emergency care."
            )

        if severity == "high":
            return (
                "⚠️ This condition may threaten vision. "
                "Consult an ophthalmologist as soon as possible."
            )

        if severity == "moderate":
            return "ℹ️ Schedule an eye examination with a specialist."

        return (
            "✅ No urgent signs detected. Monitor symptoms and "
            "seek professional advice if concerns arise."
        )


# ============================================================
# CHECKPOINT SAVER (FOR TRAINING)
# ============================================================

def save_checkpoint(model, optimizer, epoch, loss):
    """
    Save model checkpoint to disk
    """
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        Config.MODEL_PATH
    )
