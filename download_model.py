import os
import gdown

MODEL_URL = "https://github.com/AssadullahAI/Opticare_AI/releases/download/v1.0/eye_disease_model.pth"
MODEL_PATH = "models/eye_disease_model.pth"

def download_model():
    os.makedirs("models", exist_ok=True)

    # download model
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # verify file
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not downloaded")

    if os.path.getsize(MODEL_PATH) < 10000000:
        raise ValueError("Model file is too small. Download failed.")

    return MODEL_PATH





