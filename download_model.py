import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=10gb2HGZfkYlgq9B78DixTL309vx7nAdF"
MODEL_PATH = "models/eye_disease_model.pth"

def download_model():
    os.makedirs("models", exist_ok=True)

    # Download model
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # Verify model size
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found after download.")

    size = os.path.getsize(MODEL_PATH)
    if size < 10000000:  # 10MB minimum size
        raise ValueError(
            f"Downloaded model file is too small ({size} bytes). "
            "Download likely failed or got corrupted."
        )

    return MODEL_PATH
