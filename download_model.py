import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=10gb2HGZfkYlgq9B78DixTL309vx7nAdF"
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


    return MODEL_PATH



