import gdown
import os

MODEL_PATH = "model/model.pt"
DRIVE_ID = "10gb2HGZfkYlgq9B78DixTL309vx7nAdF"

def download_model():
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

    # Verify file exists and size
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
        raise Exception("Model download failed or file is incomplete. Check Google Drive link.")
