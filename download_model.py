import os
import gdown
from config.settings import Config

MODEL_PATH = Config.MODEL_PATH
DRIVE_ID = "10gb2HGZfkYlgq9B78DixTL309vx7nAdF"

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(
        f"https://drive.google.com/uc?id={DRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )
