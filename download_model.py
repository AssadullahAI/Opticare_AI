import os
import gdown
from config.settings import Config

MODEL_PATH = Config.MODEL_PATH
DRIVE_LINK = "https://drive.google.com/uc?id=10gb2HGZfkYlgq9B78DixTL309vx7nAdF"

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)
