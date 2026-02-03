import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=10gb2HGZfkYlgq9B78DixTL309vx7nAdF"
MODEL_PATH = "models/eye_disease_model.pth"

os.makedirs("models", exist_ok=True)
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

