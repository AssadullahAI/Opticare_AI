import gdown
import os

url = "https://drive.google.com/uc?id=10gb2HGZfkYlgq9B78DixTL309vx7nAdF"
output = "models/eye_disease_model.pth"

os.makedirs("models", exist_ok=True)
gdown.download(url, output, quiet=False)
