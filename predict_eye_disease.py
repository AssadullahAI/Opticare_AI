import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from class_names import CLASS_NAMES

# -----------------------
# Device
# -----------------------
device = torch.device("cpu")

# -----------------------
# Load model
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(torch.load("models/eye_classifier.pth", map_location=device))
model.eval()

# -----------------------
# Transform (same as training)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Prediction function
# -----------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    disease = CLASS_NAMES[predicted.item()]
    return disease
