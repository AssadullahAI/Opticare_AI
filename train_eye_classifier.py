import os
import ssl, certifi
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import UnidentifiedImageError


# -----------------------
# SSL FIX (correct)
# -----------------------
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# -----------------------
# Skip corrupt images
# -----------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


data_dir = "data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = SafeImageFolder(f"{data_dir}/train", transform=transform)
val_ds = SafeImageFolder(f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Training on", device)

# -----------------------
# Model (ResNet18 pretrained)
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    model.train()
    total, correct, train_loss = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    train_acc = correct / total
    train_loss = train_loss / len(train_loader)

    model.eval()
    val_total, val_correct, val_loss = 0, 0, 0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            out_val = model(x_val)
            loss_val = criterion(out_val, y_val)

            val_loss += loss_val.item()
            _, pred_val = torch.max(out_val, 1)
            val_total += y_val.size(0)
            val_correct += (pred_val == y_val).sum().item()

    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}"
    )

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/eye_classifier.pth")
print("Saved model to models/eye_classifier.pth")
