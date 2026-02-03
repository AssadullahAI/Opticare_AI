from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder("data/eye_images/train", transform=transform)
print(train_ds.class_to_idx)
