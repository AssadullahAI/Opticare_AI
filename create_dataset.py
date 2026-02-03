import os
import csv
import random
import shutil

from pathlib import Path

# ====== CONFIG ======
DATA_DIR = Path("data")
ORIGINAL_DIR = DATA_DIR / "original"
AUGMENTED_DIR = DATA_DIR / "augmented"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

VAL_SPLIT = 0.20  # 20% validation
RANDOM_SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def gather_images_from_folder(folder: Path):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                img_path = Path(root) / file
                label = Path(root).name
                images.append((str(img_path), label))
    return images


def create_csv_and_copy(images, dest_dir: Path, csv_path: Path):
    os.makedirs(dest_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])

        for img_path, label in images:
            filename = os.path.basename(img_path)
            new_path = dest_dir / filename

            # Copy image to train/val folder
            shutil.copy(img_path, new_path)

            writer.writerow([str(new_path), label])


def main():
    random.seed(RANDOM_SEED)

    # Gather all images from original + augmented
    original_images = gather_images_from_folder(ORIGINAL_DIR)
    augmented_images = gather_images_from_folder(AUGMENTED_DIR)

    all_images = original_images + augmented_images
    random.shuffle(all_images)

    # Split
    split_index = int(len(all_images) * (1 - VAL_SPLIT))
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Clear existing train/val folders
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(VAL_DIR, ignore_errors=True)

    # Create CSV and copy files
    create_csv_and_copy(train_images, TRAIN_DIR, TRAIN_CSV)
    create_csv_and_copy(val_images, VAL_DIR, VAL_CSV)

    print("✔️ CSV files created successfully!")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")


if __name__ == "__main__":
    main()
