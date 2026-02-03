import os
import shutil
import random
from pathlib import Path

def create_train_val(original_dir, augmented_dir, output_dir, split=0.8):
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # Delete old train/val
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)

    # Combine both sources
    sources = [original_dir, augmented_dir]

    for source in sources:
        for class_name in os.listdir(source):
            class_path = Path(source) / class_name
            if not class_path.is_dir():
                continue

            images = list(class_path.glob("*.*"))
            random.shuffle(images)

            split_idx = int(len(images) * split)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            for img in train_images:
                dest = train_dir / class_name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(img, dest / img.name)

            for img in val_images:
                dest = val_dir / class_name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(img, dest / img.name)

    print("[INFO] Dataset prepared successfully!")
    print(f"Train samples: {sum(len(list((train_dir / d).glob('*.*'))) for d in os.listdir(train_dir))}")
    print(f"Val samples: {sum(len(list((val_dir / d).glob('*.*'))) for d in os.listdir(val_dir))}")

if __name__ == "__main__":
    create_train_val(
        original_dir="data/original",
        augmented_dir="data/augmented",
        output_dir="data",
        split=0.8  # 80% train, 20% val
    )
