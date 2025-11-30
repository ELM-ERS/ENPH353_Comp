#!/usr/bin/env python3

from ultralytics import YOLO
import os
import glob
import random

# Adjust if you keep things somewhere else
ROOT = os.path.dirname(os.path.realpath(__file__))


IMAGES_DIR = os.path.join(ROOT, "images")  # or your one folder
LABELS_DIR = os.path.join(ROOT, "labels")  # required

TRAIN_SPLIT = 0.9  # 90% train, 10% val
DATA_FRACTION = 0.005 # How much of the data to actually use (dataset is massive lmao)

def build_split_lists():
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    random.shuffle(images)

    if 0.0 < DATA_FRACTION < 1.0:
        n_keep = int(len(images) * DATA_FRACTION)
        images = images[:n_keep]

    n_train = int(len(images) * TRAIN_SPLIT)

    train_images = images[:n_train]
    val_images = images[n_train:]

    # Write lists to text files YOLO reads
    with open(os.path.join(ROOT, "train.txt"), "w") as f:
        for img in train_images:
            f.write(img + "\n")

    with open(os.path.join(ROOT, "val.txt"), "w") as f:
        for img in val_images:
            f.write(img + "\n")

    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")


def train():
    """
    Train YOLO12 to detect the sign (and optionally characters) on full 1400x1400 images.
    """
    model = YOLO("yolo12m.pt")  # or yolo12s.pt / yolo12m.pt if you want bigger

    model.train(
        data=os.path.join(ROOT, "dataset.yaml"),
        epochs=100,
        imgsz=640,  # network input size; 640 is a good starting point
        batch=64,  # adjust for your GPU
        # lr0=0.01,  # base LR
        optimizer="sgd",  # or "adamw"
        device=0,  # GPU index or 'cpu'
        workers=20,
        project=os.path.join(ROOT, "runs_yolo12"),
        name="data1_attempt_1",
        pretrained=True,  # use the yolo12n.pt weights
        visualize=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        cache="disk",
        
        # degrees=10,
        # scale=0.5,
        # mosaic=1.0,
    )


if __name__ == "__main__":
    # pick which stage to run
    build_split_lists()
    train()
