import os
import csv
import glob
from typing import Tuple

import cv2
import numpy as np


class ImageVectorRecorder:
    """
    Simple helper to record (image, (x, y)) pairs to disk.

    Directory layout:

        root_dir/
          images/
            000001.png
            000002.png
            ...
          labels.csv   # filename,x,y
    """

    def __init__(self, root_dir: str):
        """
        root_dir: Base directory where data will be stored.
                  Will be created if it doesn't exist.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        self.labels_path = os.path.join(root_dir, "labels.csv")

        # If labels.csv doesn't exist, create it with header
        if not os.path.exists(self.labels_path):
            with open(self.labels_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "x", "y"])

        # Figure out what the next ID should be based on existing images
        self.next_id = self._find_next_id()

    def _find_next_id(self) -> int:
        """
        Scan existing PNGs in images/ and set the next ID
        to max(existing_ids) + 1. Assumes names like '000123.png'.
        """
        pattern = os.path.join(self.img_dir, "*.png")
        files = glob.glob(pattern)

        max_id = 0
        for path in files:
            name = os.path.splitext(os.path.basename(path))[0]
            if name.isdigit():
                max_id = max(max_id, int(name))

        return max_id + 1

    def add_sample(self, image_bgr: np.ndarray, xy: Tuple[float, float]):
        """
        Save an OpenCV image (BGR numpy array) and a (x, y) pair.

        image_bgr: np.ndarray in BGR format (as from cv2).
        xy: tuple (x, y), numeric (float/int).
        """
        # Build filename like '000001.png'
        filename = f"{self.next_id:06d}.png"
        img_path = os.path.join(self.img_dir, filename)

        # Save image (cv2 expects BGR)
        success = cv2.imwrite(img_path, image_bgr)
        if not success:
            raise RuntimeError(f"Failed to write image to {img_path}")

        x, y = xy

        # Append to labels.csv
        with open(self.labels_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([filename, float(x), float(y)])

        # Increment ID for next call
        self.next_id += 1
