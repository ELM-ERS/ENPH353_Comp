#!/usr/bin/env python3

import os
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_IMAGES = 100  # number of random samples

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
BANNER_TEMPLATE = os.path.join(SCRIPT_PATH, "clue_banner.png")

IMAGES_DIR = os.path.join(SCRIPT_PATH, "banners")
LABELS_DIR = os.path.join(SCRIPT_PATH, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"
FONT_SIZE = 90
FONT_COLOR = (255, 0, 0)  # BGR in OpenCV; we'll reverse for PIL (RGB)

# Positions for the two lines (same as your original script)
KEY_ORIGIN = (250, 30)  # x, y for top line
VALUE_ORIGIN = (30, 250)  # x, y for bottom line

# Random text config
CHARSET = "0123456789" + string.ascii_uppercase  # 0–9 + A–Z
KEY_LEN_RANGE = (3, 7)
VALUE_LEN_RANGE = (3, 12)

# how much of the character cell to trim off the top of each box
TOP_TRIM_FRACTION = 0.17  # ~12% of char height; tweak if you like
BOTTOM_TRIM_FRACTION = 0.1  # ~12% of char height; tweak if you like

# Character classes
CLASS_ORDER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 classes
CHAR_TO_ID = {ch: i for i, ch in enumerate(CLASS_ORDER)}

# Extra class for the whole sign
SIGN_CLASS_ID = len(CLASS_ORDER)  # class 36

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_text(min_len, max_len):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(CHARSET) for _ in range(length))


def measure_char(font):
    """
    Measure a single monospaced character using ImageDraw.textsize,
    which matches how text is actually rendered.
    """
    dummy = Image.new("L", (256, 256))
    d = ImageDraw.Draw(dummy)
    # w, h = d.textsize("A", font=font)  # "A" as representative character
    w = d.textlength("A", font=font)  # "A" as representative character
    h = FONT_SIZE
    return w, h


def add_line_boxes(text, origin, img_w, img_h, char_w, char_h):
    """
    Compute YOLO boxes (one per character) for a monospaced text line.
    Returns list of (class_id, cx_norm, cy_norm, w_norm, h_norm).
    """
    x0_origin, y0_origin = origin
    labels = []

    # we keep the bottom of the box at y0_origin + char_h
    # and move the top down a bit
    top_offset = int(TOP_TRIM_FRACTION * char_h)
    bottom_offset = int(BOTTOM_TRIM_FRACTION * char_h)

    for i, ch in enumerate(text):
        if ch not in CHAR_TO_ID:
            continue

        x0 = x0_origin + i * char_w
        y0 = y0_origin + top_offset  # shifted down
        x1 = x0 + char_w
        y1 = y0_origin + char_h + bottom_offset  # same as before -> bottom unchanged

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        labels.append(
            (
                CHAR_TO_ID[ch],
                cx / img_w,
                cy / img_h,
                (x1 - x0) / img_w,
                (y1 - y0) / img_h,
            )
        )

    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    banner = cv2.imread(BANNER_TEMPLATE)
    if banner is None:
        raise RuntimeError(f"Could not load template image at {BANNER_TEMPLATE}")

    img_h, img_w = banner.shape[:2]

    pil_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    char_w, char_h = measure_char(pil_font)

    for i in range(NUM_IMAGES):
        base = banner.copy()

        # Convert to PIL (OpenCV BGR -> PIL RGB)
        base_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(base_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Generate random strings
        key_text = random_text(*KEY_LEN_RANGE)
        value_text = random_text(*VALUE_LEN_RANGE)

        # Draw text (FONT_COLOR reversed to RGB)
        draw.text(KEY_ORIGIN, key_text, fill=FONT_COLOR[::-1], font=pil_font)
        draw.text(VALUE_ORIGIN, value_text, fill=FONT_COLOR[::-1], font=pil_font)

        # Character-level labels
        labels = []
        labels += add_line_boxes(key_text, KEY_ORIGIN, img_w, img_h, char_w, char_h)
        labels += add_line_boxes(value_text, VALUE_ORIGIN, img_w, img_h, char_w, char_h)

        # Whole-sign label (covers the full image)
        labels.append(
            (
                SIGN_CLASS_ID,
                0.5,  # cx
                0.5,  # cy
                1.0,  # w
                1.0,  # h
            )
        )

        # Convert back to OpenCV BGR
        final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Save image
        img_name = f"plate_{i}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        cv2.imwrite(img_path, final_img)

        # Save YOLO labels
        label_name = f"plate_{i}.txt"
        label_path = os.path.join(LABELS_DIR, label_name)
        with open(label_path, "w") as f:
            for cls_id, cx, cy, w, h in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"[{i}] {img_name}: {len(labels) - 1} char boxes + 1 sign box")


if __name__ == "__main__":
    main()
