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
KEY_ORIGIN = (250, 30)  # x, y for top line (in inner banner coords)
VALUE_ORIGIN = (30, 250)  # x, y for bottom line (in inner banner coords)

# Random text config
CHARSET = "0123456789" + string.ascii_uppercase  # 0–9 + A–Z
KEY_LEN_RANGE = (3, 7)
VALUE_LEN_RANGE = (3, 12)

# how much of the character cell to trim off the top of each box
TOP_TRIM_FRACTION = 0.17
BOTTOM_TRIM_FRACTION = 0.10

# Character classes
CLASS_ORDER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 classes
CHAR_TO_ID = {ch: i for i, ch in enumerate(CLASS_ORDER)}

# Extra class for the whole sign (inner white area only)
SIGN_CLASS_ID = len(CLASS_ORDER)  # class 36

# Blue outer background (BGR)
BLUE_BG_COLOR = (255, 0, 0)  # bright blue in BGR

# Separate border widths
TOP_BOTTOM_BORDER_PX = 35  # top & bottom border thickness
LEFT_RIGHT_BORDER_PX = 90  # left & right border thickness

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_text(min_len, max_len):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(CHARSET) for _ in range(length))


def measure_char(font):
    dummy = Image.new("L", (256, 256))
    d = ImageDraw.Draw(dummy)
    w = d.textlength("A", font=font)  # "A" as representative character
    h = FONT_SIZE
    return w, h


def add_line_boxes(text, origin, img_w, img_h, char_w, char_h):
    x0_origin, y0_origin = origin
    labels = []

    top_offset = int(TOP_TRIM_FRACTION * char_h)
    bottom_offset = int(BOTTOM_TRIM_FRACTION * char_h)

    for i, ch in enumerate(text):
        if ch not in CHAR_TO_ID:
            continue

        x0 = x0_origin + i * char_w
        y0 = y0_origin + top_offset
        x1 = x0 + char_w
        y1 = y0_origin + char_h + bottom_offset

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

    # original clue_banner size (inner white/grey panel with logo)
    inner_h, inner_w = banner.shape[:2]

    pil_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    char_w, char_h = measure_char(pil_font)

    for i in range(NUM_IMAGES):
        base = banner.copy()

        # Draw text on the INNER banner (no blue yet)
        base_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(base_rgb)
        draw = ImageDraw.Draw(pil_img)

        key_text = random_text(*KEY_LEN_RANGE)
        value_text = random_text(*VALUE_LEN_RANGE)

        draw.text(KEY_ORIGIN, key_text, fill=FONT_COLOR[::-1], font=pil_font)
        draw.text(VALUE_ORIGIN, value_text, fill=FONT_COLOR[::-1], font=pil_font)

        # Character-level boxes in inner-banner coordinates
        inner_char_labels = []
        inner_char_labels += add_line_boxes(
            key_text, KEY_ORIGIN, inner_w, inner_h, char_w, char_h
        )
        inner_char_labels += add_line_boxes(
            value_text, VALUE_ORIGIN, inner_w, inner_h, char_w, char_h
        )

        # Convert back to OpenCV BGR
        inner_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # -------------------------------------------------------
        # Add blue background/border around the inner banner
        # -------------------------------------------------------
        pad_y = TOP_BOTTOM_BORDER_PX
        pad_x = LEFT_RIGHT_BORDER_PX

        out_h = inner_h + 2 * pad_y
        out_w = inner_w + 2 * pad_x

        blue_bg = np.full((out_h, out_w, 3), BLUE_BG_COLOR, dtype=np.uint8)
        blue_bg[pad_y : pad_y + inner_h, pad_x : pad_x + inner_w] = inner_img

        # -------------------------------------------------------
        # Adjust character boxes for padding and new size
        # -------------------------------------------------------
        labels = []
        for cls_id, cx, cy, w, h in inner_char_labels:
            # original inner pixel corners
            x0_inner = (cx - w / 2.0) * inner_w
            y0_inner = (cy - h / 2.0) * inner_h
            x1_inner = (cx + w / 2.0) * inner_w
            y1_inner = (cy + h / 2.0) * inner_h

            # shift by padding into outer coords
            x0 = x0_inner + pad_x
            y0 = y0_inner + pad_y
            x1 = x1_inner + pad_x
            y1 = y1_inner + pad_y

            bw = x1 - x0
            bh = y1 - y0

            cx_new = (x0 + x1) / 2.0 / out_w
            cy_new = (y0 + y1) / 2.0 / out_h
            w_new = bw / out_w
            h_new = bh / out_h

            labels.append((cls_id, cx_new, cy_new, w_new, h_new))

        # -------------------------------------------------------
        # Whole-sign box: ONLY the inner white area
        # -------------------------------------------------------
        sign_x0 = pad_x
        sign_y0 = pad_y
        sign_x1 = pad_x + inner_w
        sign_y1 = pad_y + inner_h

        sign_w = sign_x1 - sign_x0
        sign_h = sign_y1 - sign_y0

        sign_cx = (sign_x0 + sign_x1) / 2.0 / out_w
        sign_cy = (sign_y0 + sign_y1) / 2.0 / out_h
        sign_w_norm = sign_w / out_w
        sign_h_norm = sign_h / out_h

        labels.append(
            (
                SIGN_CLASS_ID,
                sign_cx,
                sign_cy,
                sign_w_norm,
                sign_h_norm,
            )
        )

        # Save image
        img_name = f"plate_{i}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        cv2.imwrite(img_path, blue_bg)

        # Save YOLO labels
        label_name = f"plate_{i}.txt"
        label_path = os.path.join(LABELS_DIR, label_name)
        with open(label_path, "w") as f:
            for cls_id, cx, cy, w, h in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"[{i}] {img_name}: {len(labels) - 1} char boxes + 1 sign box")


if __name__ == "__main__":
    main()
