#!/usr/bin/env python3

import os
import glob
import random
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

IN_IMAGES_DIR = os.path.join(SCRIPT_PATH, "banners")
IN_LABELS_DIR = os.path.join(SCRIPT_PATH, "labels")

OUT_IMAGES_DIR = os.path.join(SCRIPT_PATH, "aug_images")
OUT_LABELS_DIR = os.path.join(SCRIPT_PATH, "aug_labels")
OUT_SIGN_IMAGES_DIR = os.path.join(SCRIPT_PATH, "aug_sign_images")
OUT_SIGN_LABELS_DIR = os.path.join(SCRIPT_PATH, "aug_sign_labels")

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)
os.makedirs(OUT_SIGN_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_SIGN_LABELS_DIR, exist_ok=True)

AUG_PER_IMAGE = 4

# camera resolution
OUT_W = 1400
OUT_H = 1400

RESIZE_DIM = 680

# affine parameter ranges
ANGLE_RANGE_DEG = (-5.0, 5.0)
SCALE_RANGE = (0.2, 2.2)

NO_OBJECT_FRACTION=0.2
SHEAR_RANGE_DEG=(-20,20)

# must match your generator
SIGN_CLASS_ID = 36

# how many processes to use
NUM_WORKERS = 20  # or: os.cpu_count()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_yolo_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            labels.append((cls_id, cx, cy, w, h))
    return labels


def yolo_to_corners(box, img_w, img_h):
    cls_id, cx, cy, w, h = box
    bw = w * img_w
    bh = h * img_h
    x_c = cx * img_w
    y_c = cy * img_h
    x0 = x_c - bw / 2.0
    y0 = y_c - bh / 2.0
    x1 = x_c + bw / 2.0
    y1 = y_c + bh / 2.0
    return cls_id, x0, y0, x1, y1


def corners_to_yolo(cls_id, x0, y0, x1, y1, img_w, img_h):
    bw = x1 - x0
    bh = y1 - y0
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0
    return (
        cls_id,
        cx / img_w,
        cy / img_h,
        bw / img_w,
        bh / img_h,
    )

def random_affine_matrix(src_w, src_h, dst_w, dst_h):
    """
    Affine that:
    - rotates in ANGLE_RANGE_DEG
    - scales in SCALE_RANGE but clamped so rotated sign fits inside dst
    - applies shear in SHEAR_RANGE_DEG
    - places sign fully inside dst
    """
    # rotation
    angle_deg = random.uniform(*ANGLE_RANGE_DEG)
    angle_rad = math.radians(angle_deg)

    # shear
    shear_deg = random.uniform(*SHEAR_RANGE_DEG)
    shear_rad = math.radians(shear_deg)

    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))

    # base scale
    base_scale = random.uniform(*SCALE_RANGE)

    w2 = src_w / 2.0
    h2 = src_h / 2.0

    # account for rotation with no shear (shear doesn't change bounding extents much)
    denom_w = 2.0 * (cos_a * w2 + sin_a * h2)
    denom_h = 2.0 * (sin_a * w2 + cos_a * h2)
    s_limit_w = dst_w / denom_w
    s_limit_h = dst_h / denom_h
    s_max_angle = min(s_limit_w, s_limit_h)

    if s_max_angle <= 0:
        s_max_angle = base_scale

    scale = min(base_scale, s_max_angle)

    # random center inside dst
    half_w = scale * (cos_a * w2 + sin_a * h2)
    half_h = scale * (sin_a * w2 + cos_a * h2)

    cx_min = half_w
    cx_max = dst_w - half_w
    cy_min = half_h
    cy_max = dst_h - half_h

    if cx_max < cx_min:
        cx_min = cx_max = dst_w / 2.0
    if cy_max < cy_min:
        cy_min = cy_max = dst_h / 2.0

    center_x = random.uniform(cx_min, cx_max)
    center_y = random.uniform(cy_min, cy_max)

    # --- build full affine matrix manually (rotation + scale + shear) ---
    M = np.eye(3, dtype=np.float32)

    # scale + rotation
    rot = np.array([
        [ math.cos(angle_rad), -math.sin(angle_rad)],
        [ math.sin(angle_rad),  math.cos(angle_rad)]
    ], dtype=np.float32) * scale

    # shear in x direction
    shear = np.array([
        [1, math.tan(shear_rad)],
        [0, 1]
    ], dtype=np.float32)

    A = rot @ shear   # combine ops
    M[:2, :2] = A

    # translate so source center maps to desired random center
    src_center = np.array([src_w / 2.0, src_h / 2.0, 1], dtype=np.float32)
    current_center = M @ src_center
    dx = center_x - current_center[0]
    dy = center_y - current_center[1]
    M[0, 2] = dx
    M[1, 2] = dy

    return M[:2, :]



def apply_affine_to_points(points, M):
    pts = np.array(points, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_t = pts_h @ M.T
    return pts_t


def transform_boxes_yolo(boxes, M, src_w, src_h, dst_w, dst_h):
    new_boxes = []
    for box in boxes:
        cls_id, x0, y0, x1, y1 = yolo_to_corners(box, src_w, src_h)

        corners = [
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y1),
        ]
        corners_t = apply_affine_to_points(corners, M)

        xs = corners_t[:, 0]
        ys = corners_t[:, 1]

        x_min = max(0.0, float(xs.min()))
        y_min = max(0.0, float(ys.min()))
        x_max = min(dst_w - 1.0, float(xs.max()))
        y_max = min(dst_h - 1.0, float(ys.max()))

        if x_max <= x_min or y_max <= y_min:
            continue

        new_boxes.append(
            corners_to_yolo(cls_id, x_min, y_min, x_max, y_max, dst_w, dst_h)
        )
    return new_boxes


def crop_to_sign(image, boxes, dst_w, dst_h):
    """
    Given the 1400x1400 image and its YOLO boxes:
    - find the sign box (SIGN_CLASS_ID)
    - crop the sign region
    - re-express all character boxes in cropped coordinates
    - add a sign box that covers the whole crop
    Returns (crop_img, crop_labels) or (None, None) if sign not found.
    """
    sign_box = None
    char_boxes = []
    for b in boxes:
        if b[0] == SIGN_CLASS_ID:
            sign_box = b
        else:
            char_boxes.append(b)

    if sign_box is None:
        return None, None

    # sign crop rectangle in dst pixels
    _, sx0, sy0, sx1, sy1 = yolo_to_corners(sign_box, dst_w, dst_h)
    sx0 = max(0, int(np.floor(sx0)))
    sy0 = max(0, int(np.floor(sy0)))
    sx1 = min(dst_w, int(np.ceil(sx1)))
    sy1 = min(dst_h, int(np.ceil(sy1)))

    if sx1 <= sx0 or sy1 <= sy0:
        return None, None

    crop = image[sy0:sy1, sx0:sx1].copy()
    crop_w = sx1 - sx0
    crop_h = sy1 - sy0

    crop_labels = []

    # full-sign box in crop coords
    crop_labels.append((SIGN_CLASS_ID, 0.5, 0.5, 1.0, 1.0))

    # character boxes
    for b in char_boxes:
        cls_id, x0, y0, x1, y1 = yolo_to_corners(b, dst_w, dst_h)

        x0 -= sx0
        x1 -= sx0
        y0 -= sy0
        y1 -= sy0

        x0_cl = max(0.0, x0)
        y0_cl = max(0.0, y0)
        x1_cl = min(float(crop_w - 1), x1)
        y1_cl = min(float(crop_h - 1), y1)

        if x1_cl <= x0_cl or y1_cl <= y0_cl:
            continue

        crop_labels.append(
            corners_to_yolo(cls_id, x0_cl, y0_cl, x1_cl, y1_cl, crop_w, crop_h)
        )

    return crop, crop_labels


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------


def process_image(img_path):
    """
    Worker function run in a separate process.
    Returns (base_name, num_full, num_crops).
    """
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(IN_LABELS_DIR, base_name + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[{base_name}] Could not read image, skipping")
        return base_name, 0, 0

    src_h, src_w = img.shape[:2]
    dst_w, dst_h = OUT_W, OUT_H

    orig_boxes = read_yolo_labels(lbl_path)
    if not orig_boxes:
        print(f"[{base_name}] No labels found, skipping")
        return base_name, 0, 0

    full_count = 0
    crop_count = 0

    for k in range(AUG_PER_IMAGE):

        # -------------------------------------------------------
        # No-object images (noise only) with empty label file
        # -------------------------------------------------------
        if random.random() < NO_OBJECT_FRACTION:
            # create random noise background
            bg = np.random.randint(0, 256, (dst_h, dst_w, 3), dtype=np.uint8)

            out_img_name = f"{base_name}_aug{k}_empty.png"
            out_lbl_name = f"{base_name}_aug{k}_empty.txt"

            cv2.imwrite(
                os.path.join(OUT_IMAGES_DIR, out_img_name),
                cv2.resize(bg, (RESIZE_DIM, RESIZE_DIM))
            )

            # create empty label file
            open(os.path.join(OUT_LABELS_DIR, out_lbl_name), "w").close()

            # no crops for empty samples
            continue

        
        # noise background
        bg = np.random.randint(0, 256, (dst_h, dst_w, 3), dtype=np.uint8)

        # affine in source coords
        M = random_affine_matrix(src_w, src_h, dst_w, dst_h)

        warped = cv2.warpAffine(
            img,
            M,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        bg[mask] = warped[mask]

        new_boxes = transform_boxes_yolo(orig_boxes, M, src_w, src_h, dst_w, dst_h)
        if not new_boxes:
            continue

        # save full 1400x1400
        out_img_name = f"{base_name}_aug{k}.png"
        out_lbl_name = f"{base_name}_aug{k}.txt"
        cv2.imwrite(
            os.path.join(OUT_IMAGES_DIR, out_img_name), cv2.resize(bg, (680, 680))
        )
        with open(os.path.join(OUT_LABELS_DIR, out_lbl_name), "w") as f:
            for cls_id, cx, cy, w, h in new_boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        full_count += 1

        # save cropped sign
        crop_img, crop_labels = crop_to_sign(bg, new_boxes, dst_w, dst_h)
        if crop_img is not None and crop_labels:
            crop_img_name = f"{base_name}_aug{k}_crop.png"
            crop_lbl_name = f"{base_name}_aug{k}_crop.txt"
            cv2.imwrite(
                os.path.join(OUT_SIGN_IMAGES_DIR, crop_img_name),
                # cv2.resize(crop_img, (RESIZE_DIM, RESIZE_DIM)),
                crop_img,
            )
            with open(os.path.join(OUT_SIGN_LABELS_DIR, crop_lbl_name), "w") as f:
                for cls_id, cx, cy, w, h in crop_labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            crop_count += 1

    return base_name, full_count, crop_count


# ---------------------------------------------------------------------------
# Main: parallel dispatch
# ---------------------------------------------------------------------------


def main():
    image_paths = sorted(glob.glob(os.path.join(IN_IMAGES_DIR, "plate_*.png")))
    print(f"Found {len(image_paths)} base images")

    if not image_paths:
        return

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(process_image, p): p for p in image_paths}
        for fut in as_completed(futures):
            base_name, full_cnt, crop_cnt = fut.result()
            print(f"[{base_name}] full={full_cnt}  cropped={crop_cnt}")


if __name__ == "__main__":
    main()
