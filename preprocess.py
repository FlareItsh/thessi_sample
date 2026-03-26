import os
import cv2
import numpy as np
import glob
from pathlib import Path

# Paths to YOLO dataset
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")
SPLITS = ["train", "valid", "test"]

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
IMG_HEIGHT, IMG_WIDTH = 256, 256


def dull_razor(img):
    """
    Improved DullRazor algorithm for hair removal.
    Uses multiple oriented morphological kernels to detect hairs at all
    angles, then merges the masks and inpaints the detected regions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Multiple oriented kernels to catch hairs at every major angle
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)),   # horizontal
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17)),   # vertical
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),    # square fallback
    ]

    # 45° and 135° diagonal line kernels
    diag_45 = np.zeros((9, 9), dtype=np.uint8)
    np.fill_diagonal(diag_45, 1)
    kernels.append(diag_45)

    diag_135 = np.zeros((9, 9), dtype=np.uint8)
    np.fill_diagonal(np.fliplr(diag_135), 1)
    kernels.append(diag_135)

    # Merge blackhat results from all orientations
    combined_mask = np.zeros_like(gray)
    for kernel in kernels:
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_or(combined_mask, thresh)

    # Dilate the hair mask slightly so inpainting covers edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)

    # Inpaint on the original colour image
    inpainted = cv2.inpaint(img, combined_mask, 3, cv2.INPAINT_TELEA)

    return inpainted


def optimize_color_space(img):
    """
    Converts a BGR image to CIE L*a*b* and produces a 3-channel
    acne-optimised image:

        Channel 1 — a* (redness):   highlights red/pink inflammation
        Channel 2 — b* (yellowness): highlights pustules and yellow tones
        Channel 3 — CLAHE-L (texture): lighting-normalised structural detail

    Returns:
        acne_optimized: 3-channel uint8 image [a*, b*, CLAHE-L]
        a_channel:      raw a* channel
        b_channel:      raw b* channel
        clahe_l:        CLAHE-enhanced L channel
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE on the L channel to normalise uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)

    # Stack the three informative channels
    acne_optimized = cv2.merge([a_ch, b_ch, l_enhanced])

    return acne_optimized, a_ch, b_ch, l_enhanced


def get_yolo_masks(label_path, img_width, img_height):
    """
    Reads a YOLO-format txt file and generates a binary mask using FILLED
    ELLIPSES instead of rectangles. Acne lesions are roughly circular, so
    ellipses produce much cleaner ground truth than axis-aligned rectangles.

    Returns a uint8 mask where 255 = acne, 0 = clear skin.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height (normalised)
                x_center, y_center, w, h = map(float, parts[1:5])

                # Convert to pixel coordinates
                cx = int(x_center * img_width)
                cy = int(y_center * img_height)
                # Semi-axes of the ellipse (half of bounding box dims)
                ax = max(int((w * img_width) / 2), 1)
                ay = max(int((h * img_height) / 2), 1)

                # Draw filled ellipse — much tighter fit for round acne spots
                cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    return mask


def process_dataset():
    """
    Full preprocessing pipeline:
        1. Hair removal   (DullRazor)
        2. De-noise        (Median Blur)
        3. Colour-space optimisation → 3-channel [a*, b*, CLAHE-L]
        4. Mask generation from YOLO labels (elliptical)
        5. Resize & save
    """
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

        images_dir = os.path.join(DATASET_DIR, split, "images")
        labels_dir = os.path.join(DATASET_DIR, split, "labels")

        image_files = glob.glob(os.path.join(images_dir, "*.*"))
        print(f"Processing {split} split: {len(image_files)} images found.")

        processed = 0
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. Hair removal
            img_clean = dull_razor(img)

            # 2. De-noise
            img_blurred = cv2.medianBlur(img_clean, 5)

            # 3. Colour-space optimisation → 3-channel
            acne_optimized, _, _, _ = optimize_color_space(img_blurred)

            # 4. Resize to fixed model input size
            img_final = cv2.resize(acne_optimized, (IMG_WIDTH, IMG_HEIGHT))

            # 5. Generate binary mask from YOLO labels (elliptical)
            img_h, img_w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            mask = get_yolo_masks(label_path, img_w, img_h)
            mask_final = cv2.resize(
                mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST
            )

            # 6. Save
            out_img_path = os.path.join(
                OUTPUT_DIR, split, "images", f"{base_name}.png"
            )
            out_mask_path = os.path.join(
                OUTPUT_DIR, split, "masks", f"{base_name}_mask.png"
            )

            cv2.imwrite(out_img_path, img_final)
            cv2.imwrite(out_mask_path, mask_final)

            processed += 1

        print(f"  {split}: {processed} images processed.")

    print("Pipeline processing complete!")


if __name__ == "__main__":
    process_dataset()
