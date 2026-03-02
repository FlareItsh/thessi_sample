import os
import cv2
import random
import glob
from pathlib import Path

# Paths to YOLO dataset
DATASET_DIR = "/home/flare/Dev/Thesis_Sample/data"
SPLITS = ["train", "valid", "test"]

# Output paths
OUTPUT_DIR = "/home/flare/Dev/Thesis_Sample/processed_data"
PATCH_SIZE = 64

def get_yolo_boxes(label_path, img_width, img_height):
    """Reads YOLO format txt file and returns a list of bounding boxes (x, y, w, h) in pixels."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height (normalized)
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                w_px = int(w * img_width)
                h_px = int(h * img_height)
                x_px = int((x_center * img_width) - (w_px / 2))
                y_px = int((y_center * img_height) - (h_px / 2))
                
                boxes.append((x_px, y_px, w_px, h_px))
    return boxes

def extract_patches():
    # Create output directories
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "acne"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "clear"), exist_ok=True)

    acne_count = 0
    clear_count = 0

    for split in SPLITS:
        images_dir = os.path.join(DATASET_DIR, split, "images")
        labels_dir = os.path.join(DATASET_DIR, split, "labels")
        
        image_files = glob.glob(os.path.join(images_dir, "*.*"))
        print(f"Processing {split} split: {len(image_files)} images found.")

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Find corresponding label file
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            boxes = get_yolo_boxes(label_path, img_width, img_height)
            
            # 1. Extract positive (Acne) patches
            for i, (x, y, w, h) in enumerate(boxes):
                # Add some padding/context
                pad_x = w // 4
                pad_y = h // 4
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_width, x + w + pad_x)
                y2 = min(img_height, y + h + pad_y)
                
                patch = img[y1:y2, x1:x2]
                
                if patch.size > 0:
                    patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                    out_path = os.path.join(OUTPUT_DIR, split, "acne", f"{base_name}_acne_{i}.jpg")
                    cv2.imwrite(out_path, patch_resized)
                    acne_count += 1
            
            # 2. Extract negative (Clear) patches
            # Sample random regions and ensure they don't significantly overlap with any acne box
            num_negative = min(len(boxes) * 2, 5) # Try to extract up to twice as many negative patches or at least 5
            attempts = 0
            neg_extracted = 0
            
            while neg_extracted < num_negative and attempts < 20:
                attempts += 1
                # Random patch size similar to average acne size or a fixed size
                neg_w, neg_h = PATCH_SIZE, PATCH_SIZE
                
                if img_width <= neg_w or img_height <= neg_h:
                    continue
                    
                rx = random.randint(0, img_width - neg_w)
                ry = random.randint(0, img_height - neg_h)
                
                # Check overlap with existing acne boxes
                overlap = False
                for (ax, ay, aw, ah) in boxes:
                    # Simple rectangle intersection check
                    if (rx < ax + aw and rx + neg_w > ax and
                        ry < ay + ah and ry + neg_h > ay):
                        overlap = True
                        break
                
                if not overlap:
                    patch = img[ry:ry+neg_h, rx:rx+neg_w]
                    if patch.size > 0:
                        patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                        out_path = os.path.join(OUTPUT_DIR, split, "clear", f"{base_name}_clear_{neg_extracted}.jpg")
                        cv2.imwrite(out_path, patch_resized)
                        clear_count += 1
                        neg_extracted += 1

    print(f"Extraction complete! Total Acne patches: {acne_count}, Total Clear patches: {clear_count}")

if __name__ == "__main__":
    random.seed(42)
    extract_patches()
