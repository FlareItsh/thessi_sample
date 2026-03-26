import os
import cv2
import numpy as np
import glob
from pathlib import Path

# Paths to YOLO dataset
DATASET_DIR = "/Users/MYPC1/Desktop/vs code/thesis/thessi_sample/data"
SPLITS = ["train", "valid", "test"]

# Output paths
OUTPUT_DIR = "/Users/MYPC1/Desktop/vs code/thesis/thessi_sample/processed_data"
IMG_HEIGHT, IMG_WIDTH = 256, 256

def dull_razor(img):
    """
    Implements the DullRazor algorithm for hair removal using morphological operations.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Black-Hat transformation to identify hair-like structures
    # A 7x7 or 9x9 kernel is usually effective for hairs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Thresholding to create a mask for hairs
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Inpaint the hair regions using the original image
    # Telea's algorithm is fast and effective for linear structures like hair
    inpainted = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
    
    return inpainted

def optimize_color_space(img):
    """
    Converts image to CIE L*a*b* and isolates the 'a' channel.
    Applies CLAHE to the 'L' channel for lighting normalization.
    """
    # Convert BGR to Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge optimized channels (we specifically keep 'a' as it highlights inflammation)
    # For U-Net, we might use a 3-channel input or just the 'a' channel.
    # Here we return the full Lab with enhanced L and the original 'a' channel.
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    
    # Return both the enhanced Lab and specifically the 'a' channel for visualization/analysis
    return lab_enhanced, a

def get_yolo_masks(label_path, img_width, img_height):
    """
    Reads YOLO format txt file and generates a binary mask where 1 = acne, 0 = skin.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not os.path.exists(label_path):
        return mask
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height (normalized)
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # Convert to pixel coordinates for the bounding box
                w_px = int(w * img_width)
                h_px = int(h * img_height)
                x1 = int((x_center * img_width) - (w_px / 2))
                y1 = int((y_center * img_height) - (h_px / 2))
                x2 = x1 + w_px
                y2 = y1 + h_px
                
                # Fill the rectangle in the mask
                cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(img_width, x2), min(img_height, y2)), 1, -1)
                
    return mask

def process_dataset():
    """
    Full pipeline processing: Hair removal -> Color Optimization -> Mask Generation.
    """
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

        images_dir = os.path.join(DATASET_DIR, split, "images")
        labels_dir = os.path.join(DATASET_DIR, split, "labels")
        
        image_files = glob.glob(os.path.join(images_dir, "*.*"))
        print(f"Processing {split} split: {len(image_files)} images found.")

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 1. Preprocessing & Hair Removal
            img_clean = dull_razor(img)
            
            # Median Blurring (kernel size 5x5)
            img_blurred = cv2.medianBlur(img_clean, 5)
            
            # 2. Color Space Optimization
            lab_enhanced, a_channel = optimize_color_space(img_blurred)
            
            # For the segmentation model, we use the 'a' channel as input
            # since it strongest highlights acne inflammation.
            # We resize to a fixed input size (e.g., 256x256).
            img_final = cv2.resize(a_channel, (IMG_WIDTH, IMG_HEIGHT))
            
            # 3. Generate Binary Mask
            img_h, img_w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            mask = get_yolo_masks(label_path, img_w, img_h)
            mask_final = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            # Save results
            out_img_path = os.path.join(OUTPUT_DIR, split, "images", f"{base_name}.png")
            out_mask_path = os.path.join(OUTPUT_DIR, split, "masks", f"{base_name}_mask.png")
            
            cv2.imwrite(out_img_path, img_final)
            cv2.imwrite(out_mask_path, mask_final * 255) # Save as visual mask (0 or 255)

    print("Pipeline processing complete!")

if __name__ == "__main__":
    process_dataset()
