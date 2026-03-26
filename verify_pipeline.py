import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import dull_razor, optimize_color_space
from augmentation import get_train_augmentation

def verify_pipeline(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # 1. Load Original
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Pre-processing: Hair Removal (DullRazor)
    hair_removed = dull_razor(image)
    hair_removed_rgb = cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)

    # 3. Color Optimization: L*a*b* and CLAHE
    a_channel, clahe_l = optimize_color_space(hair_removed)

    # 4. Augmentation (Dry Run)
    # We need a dummy mask for augmentation test
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    aug = get_train_augmentation()
    augmented = aug(image=image, mask=mask)
    aug_img = augmented['image']
    aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)

    # Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("After DullRazor")
    plt.imshow(hair_removed_rgb)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("'a' Channel (Inflammation)")
    plt.imshow(a_channel, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("CLAHE 'L' (Normalized)")
    plt.imshow(clahe_l, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Augmented Image")
    plt.imshow(aug_img_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("pipeline_verification.png")
    print("Verification results saved to pipeline_verification.png")

if __name__ == "__main__":
    # Test on the first image in train set
    sample_path = "/home/flare/Dev/Thesis_Sample/data/train/images/acne-101_jpeg.rf.af45ba09bf2981ed7cc4af94b54aae48.jpg"
    verify_pipeline(sample_path)
