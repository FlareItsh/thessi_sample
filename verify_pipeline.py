import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import dull_razor, optimize_color_space


def verify_pipeline(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # 1. Load Original
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Hair Removal (DullRazor — multi-directional)
    hair_removed = dull_razor(image)
    hair_removed_rgb = cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB)

    # 3. Colour Optimisation → 3-channel [a*, b*, CLAHE-L]
    acne_opt, a_channel, b_channel, clahe_l = optimize_color_space(hair_removed)

    # Visualisation — 6 panels
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.title("1 — Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("2 — After DullRazor (Hair Removal)")
    plt.imshow(hair_removed_rgb)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("3 — Ch1: a* Channel (Redness)")
    plt.imshow(a_channel, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("4 — Ch2: b* Channel (Yellowness)")
    plt.imshow(b_channel, cmap='YlOrBr')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("5 — Ch3: CLAHE-L (Texture)")
    plt.imshow(clahe_l, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("6 — Final 3-Channel Composite")
    # Show the composite as a pseudo-RGB: a*→R, b*→G, CLAHE-L→B
    plt.imshow(acne_opt)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("pipeline_verification.png", dpi=150)
    print("Verification results saved to pipeline_verification.png")


if __name__ == "__main__":
    # Test on the first available training image
    data_dir = os.path.join(os.path.dirname(__file__), "data", "train", "images")
    if os.path.isdir(data_dir):
        images = sorted(os.listdir(data_dir))
        if images:
            sample_path = os.path.join(data_dir, images[0])
            print(f"Using sample: {sample_path}")
            verify_pipeline(sample_path)
        else:
            print(f"No images found in {data_dir}")
    else:
        print(f"Directory not found: {data_dir}")
