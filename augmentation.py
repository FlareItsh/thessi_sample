import albumentations as A
import cv2
import numpy as np
import os
import glob

def get_train_augmentation():
    """
    Returns an Albumentations pipeline for training data augmentation.
    """
    return A.Compose([
        A.Rotate(limit=45, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Resize(256, 256),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def augment_sample(image_path, label_path):
    """
    Demonstrates augmentation on a single sample.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bboxes = []
    class_labels = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height
                cls = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                bboxes.append(bbox)
                class_labels.append(cls)
                
    transform = get_train_augmentation()
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    aug_image = transformed['image']
    aug_bboxes = transformed['bboxes']
    
    return cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR), aug_bboxes

if __name__ == "__main__":
    # Example usage:
    # img, bboxes = augment_sample("sample.jpg", "sample_labels.txt")
    # cv2.imwrite("augmented_sample.jpg", img)
    print("Augmentation script initialized.")
