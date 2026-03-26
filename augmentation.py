import albumentations as A
import cv2


def get_train_augmentation(img_size=256):
    """
    Segmentation-compatible augmentation pipeline.
    Both image and mask are transformed identically for geometric transforms.
    Colour transforms are applied only to the image.
    """
    return A.Compose([
        # Geometric (applied to both image AND mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.ElasticTransform(
            alpha=40, sigma=40 * 0.05,
            p=0.2,
        ),

        # Colour / intensity (applied only to image, not mask)
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.4
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Final resize to guarantee correct dimensions
        A.Resize(img_size, img_size),
    ])


def get_valid_augmentation(img_size=256):
    """
    Validation / test augmentation — resize only, no randomness.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
    ])
