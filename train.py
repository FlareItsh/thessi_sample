import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from model import unet_model, dice_coef, bce_dice_loss
from augmentation import get_train_augmentation, get_valid_augmentation

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 30


# ── Data Generator ───────────────────────────────────────────────────────────
class AcneDataGenerator(tf.keras.utils.Sequence):
    """
    Keras-compatible generator that loads 3-channel images + binary masks
    and applies on-the-fly augmentation.
    """

    def __init__(self, image_paths, mask_paths, batch_size, augmentation=None,
                 shuffle=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.image_paths) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size: (idx + 1) * self.batch_size
        ]

        X_batch = []
        y_batch = []

        for i in batch_indices:
            # Load 3-channel image and single-channel mask
            img = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # Apply augmentation (both image and mask transformed together)
            if self.augmentation:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            # Normalise to [0, 1]
            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0

            X_batch.append(img)
            y_batch.append(mask)

        X = np.array(X_batch)                            # (B, 256, 256, 3)
        y = np.expand_dims(np.array(y_batch), axis=-1)   # (B, 256, 256, 1)

        return X, y


def get_file_pairs(split):
    """
    Returns sorted lists of (image_path, mask_path) for a given split,
    ensuring each image has a matching mask file.
    """
    images_dir = os.path.join(DATA_DIR, split, "images")
    masks_dir = os.path.join(DATA_DIR, split, "masks")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    # Build lookup by base name to ensure proper pairing
    mask_lookup = {}
    for mp in mask_paths:
        base = os.path.basename(mp).replace("_mask", "")
        mask_lookup[base] = mp

    paired_imgs = []
    paired_masks = []
    for ip in image_paths:
        base = os.path.basename(ip)
        if base in mask_lookup:
            paired_imgs.append(ip)
            paired_masks.append(mask_lookup[base])

    return paired_imgs, paired_masks


def main():
    print("Setting up data generators...")

    train_imgs, train_masks = get_file_pairs("train")
    valid_imgs, valid_masks = get_file_pairs("valid")
    test_imgs, test_masks = get_file_pairs("test")

    print(f"Train: {len(train_imgs)} | Valid: {len(valid_imgs)} | Test: {len(test_imgs)}")

    # Augmented generator for training, plain resize for validation/test
    train_aug = get_train_augmentation(IMG_SIZE[0])
    valid_aug = get_valid_augmentation(IMG_SIZE[0])

    train_gen = AcneDataGenerator(train_imgs, train_masks, BATCH_SIZE,
                                  augmentation=train_aug, shuffle=True)
    valid_gen = AcneDataGenerator(valid_imgs, valid_masks, BATCH_SIZE,
                                  augmentation=valid_aug, shuffle=False)
    test_gen = AcneDataGenerator(test_imgs, test_masks, BATCH_SIZE,
                                 augmentation=valid_aug, shuffle=False)

    # Initialise U-Net
    model = unet_model(input_size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "acne_unet_best.keras", save_best_only=True,
            monitor='val_dice_coef', mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True,
            monitor='val_dice_coef', mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=4,
            monitor='val_dice_coef', mode='max',
            min_lr=1e-7, verbose=1,
        ),
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("\nEvaluating model on Test Set...")
    results = model.evaluate(test_gen)
    print(f"Test Loss:      {results[0]:.4f}")
    print(f"Test Dice Coef: {results[1]:.4f}")
    print(f"Test Accuracy:  {results[2]:.4f}")

    model.save("acne_unet_final.keras")
    print("Model saved to acne_unet_final.keras")


if __name__ == "__main__":
    main()
