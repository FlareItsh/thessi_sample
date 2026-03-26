import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import unet_model, dice_coef, dice_loss

# Directories
DATA_DIR = "/home/flare/Dev/Thesis_Sample/processed_data"
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 20

def load_data(split):
    """
    Loads processed images and masks for a given split.
    """
    X = []
    y = []
    
    images_dir = os.path.join(DATA_DIR, split, "images")
    masks_dir = os.path.join(DATA_DIR, split, "masks")
    
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Already 'a' channel
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None and mask is not None:
            # Normalize
            X.append(img.astype(np.float32) / 255.0)
            y.append(mask.astype(np.float32) / 255.0)
            
    X = np.expand_dims(np.array(X), axis=-1)
    y = np.expand_dims(np.array(y), axis=-1)
    
    return X, y

def main():
    print("Loading data...")
    X_train, y_train = load_data("train")
    X_valid, y_valid = load_data("valid")
    X_test, y_test = load_data("test")
    
    print(f"Train samples: {len(X_train)}")
    print(f"Valid samples: {len(X_valid)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize U-Net model
    model = unet_model(input_size=(IMG_SIZE[0], IMG_SIZE[1], 1))
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("acne_unet_best.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("Evaluating model on Test Set...")
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Dice Coef: {results[1]:.4f}")
    print(f"Test Accuracy: {results[2]:.4f}")
    
    model.save("acne_unet_final.h5")
    print("Model saved to acne_unet_final.h5")

if __name__ == "__main__":
    main()

