import os
import cv2
import glob
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Directories
DATA_DIR = "/home/flare/Dev/Thesis_Sample/processed_data"

# Hyperparameters
IMG_SIZE = (64, 64)

def extract_hog_features(img):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    This provides a good texture/shape representation for traditional ML algorithms.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    
    # Compute HOG features
    features = hog.compute(gray)
    return features.flatten()

def extract_color_histogram(img):
    """
    Extracts a color histogram which is very useful for detecting redness/discoloration in acne.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_data(split):
    X = []
    y = []
    
    split_dir = os.path.join(DATA_DIR, split)
    
    # Load Clear (Class 0)
    clear_dir = os.path.join(split_dir, "clear")
    for img_path in glob.glob(os.path.join(clear_dir, "*.jpg")):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            hog_feat = extract_hog_features(img)
            col_feat = extract_color_histogram(img)
            features = np.hstack([hog_feat, col_feat])
            X.append(features)
            y.append(0)
            
    # Load Acne (Class 1)
    acne_dir = os.path.join(split_dir, "acne")
    for img_path in glob.glob(os.path.join(acne_dir, "*.jpg")):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            hog_feat = extract_hog_features(img)
            col_feat = extract_color_histogram(img)
            features = np.hstack([hog_feat, col_feat])
            X.append(features)
            y.append(1)
            
    return np.array(X), np.array(y)

def main():
    print("Loading training data...")
    X_train, y_train = load_data("train")
    print(f"Loaded {len(y_train)} training samples.")
    
    print("Loading validation data...")
    X_valid, y_valid = load_data("valid")
    print(f"Loaded {len(y_valid)} validation samples.")
    
    print("Loading test data...")
    X_test, y_test = load_data("test")
    print(f"Loaded {len(y_test)} test samples.")
    
    # We can combine train and valid for scikit-learn training
    X_train_full = np.vstack((X_train, X_valid))
    y_train_full = np.concatenate((y_train, y_valid))

    print("Training Random Forest Classifier...")
    # Random Forest is robust and requires less hyperparameter tuning than SVM
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_full, y_train_full)
    
    print("Evaluating model on Test Set...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clear", "Acne"]))
    
    model_path = "acne_model.pkl"
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print("Done!")

if __name__ == "__main__":
    main()
