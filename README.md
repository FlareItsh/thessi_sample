# Acne Detector Pro

A Machine Learning-inspired Web Application developed to identify facial acne. This system uses advanced Image Processing (OpenCV) combined with a Python Flask Backend and an elegant Tailwind CSS frontend.

## Overview of Algorithms

### Preprocessing & Fallback Detection (OpenCV)

Initially, a script `preprocess.py` was developed to parse YOLO formatted dataset bounding boxes, cropping out "Acne" regions (positive class) and random "Clear" skin regions (negative class).

Due to immense download sizes and environment constraints regarding heavy Machine Learning models (like TensorFlow or Scikit-Learn SVMs), the live application `app.py` uses a **Pure OpenCV Fallback Algorithm**:

1. **Color Space Conversion:** Images are converted from `BGR` to `HSV`.
2. **Skin/Acne Masking:** We define explicit `HSV` bounds to isolate reddish skin discolorations typical of acne and inflammation.
3. **Morphological Filtering:** We apply Opening and Closing elliptical kernels of size 5x5 to remove random noise and connect adjacent inflamed groupings.
4. **Contour Extraction:** We locate all blobs/contours on the filtered mask.
5. **Feature Filtering:** We filter down to contours that have a pixel area characteristic of acne (10 < area < 5000) and analyze their geometric circularity (> 0.2) to differentiate from long scratches or lighting artifacts.

## Setup Environment

Before running the application or training the model, set up your Python environment:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

## How to Train the Model

If you have a new dataset or want to train the U-Net model from scratch, follow these steps:

#### 1. Data Preprocessing
Run the preprocessing script to prepare the images and masks. This script implements the **DullRazor** algorithm for hair removal and optimizes the color space using the **CIE L*a*b*** 'a' channel to highlight inflammation.

```bash
python preprocess.py
```

#### 2. Model Training
Run the training script. It will use the processed data to train a U-Net architecture. If a GPU (CUDA) is detected, TensorFlow will automatically use it. The trained model weights will be saved (e.g., `acne_unet_best.h5` and `acne_unet_final.h5`).

```bash
python train.py
```

#### 3. Verification (Optional)
You can verify the preprocessing and augmentation pipeline by running:

```bash
python verify_pipeline.py
```
This generates a `pipeline_verification.png` file showing the visual stages of the pipeline.

## How to Run the App

Once your environment is set up and the model is ready, you can start the application web server for inference.

```bash
python app.py
```
This will start the backend server at `http://localhost:5000`. Open this URL in your web browser to use the Acne Detector Pro.

### Using the Interface

1. Navigate to `http://localhost:5000/` in your web browser.
2. The UI provides two intuitive pathways:
   - **Upload Image:** Select a local image from your hard drive containing a face.
   - **Use Camera:** Instantly open your computer camera and snap a photo to analyze.
3. Click "Analyze Image". The Flask server will return JSON containing either `"Acne Detected (# spots)"` or `"No Acne Detected"`.

## Project Structure

- `app.py`: Flask Web Server and OpenCV Detection Logic.
- `preprocess.py`: Extractor script to slice images using YOLO txt bounds.
- `templates/index.html`: Stunning responsive user interface.
- `data/`: Original image and annotation dataset folders.
