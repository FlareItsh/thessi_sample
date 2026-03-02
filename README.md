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

## How to Run the App

### Quick Start Commands

To easily run the application from your terminal, execute the following commands in order:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install the required dependencies
pip install flask opencv-python

# 4. Run the Flask web server
python3 app.py
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
