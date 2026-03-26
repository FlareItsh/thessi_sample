# Acne Detector Pro

A Machine Learning-inspired Web Application developed to identify facial acne. This system uses **MediaPipe Face Landmarker** for precise face detection, combined with **multi-feature acne analysis** (OpenCV), a Python Flask Backend, and an elegant Tailwind CSS frontend.

## Overview of Algorithms

### Face Landmark Detection (MediaPipe)

The system uses **MediaPipe Face Landmarker** (Tasks API) with a pre-trained model (`face_landmarker.task`) to detect **478 facial landmarks** on each face. These landmarks are used to:

1. **Define the face boundary** — A convex hull from the face oval landmarks creates a precise face mask
2. **Map facial zones** — The face is divided into 5 acne-prone zones:
   - Forehead
   - Left Cheek
   - Right Cheek
   - Nose
   - Chin
3. **Exclude non-skin regions** — Eyes, eyebrows, lips, and nostrils are excluded from scanning to prevent false positives

### Multi-Feature Acne Detection Pipeline

Within the detected face region, acne analysis uses a 4-step scoring pipeline:

1. **Skin Segmentation**: Dual-space skin masking using both **YCrCb** and **HSV** colour spaces, intersected with the face landmark mask.

2. **Multi-Pass Candidate Detection**:
   - **Pass A (Colour)**: HSV redness detection across two red hue ranges (0–12° and 165–180°) plus pink/inflamed detection
   - **Pass B (Blob)**: Difference of Gaussians (DoG) blob detection for texture anomalies

3. **Non-Maximum Suppression**: Nearby detections are merged to avoid double-counting.

4. **Multi-Feature Scoring** — Each candidate is scored with 4 features:
   - **Local Redness (40%)**: Compares spot colour to surrounding skin using CIELAB a* channel (contextual, not absolute)
   - **Texture Roughness (20%)**: Local Binary Pattern (LBP) variance measures skin roughness
   - **Shape Score (20%)**: Circularity analysis — acne tends to be round
   - **Size Score (20%)**: Area appropriateness (too large = not acne, too small = noise)
   - **Multi-source bonus**: Candidates detected by both colour AND blob get a 20% confidence boost

### Severity Classification

Based on confirmed spot count:
- **Clear** — 0 spots
- **Mild** — 1–5 spots
- **Moderate** — 6–15 spots
- **Severe** — 16+ spots

## How to Run the App

### Quick Start Commands

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install the required dependencies
pip install flask opencv-python mediapipe

# 4. Download the MediaPipe face landmarker model (only needed once)
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')"

# 5. Run the Flask web server
python app.py
```

This will start the backend server at `http://localhost:5001`. Open this URL in your web browser to use the Acne Detector Pro.

### Using the Interface

1. Navigate to `http://localhost:5001/` in your web browser.
2. The UI provides two intuitive pathways:
   - **Upload Image:** Select a local image from your hard drive containing a face.
   - **Use Camera:** Instantly open your computer camera and snap a photo to analyze.
3. Click "Analyze Image". The system will:
   - Detect the face using MediaPipe (with a real-time status indicator)
   - Scan skin zones for acne spots
   - Show an AI-annotated image with confidence scores
   - Display severity classification and zone breakdown

### Understanding Results

- **Green outline** — Detected face boundary
- **Red circles** — High confidence acne spots (≥60%)
- **Orange circles** — Medium confidence spots (40–59%)
- **Yellow circles** — Low confidence spots (20–39%)
- **Percentage labels** — Individual confidence score for each spot
- **Zone badges** — Which facial areas are affected

## Project Structure

- `app.py`: Flask Web Server, MediaPipe Face Detection & Multi-Feature Acne Analysis
- `face_landmarker.task`: Pre-trained MediaPipe face landmark model
- `preprocess.py`: Extractor script to slice images using YOLO txt bounds
- `train.py`: Random Forest training pipeline (HOG + colour histogram features)
- `templates/index.html`: Responsive user interface with Tailwind CSS
- `data/`: Original image and annotation dataset folders
- `processed_data/`: Extracted positive/negative patches for training

## Dependencies

- Flask
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
