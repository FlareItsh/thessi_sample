import os
import cv2
import math
import numpy as np
import base64
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker (Tasks API — v0.10+)
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

_base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
_options = mp_vision.FaceLandmarkerOptions(
    base_options=_base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_options)

# Face-oval / silhouette connection set
FACE_OVAL_CONNECTIONS = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL


# ---------------------------------------------------------------------------
# Helper: build face polygon mask
# ---------------------------------------------------------------------------
def build_face_mask(face_landmarks_list, img_h, img_w):
    """
    Uses the face-oval landmarks to create a filled convex-hull mask.
    Returns a uint8 mask (255 = inside face, 0 = outside).
    """
    # Collect indices for the face oval
    oval_indices = sorted({idx for conn in FACE_OVAL_CONNECTIONS for idx in (conn.start, conn.end)})

    pts = []
    for idx in oval_indices:
        lm = face_landmarks_list[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Erode to exclude hairline / jaw boundary noise
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, erode_k, iterations=1)

    return mask


# ---------------------------------------------------------------------------
# Helper: build acne candidate mask (HSV + LAB dual-channel)
# ---------------------------------------------------------------------------
def build_acne_mask(img_bgr, face_mask):
    """
    Returns a binary mask of inflamed/acne pixels using:
    1. HSV  — targets red/pink hues characteristic of acne inflammation.
    2. LAB  — a* channel captures redness independent of brightness.
    Both masks are restricted to the face region.
    """
    # ── HSV redness ────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red spans 0-15° and 160-180° in OpenCV HSV
    low_r1,  up_r1  = np.array([0,   40,  50]), np.array([15,  255, 255])
    low_r2,  up_r2  = np.array([160, 40,  50]), np.array([180, 255, 255])
    # Pink / light-red for fair-skinned acne
    low_pk, up_pk   = np.array([140, 20,  70]), np.array([175, 170, 255])

    hsv_mask = (
        cv2.inRange(hsv, low_r1, up_r1) |
        cv2.inRange(hsv, low_r2, up_r2) |
        cv2.inRange(hsv, low_pk, up_pk)
    )

    # ── LAB a* channel ─────────────────────────────────────────────────────
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    _, a_ch, _ = cv2.split(lab)
    # a* > 145 (in [0,255] space) = strongly reddish pixel
    _, lab_mask = cv2.threshold(a_ch, 145, 255, cv2.THRESH_BINARY)

    # ── Combine & restrict to face ─────────────────────────────────────────
    combined = cv2.bitwise_or(hsv_mask, lab_mask)
    combined = cv2.bitwise_and(combined, combined, mask=face_mask)

    # ── Morphological clean-up ─────────────────────────────────────────────
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)

    return combined


# ---------------------------------------------------------------------------
# Helper: shape-filter contours
# ---------------------------------------------------------------------------
def filter_acne_contours(acne_mask,
                          min_area=15, max_area=6000,
                          min_circularity=0.25, min_solidity=0.60):
    """
    Keeps contours that look like acne spots (roughly circular, solid blobs).
    Discards hairs, skin creases, and background artefacts.
    """
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        valid.append(cnt)
    return valid


# ---------------------------------------------------------------------------
# Helper: skin-tone segmentation fallback (used when full face not detected)
# ---------------------------------------------------------------------------
def build_skin_mask(img_bgr):
    """
    Detects skin-coloured pixels using a dual colour-space approach:
      1. YCrCb — very reliable for a wide range of skin tones
      2. HSV   — captures the same region differently; combined via OR

    Returns a uint8 mask (255 = skin, 0 = non-skin) that is used as the
    region-of-interest when MediaPipe cannot find a full face (e.g. close-ups
    of a cheek, forehead, or chin).
    """
    # ── YCrCb skin range ───────────────────────────────────────────────────
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    # Cr: 135–180, Cb: 85–135  covers most human skin tones
    skin_ycrcb = cv2.inRange(ycrcb,
                              np.array([0,  135,  85], dtype=np.uint8),
                              np.array([255, 180, 135], dtype=np.uint8))

    # ── HSV skin hue range ─────────────────────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Hue 0–20° and 340–360°  with moderate saturation and value
    skin_hsv = (
        cv2.inRange(hsv, np.array([0,  15, 50],  dtype=np.uint8),
                         np.array([20, 200, 255], dtype=np.uint8)) |
        cv2.inRange(hsv, np.array([170, 15, 50],  dtype=np.uint8),
                         np.array([180, 200, 255], dtype=np.uint8))
    )

    # ── Combine ────────────────────────────────────────────────────────────
    skin = cv2.bitwise_or(skin_ycrcb, skin_hsv)

    # Morphological clean-up: close small holes, remove stray pixels
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k_open, iterations=1)

    return skin


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------
def detect_face_and_acne(img_bgr):
    """
    Full pipeline:
      1. Resize for consistent processing.
      2. MediaPipe FaceLandmarker — find 478 face landmarks.
      3. Build face polygon mask from oval landmarks.
      4. Detect inflamed pixels (HSV + LAB) inside face only.
      5. Filter by shape metrics.
      6. Draw:
         - Small cyan dots at every face landmark (structural overlay).
         - Green encircling circles around each acne spot.
      7. Return (result_string, annotated_BGR_image).
    """
    # ── 1. Resize ────────────────────────────────────────────────────────────
    h, w = img_bgr.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    # ── 2. Face Landmark Detection (optional — used when full face is visible) ─
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = FACE_LANDMARKER.detect(mp_image)

    face_lms = None
    if detection_result.face_landmarks:
        face_lms = detection_result.face_landmarks[0]   # first (and only) face

    # ── 3. Region-of-interest mask ───────────────────────────────────────────
    if face_lms is not None:
        # Full face visible → use precise face polygon mask
        roi_mask = build_face_mask(face_lms, h, w)
        mode_label = "face"
    else:
        # Partial / close-up view → fall back to skin-tone segmentation
        roi_mask = build_skin_mask(img_bgr)
        mode_label = "skin"

    # ── 4. Acne mask ─────────────────────────────────────────────────────────
    acne_mask = build_acne_mask(img_bgr, roi_mask)

    # ── 5. Filter ────────────────────────────────────────────────────────────
    valid_contours = filter_acne_contours(acne_mask)

    # ── 6. Draw annotations ──────────────────────────────────────────────────
    output = img_bgr.copy()

    # — Face landmark dots (cyan, radius=1 px) — only when full face detected —
    if face_lms is not None:
        for lm in face_lms:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(output, (px, py), 1, (255, 220, 0), -1)  # cyan

    # — Green circles around acne spots —
    for cnt in valid_contours:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        center = (int(cx), int(cy))
        radius = max(int(radius) + 3, 8)   # small padding so circle is visible

        cv2.circle(output, center, radius + 4, (0, 200, 0), 1)   # outer glow
        cv2.circle(output, center, radius,     (0, 255, 0), 2)   # main circle
        cv2.circle(output, center, 2,          (0, 255, 0), -1)  # centre dot

    # ── 7. Result text ───────────────────────────────────────────────────────
    count = len(valid_contours)
    if count >= 1:
        severity = "Mild" if count <= 5 else ("Moderate" if count <= 15 else "Severe")
        result = f"Acne Detected — {count} spot{'s' if count > 1 else ''} ({severity})"
    else:
        result = "No Acne Detected — Skin looks clear!"

    return result, output


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    elif 'image_base64' in request.form:
        img_data = request.form['image_base64']
        encoded_data = img_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    else:
        return jsonify({"error": "No image provided"}), 400

    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    try:
        result, annotated_img = detect_face_and_acne(img)

        _, buffer = cv2.imencode('.jpg', annotated_img,
                                  [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "result": result,
            "image_base64": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
