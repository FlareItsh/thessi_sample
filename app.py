import os
import cv2
import numpy as np
import base64
import pickle
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ──────────────────────────────────────────────────────────────
# MediaPipe Face Landmarker initialisation (new Tasks API)
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

face_landmarker = FaceLandmarker.create_from_options(_landmarker_options)

# ──────────────────────────────────────────────────────────────
# Load Acne Classifier ML Model (Random Forest)
# ──────────────────────────────────────────────────────────────
_ACNE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "acne_model.pkl")
acne_ml_model = None
if os.path.exists(_ACNE_MODEL_PATH):
    try:
        with open(_ACNE_MODEL_PATH, "rb") as f:
            acne_ml_model = pickle.load(f)
        print("ML Model loaded successfully.")
    except Exception as e:
        print(f"Error loading acne_model.pkl: {e}")
else:
    print("Warning: acne_model.pkl not found. Falling back to heuristic scoring.")

# ──────────────────────────────────────────────────────────────
# Facial zone landmark indices (MediaPipe 478-point mesh)
# These define convex-hull regions for each zone of interest.
# ──────────────────────────────────────────────────────────────
# Forehead region: top of face down to brow line
FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 68, 103, 109, 10]

# Cheek regions (between eyes/nose line and jaw)
LEFT_CHEEK_INDICES = [205, 206, 207, 187, 147, 213, 215, 216, 207, 192, 210, 211, 212, 214, 116, 117, 118, 119, 100, 123, 50, 205]
RIGHT_CHEEK_INDICES = [425, 426, 427, 411, 376, 433, 435, 436, 427, 416, 430, 431, 432, 434, 345, 346, 347, 348, 329, 352, 280, 425]

# Nose bridge and tip
NOSE_INDICES = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 168]

# Chin region (below lips to jaw)
CHIN_INDICES = [175, 152, 377, 400, 378, 379, 365, 397, 288, 361, 132, 58, 172, 136, 150, 149, 176, 148, 175]

# Face oval for overall face boundary
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Eyes and lips to EXCLUDE from scanning
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


def _get_landmark_points(landmarks, indices, w, h):
    """Convert landmark indices to pixel coordinates."""
    pts = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts, dtype=np.int32) if pts else np.array([], dtype=np.int32).reshape(0, 2)


def _create_skin_mask(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 15, 0], dtype=np.uint8)
    upper_hsv = np.array([17, 170, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return skin_mask


def _compute_local_redness(img_bgr, x, y, radius, face_roi_bgr):
    h, w = face_roi_bgr.shape[:2]
    surr_r = max(radius * 3, 15)
    sy1, sy2 = max(0, y - surr_r), min(h, y + surr_r)
    sx1, sx2 = max(0, x - surr_r), min(w, x + surr_r)
    surr_patch = face_roi_bgr[sy1:sy2, sx1:sx2]

    sr = max(radius, 3)
    cy1, cy2 = max(0, y - sr), min(h, y + sr)
    cx1, cx2 = max(0, x - sr), min(w, x + sr)
    spot_patch = face_roi_bgr[cy1:cy2, cx1:cx2]

    if spot_patch.size == 0 or surr_patch.size == 0:
        return 0.0

    spot_lab = cv2.cvtColor(spot_patch, cv2.COLOR_BGR2Lab)
    surr_lab = cv2.cvtColor(surr_patch, cv2.COLOR_BGR2Lab)

    spot_a = float(np.mean(spot_lab[:, :, 1]))
    surr_a = float(np.mean(surr_lab[:, :, 1]))
    redness_diff = (spot_a - surr_a) / 128.0

    spot_hsv = cv2.cvtColor(spot_patch, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(spot_hsv[:, :, 1]))
    hue_mean = float(np.mean(spot_hsv[:, :, 0]))

    is_red_hue = (hue_mean < 15) or (hue_mean > 165)
    sat_score = sat_mean / 255.0

    score = 0.0
    score += max(0, redness_diff) * 0.6
    if is_red_hue:
        score += sat_score * 0.4
    else:
        score += sat_score * 0.1

    return min(1.0, score)


def _compute_lbp_variance(gray_patch):
    if gray_patch.size < 9:
        return 0.0
    h, w = gray_patch.shape
    if h < 3 or w < 3:
        return 0.0

    center = gray_patch[1:-1, 1:-1].astype(float)
    lbp = np.zeros_like(center, dtype=np.uint8)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbour = gray_patch[1+dy:h-1+dy, 1+dx:w-1+dx].astype(float)
        lbp += ((neighbour >= center).astype(np.uint8)) << i

    variance = float(np.var(lbp))
    return min(1.0, variance / 3000.0)


def extract_hog_features(img):
    """Refined HOG extraction matching training script."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    features = hog.compute(gray)
    return features.flatten()


def extract_color_histogram(img):
    """Refined Color Histogram extraction matching training script."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _calculate_iou_circles(x1, y1, r1, x2, y2, r2):
    """Calculate IoU of two circles."""
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return (min(r1, r2)**2) / (max(r1, r2)**2)
    
    r1s, r2s, ds = r1**2, r2**2, d**2
    part1 = r1s * np.arccos((ds + r1s - r2s) / (2 * d * r1))
    part2 = r2s * np.arccos((ds + r2s - r1s) / (2 * d * r2))
    part3 = 0.5 * np.sqrt(max(0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    intersection = part1 + part2 - part3
    union = np.pi * r1s + np.pi * r2s - intersection
    return intersection / union if union > 0 else 0.0


def detect_acne_enhanced(img):
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    # ── Step 0: Global Illumination Normalization (CLAHE) ──
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    img_norm = cv2.merge((l_norm, a, b))
    img = cv2.cvtColor(img_norm, cv2.COLOR_LAB2BGR)

    # ── Step 1: Face Landmark Detection (Tasks API) ──
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = face_landmarker.detect(mp_image)

    fallback_mode = False
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        fallback_mode = True

    zone_defs = {
        "forehead": FOREHEAD_INDICES,
        "left_cheek": LEFT_CHEEK_INDICES,
        "right_cheek": RIGHT_CHEEK_INDICES,
        "nose": NOSE_INDICES,
        "chin": CHIN_INDICES,
    }

    zone_polys = {}
    face_mask_overall = np.zeros((h, w), dtype=np.uint8)
    landmarks = None

    if not fallback_mode:
        landmarks = result.face_landmarks[0]
        for zname, zindices in zone_defs.items():
            pts = _get_landmark_points(landmarks, zindices, w, h)
            if pts.shape[0] >= 3:
                zone_polys[zname] = cv2.convexHull(pts)
        
        oval_pts = _get_landmark_points(landmarks, FACE_OVAL_INDICES, w, h)
        if oval_pts.shape[0] >= 3:
            oval_hull = cv2.convexHull(oval_pts)
            cv2.fillConvexPoly(face_mask_overall, oval_hull, 255)
            for exclude_indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES, LIPS_INDICES, LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES]:
                pts = _get_landmark_points(landmarks, exclude_indices, w, h)
                if pts.shape[0] >= 3:
                    ex_hull = cv2.convexHull(pts)
                    M = cv2.moments(ex_hull)
                    if M["m00"] > 0:
                        cx_m = int(M["m10"] / M["m00"])
                        cy_m = int(M["m01"] / M["m00"])
                        expanded = ((ex_hull - [cx_m, cy_m]) * 1.3 + [cx_m, cy_m]).astype(np.int32)
                        cv2.fillConvexPoly(face_mask_overall, expanded, 0)
        else:
            fallback_mode = True

    if fallback_mode:
        face_mask_overall.fill(255)
        zone_polys["cropped_skin"] = np.array([[[0, 0]], [[w-1, 0]], [[w-1, h-1]], [[0, h-1]]], dtype=np.int32)

    scan_mask_base = face_mask_overall
    
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    candidate_list = []

    # --- Pass A: Narrowed Redness-based (HSV) ---
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # Restored saturation floor to 40 for clearer separation
    mask_r1 = cv2.inRange(hsv, np.array([0, 40, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 40, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_and(cv2.bitwise_or(mask_r1, mask_r2), scan_mask_base)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        # Reduced limits for maximum sensitivity
        if 4 < area < 4000 and circ >= 0.25:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            candidate_list.append({"x": int(cx), "y": int(cy), "r": max(int(radius), 2), "area": area, "circularity": circ, "source": "color"})

    # --- Pass B: Blob detection (DoG) ---
    g1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
    g2 = cv2.GaussianBlur(gray, (7, 7), 2.0)
    dog = cv2.absdiff(g1, g2)
    _, dog_thresh = cv2.threshold(dog, 6, 255, cv2.THRESH_BINARY)
    dog_thresh = cv2.bitwise_and(dog_thresh, scan_mask_base)
    contours_dog, _ = cv2.findContours(dog_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_dog:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        if 6 < area < 3000 and circ >= 0.35:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            candidate_list.append({"x": int(cx), "y": int(cy), "r": max(int(radius), 2), "area": area, "circularity": circ, "source": "blob"})

    # --- Pass C: Texture-based ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, tex_thresh = cv2.threshold(laplacian, 18, 255, cv2.THRESH_BINARY)
    tex_thresh = cv2.bitwise_and(tex_thresh, scan_mask_base)
    contours_tex, _ = cv2.findContours(tex_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_tex:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        if 10 < area < 2000 and circ >= 0.35:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            candidate_list.append({"x": int(cx), "y": int(cy), "r": max(int(radius), 2), "area": area, "circularity": circ, "source": "texture"})

    # --- Pass D: Blackhat morphology (Dark Spots / Blackheads) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, dark_thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dark_thresh = cv2.bitwise_and(dark_thresh, scan_mask_base)
    contours_dark, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_dark:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        if 4 < area < 800 and circ >= 0.35:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            candidate_list.append({"x": int(cx), "y": int(cy), "r": max(int(radius), 2), "area": area, "circularity": circ, "source": "dark"})

    candidate_list.sort(key=lambda x: x["area"], reverse=True)
    if len(candidate_list) > 1000:
        candidate_list = candidate_list[:1000]

    merged = []
    used = [False] * len(candidate_list)
    for i in range(len(candidate_list)):
        if used[i]: continue
        c1 = candidate_list[i]
        group = [c1]
        used[i] = True
        for j in range(i + 1, len(candidate_list)):
            if used[j]: continue
            c2 = candidate_list[j]
            dist = np.sqrt((c1["x"] - c2["x"])**2 + (c1["y"] - c2["y"])**2)
            if dist < (c1["r"] + c2["r"]) * 0.8:
                group.append(c2)
                used[j] = True
        best_candidate = max(group, key=lambda c: c["area"])
        if best_candidate["r"] >= 3:
            merged.append(best_candidate)

    confirmed_candidates = []
    # Initialise zone counts for all possible zones
    zone_counts = {z: 0 for z in ["forehead", "left_cheek", "right_cheek", "cheek", "nose", "chin", "cropped_skin", "other"]}

    for candidate in merged:
        x, y, r = candidate["x"], candidate["y"], candidate["r"]
        assigned_zone = "other"
        if fallback_mode:
            rel_y = y / h
            if rel_y < 0.30: assigned_zone = "forehead"
            elif rel_y < 0.70: assigned_zone = "cheek"
            else: assigned_zone = "chin"
        else:
            for zname, poly in zone_polys.items():
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    assigned_zone = zname
                    break
            
            # If still "other" but inside the face, find the nearest zone by centroid distance
            if assigned_zone == "other" and not fallback_mode:
                try:
                    # oval_hull is defined around line 259
                    oval_pts = _get_landmark_points(landmarks, FACE_OVAL_INDICES, w, h)
                    oval_hull = cv2.convexHull(oval_pts)
                    if cv2.pointPolygonTest(oval_hull, (float(x), float(y)), False) >= 0:
                        min_dist = float('inf')
                        nearest = "other"
                        for zname, poly in zone_polys.items():
                            M = cv2.moments(poly)
                            if M["m00"] > 0:
                                czx, czy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                                d_z = np.sqrt((x - czx)**2 + (y - czy)**2)
                                if d_z < min_dist:
                                    min_dist = d_z
                                    nearest = zname
                        assigned_zone = nearest
                except:
                    pass
        
        if assigned_zone in ["forehead", "left_cheek", "right_cheek"] and candidate["area"] <= 20:
            continue

        # Extract patch for feature analysis (redness, texture) and ML validation
        patch_r = 32
        py1, py2 = max(0, y - patch_r), min(h, y + patch_r)
        px1, px2 = max(0, x - patch_r), min(w, x + patch_r)
        patch_orig = img[py1:py2, px1:px2]
        
        redness, texture = 0.0, 0.0
        if patch_orig.size > 0:
            lab_patch = cv2.cvtColor(patch_orig, cv2.COLOR_BGR2LAB)
            a_channel = lab_patch[:, :, 1]
            
            # Combine mean redness with peak redness to target "red bumps" specifically
            mean_a = float(np.mean(a_channel))
            max_a = float(np.max(a_channel))
            
            # Redness score based on dataset floor of 128.6
            redness = np.clip((mean_a - 128.6) / (185.1 - 128.6), 0, 1)
            
            gray_patch = cv2.cvtColor(patch_orig, cv2.COLOR_BGR2GRAY)
            texture = np.clip(np.std(gray_patch) / 64.0, 0, 1)

            # --- Dark Lesion (Scar/Pigment) Filter ---
            # Compare spot lightness to immediate surroundings
            surr_r2 = patch_r * 2
            sy1_2, sy2_2 = max(0, y - surr_r2), min(h, y + surr_r2)
            sx1_2, sx2_2 = max(0, x - surr_r2), min(w, x + surr_r2)
            surr_patch = img[sy1_2:sy2_2, sx1_2:sx2_2]
            if surr_patch.size > 0:
                surr_lab = cv2.cvtColor(surr_patch, cv2.COLOR_BGR2LAB)
                mean_l_spot = float(np.mean(lab_patch[:,:,0]))
                mean_l_surr = float(np.mean(surr_lab[:,:,0]))
                # Strengthened dark lesion filter (10 L-points, 0.2 redness)
                if mean_l_spot < (mean_l_surr - 10) and redness < 0.2:
                    continue

            # --- Beard/Hair Rejection ---
            # Reject dark, gray/low-saturation patches (likely hair follicles/shadows)
            hsv_patch = cv2.cvtColor(patch_orig, cv2.COLOR_BGR2HSV)
            mean_s = float(np.mean(hsv_patch[:,:,1]))
            mean_v = float(np.mean(hsv_patch[:,:,2]))
            if mean_s < 25 and mean_v < 80:
                continue

        ml_score = 0.5
        if acne_ml_model and patch_orig.size > 0:
            p_v = cv2.resize(patch_orig, (64, 64))
            feats = np.hstack([extract_hog_features(p_v), extract_color_histogram(p_v)]).reshape(1, -1)
            ml_score = float(acne_ml_model.predict_proba(feats)[0][1])

        rest_score = (min(1.0, candidate.get("circularity", 0.5)) + (1.0 if 10<candidate["area"]<1000 else 0.5)) / 2.0
        # Weights: ML Model is primary (0.75), Redness is soft signal (0.15)
        confidence = (ml_score * 0.75 + redness * 0.15 + texture * 0.05 + rest_score * 0.05)
        
        if ml_score > 0.55:
            if confidence < 0.6: 
                confidence = max(confidence, ml_score * 0.9)
            spot_data = {"x": x, "y": y, "r": r, "confidence": confidence, "zone": assigned_zone, "ml_score": ml_score}
            confirmed_candidates.append(spot_data)

    # --- Step: Non-Maximum Suppression (NMS) ---
    # Hard-remove any spots with confidence < 0.40 before NMS and counting
    confirmed_candidates = [s for s in confirmed_candidates if s["confidence"] >= 0.40]
    
    final_spots = []
    if confirmed_candidates:
        confirmed_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        while confirmed_candidates:
            best = confirmed_candidates.pop(0)
            final_spots.append(best)
            zone_counts[best["zone"]] += 1
            
            remaining = []
            for s in confirmed_candidates:
                iou = _calculate_iou_circles(best["x"], best["y"], best["r"], s["x"], s["y"], s["r"])
                dist = np.sqrt((best["x"] - s["x"])**2 + (best["y"] - s["y"])**2)
                # Suppress if IoU > 0.15 OR center is within 1.5 * radius of best
                if iou < 0.15 and dist > (best["r"] * 1.5):
                    remaining.append(s)
            confirmed_candidates = remaining

    # Cap at 30 spots in fallback mode to prevent over-detection
    if fallback_mode and len(final_spots) > 30:
        final_spots = final_spots[:30]

    annotated = img.copy()
    if not fallback_mode and landmarks:
        for idx in FACE_OVAL_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                cv2.circle(annotated, (int(lm.x * w), int(lm.y * h)), 1, (100, 200, 100), -1)

    ZONE_DRAW_COLORS = {
        "forehead": (180, 130, 255), "left_cheek": (50, 180, 255), "right_cheek": (50, 80, 255), "cheek": (50, 180, 255), 
        "nose": (255, 180, 50), "chin": (50, 220, 130), "cropped_skin": (147, 20, 255), "other": (150, 150, 150)
    }
    # Removed overlay tint block to avoid color casting

    for zname, poly in zone_polys.items():
        if zname != "cropped_skin":
            # Use subtle (180, 180, 180) gray for all zone borders
            cv2.polylines(annotated, [poly], True, (180, 180, 180), 1)

    label_positions = []
    for spot in final_spots:
        colour = (0, 0, 255) if spot["confidence"] >= 0.6 else (0, 165, 255)
        # Use a minimum draw radius of 6 pixels
        cv2.circle(annotated, (spot["x"], spot["y"]), max(spot["r"], 6), colour, 2 if spot["confidence"] >= 0.4 else 1)
        
        # Only draw labels for higher confidence spots; use collision detection
        if spot["confidence"] >= 0.6:
            # Format zone name: "left_cheek" -> "Left Cheek"
            label_zone = spot["zone"].replace("_", " ").title()
            label_text = f"{int(spot['confidence']*100)}% {label_zone}"
            lx, ly = spot["x"] + spot["r"] + 3, spot["y"] - 2
            
            # Avoid overlapping labels (minimum 40px center-to-center)
            too_close = False
            for prev_x, prev_y in label_positions:
                if np.sqrt((lx - prev_x)**2 + (ly - prev_y)**2) < 40:
                    too_close = True
                    break
            
            if not too_close:
                cv2.putText(annotated, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
                label_positions.append((lx, ly))

    acne_count = len(final_spots)
    severity = "Clear" if acne_count == 0 else "Mild" if acne_count <= 5 else "Moderate" if acne_count <= 15 else "Severe"
    result_text = "No Acne Detected" if acne_count == 0 else f"Acne Detected — {severity} ({acne_count} spots)"
    
    zone_info = {
        "severity": severity, "total_spots": acne_count, "zones": zone_counts, 
        "active_zones": {k: v for k, v in zone_counts.items() if v > 0}, 
        "visible_zones": list(zone_polys.keys())
    }
    return result_text, annotated, [(s["x"], s["y"], s["r"]) for s in final_spots], not fallback_mode, zone_info


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = None
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    elif 'image_base64' in request.form:
        img_data = request.form['image_base64']
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)

    if img is None: return jsonify({"error": "No image provided"}), 400

    try:
        result, annotated, spots, face_detected, zone_info = detect_acne_enhanced(img)
        _, buf = cv2.imencode('.jpg', annotated)
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
        return jsonify({"result": result, "spots": spots, "face_detected": face_detected, "zone_info": zone_info, "annotated_image": annotated_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
