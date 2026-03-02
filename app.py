import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_acne_opencv(img):
    """
    Rule-based acne detection using OpenCV.
    Finds reddish regions on the skin holding acne characteristics.
    """
    # 1. Resize for consistent processing
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define redness threshold for skin/acne
    # Red spans across 0 and 180 in OpenCV HSV, so we need two ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 4. Morphological operations to remove noise and connect spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Find contours of potential acne spots
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    acne_count = 0
    # 6. Filter contours based on size and circularity
    for contour in contours:
        area = cv2.contourArea(contour)
        # Assuming acne spots fall within a certain pixel area range
        if 10 < area < 5000: 
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                # Acne tends to be somewhat circular
                if circularity > 0.2:
                    acne_count += 1
                    
    # 7. Decision logic
    if acne_count >= 1:
        return f"Acne Detected ({acne_count} spots)"
    return "No Acne Detected"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = None

    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Handle base64 camera image
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
        result = detect_acne_opencv(img)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
