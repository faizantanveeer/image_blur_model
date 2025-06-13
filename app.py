from flask import Flask, request, jsonify
import numpy as np
import cv2
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# === Final thresholds ===
BLURRINESS_THRESHOLD = 7.5
MOTION_DIRECTIONALITY_THRESHOLD = 1.5
MOTION_BLUR_SCORE_UPPER_LIMIT = 10
MIN_EDGE_PIXELS = 500  # avoid false motion detection in plain images


@app.route("/check-image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = request.files["image"].read()
    try:
        result = analyze_image(image_bytes)
        return jsonify(result), 200
    except Exception as e:
        logging.exception("Error processing image")
        return jsonify({"error": str(e)}), 500


def analyze_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")

    # === Resize huge images ===
    image = resize_image(image, max_dim=512)

    # === Convert to grayscale (handle grayscale inputs too)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # === Normalize contrast to handle over/under-exposure
    gray = cv2.equalizeHist(gray)

    # === Light Gaussian Blur to suppress noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # === Blurriness Score ===
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.std(np.abs(laplacian))

    # === Directionality (only for high-texture images)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)

    mag_thresh = np.percentile(magnitude, 95)
    strong_angles = angle[magnitude > mag_thresh]
    edge_pixel_count = np.sum(magnitude > mag_thresh)

    # Only trust directionality if there are enough strong edges
    if edge_pixel_count >= MIN_EDGE_PIXELS:
        angle_std = np.std(strong_angles)
    else:
        angle_std = 0

    # === Final decision
    is_blurry = blur_score < BLURRINESS_THRESHOLD or (
        angle_std > MOTION_DIRECTIONALITY_THRESHOLD
        and blur_score < MOTION_BLUR_SCORE_UPPER_LIMIT
    )

    return {
        "blurriness_score": round(blur_score, 2),
        "directionality": round(angle_std, 2),
        "is_blurry": bool(is_blurry),
        "flag": "Image is blurry" if is_blurry else "Image is sharp",
        "edge_pixels_used": int(edge_pixel_count),
    }


def resize_image(img, max_dim=512):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


if __name__ == "__main__":
    app.run(debug=True)
