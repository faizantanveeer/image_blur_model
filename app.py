from flask import Flask, request, jsonify
import numpy as np
import cv2
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# === Thresholds (Tuned based on your data) ===
BLURRINESS_THRESHOLD = 10.0  # Defocus blur threshold
DIRECTIONALITY_STD_THRESHOLD = 0.25  # Lower = more directional (motion blur)


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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # === Blurriness Score (Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.std(np.abs(laplacian))

    # === Edge Direction Analysis (for motion blur)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)

    # Focus only on strong edges
    mag_thresh = np.percentile(magnitude, 95)
    strong_angles = angle[magnitude > mag_thresh]

    # Motion blur → low std (edges mostly in one direction)
    if len(strong_angles) > 0:
        angle_std = np.std(strong_angles)
    else:
        angle_std = 1.0  # assume non-directional if no edges

    # === Final classification
    is_motion_blur = angle_std < DIRECTIONALITY_STD_THRESHOLD
    is_defocus_blur = blur_score < BLURRINESS_THRESHOLD

    if is_motion_blur:
        blur_type = "Motion"
    elif is_defocus_blur:
        blur_type = "Defocus"
    else:
        blur_type = "None"

    return {
        "blurriness_score": round(blur_score, 2),
        "directionality": round(angle_std, 2),
        "is_blurry": blur_type != "None",
        "blur_type": blur_type,
        "flag": "Image is blurry" if blur_type != "None" else "Image is sharp",
    }


if __name__ == "__main__":
    app.run(debug=True)
