from flask import Flask, request, jsonify
import numpy as np
import cv2
import logging

app = Flask(__name__)

# === Configuration ===
IMAGE_PATCH_SIZE = 20
BLURRINESS_THRESHOLD = 15.0

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)


def calculate_blurriness(image_bytes):
    try:
        # Decode image bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_16S)
        abs_laplacian = cv2.convertScaleAbs(laplacian)

        # Find the location with maximum edge intensity
        _, _, _, max_loc = cv2.minMaxLoc(abs_laplacian)
        h, w = abs_laplacian.shape
        radius = IMAGE_PATCH_SIZE if w > 2 * IMAGE_PATCH_SIZE else int(w / 2)

        # Define patch coordinates
        x1 = max(0, int(max_loc[0] - radius))
        x2 = min(w - 1, x1 + 2 * radius)
        if x2 >= w:
            x2 = w - 1
            x1 = max(0, x2 - 2 * radius)

        y1 = max(0, int(max_loc[1] - radius))
        y2 = min(h - 1, y1 + 2 * radius)
        if y2 >= h:
            y2 = h - 1
            y1 = max(0, y2 - 2 * radius)

        patch = abs_laplacian[y1:y2, x1:x2]
        std_dev = np.std(patch)
        return round(std_dev, 2)

    except Exception as e:
        logging.exception("Error calculating blurriness")
        raise


@app.route("/check-image", methods=["POST"])
def check_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({"error": "Empty image file"}), 400

        score = calculate_blurriness(image_bytes)
        is_blurry = score < BLURRINESS_THRESHOLD

        return (
            jsonify(
                {
                    "blurriness_score": score,
                    "is_blurry": bool(is_blurry),
                    "flag": "Image is blurry" if is_blurry else "Image is sharp",
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception("Failed to process image")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
