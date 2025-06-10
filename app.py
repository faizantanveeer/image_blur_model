from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


def is_blurry(image):
    resized = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Sobel gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_var = np.var(sobelx) + np.var(sobely)

    height, width = gray.shape
    size_factor = (height * width) / (640 * 480)  # normalized to VGA
    adjusted_threshold = 150 * size_factor

    print("threshold = ", adjusted_threshold)
    print(f"Blur = {laplacian_var}, sobel = {sobel_var}")

    return laplacian_var < adjusted_threshold or sobel_var < 200


def is_relevant(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray)

    print("Relevancy = ", mean_intensity)
    return mean_intensity > 50  # Adjust threshold based on your needs


@app.route("/check-image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    np_image = np.frombuffer(file.read(), np.uint8)

    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid Image"}), 400

    result = {
        "is_blurry": bool(is_blurry(image)),
        "is_relevant": bool(is_relevant(image)),
    }

    if result["is_blurry"]:
        result["flag"] = "Image is blurry"
    elif not result["is_relevant"]:
        result["flag"] = "Image is not relevant"
    else:
        result["flag"] = "Image is clear and relevant"

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
