from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
IMG_SIZE = 224

model = tf.keras.models.load_model("models/blur_classifier.keras")


model.summary()


def preprocess_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)

    print("Input tensor mean:", np.mean(img))

    return np.expand_dims(img, axis=0)


@app.route("/check-image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"].read()
    input_tensor = preprocess_image(image)
    print("Tenser = ", input_tensor)
    prediction = model(input_tensor, training=False).numpy()[0][0]
    print("Prediction = ", prediction)

    result = {
        "confidence": float(prediction),
        "is_blurry": bool(prediction > 0.3),
        "flag": "Image is blurry" if prediction > 0.3 else "Image is sharp",
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
