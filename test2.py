import requests
import os
from tabulate import tabulate

# === CONFIG ===
IMAGE_DIR = "motion_blur"  # All images are directly in this folder
API_URL = "http://127.0.0.1:5000/check-image"

# === COLLECT RESULTS ===
results = []
total = 0
blurry_count = 0
sharp_count = 0

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(IMAGE_DIR, filename)
        with open(filepath, "rb") as img_file:
            response = requests.post(API_URL, files={"image": img_file})

        if response.status_code == 200:
            data = response.json()
            predicted_label = "Blurry" if data["is_blurry"] else "Sharp"

            if predicted_label == "Blurry":
                blurry_count += 1
            else:
                sharp_count += 1

            total += 1
            results.append(
                [
                    filename,
                    round(data.get("blurriness_score", 0), 2),
                    round(data.get("directionality", 0), 2),
                    predicted_label,
                    data.get("blur_type", "-"),
                    data["flag"],
                ]
            )
        else:
            total += 1
            results.append(
                [filename, "Error", "Error", "Error", "-", "Failed to process"]
            )

# === PRINT TABLE ===
headers = [
    "Image",
    "Blur Score",
    "Directionality",
    "Predicted",
    "Type",
    "Flag",
]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

# === REPORT ===
print("\n" + "=" * 60)
print(f"📂 TOTAL IMAGES TESTED  : {total}")
print(f"🌀 PREDICTED AS BLURRY  : {blurry_count}")
print(f"🧊 PREDICTED AS SHARP   : {sharp_count}")
print("=" * 60)
