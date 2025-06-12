import requests
import os
from tabulate import tabulate

# === CONFIG ===
ROOT_DIR = "test_images"  # This should have 'sharp' and 'defocused_blurred' folders
API_URL = "http://127.0.0.1:5000/check-image"

results = []
correct = 0
total = 0

for category in ["sharp", "defocused_blurred"]:
    folder_path = os.path.join(ROOT_DIR, category)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "rb") as img_file:
                try:
                    response = requests.post(API_URL, files={"image": img_file})
                except requests.exceptions.RequestException as e:
                    results.append(
                        [filename, category, "Error", "-", "-", "Request Failed"]
                    )
                    continue

            if response.status_code == 200:
                data = response.json()
                predicted = "blurry" if data["is_blurry"] else "sharp"
                ground_truth = "blurry" if category == "defocused_blurred" else "sharp"
                is_correct = predicted == ground_truth

                results.append(
                    [
                        filename,
                        category,
                        predicted,
                        data["blurriness_score"],
                        data["flag"],
                        "✅" if is_correct else "❌",
                    ]
                )

                total += 1
                correct += 1 if is_correct else 0
            else:
                results.append([filename, category, "Error", "-", "-", "API Error"])
                total += 1

# === TABLE PRINT ===
headers = ["Image Name", "Actual Folder", "Predicted", "Score", "Flag", "Correct?"]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

# === SHORT REPORT ===
accuracy = (correct / total) * 100 if total else 0
print("\n" + "=" * 50)
print(f"✅ TOTAL IMAGES TESTED: {total}")
print(f"🎯 CORRECT PREDICTIONS : {correct}")
print(f"📊 ACCURACY            : {accuracy:.2f}%")
print("=" * 50)
