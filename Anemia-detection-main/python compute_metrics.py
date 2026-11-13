import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from model_service import predict_image  # your inference wrapper
from tensorflow.keras.utils import load_img, img_to_array

# -------------------------------------------------
# Adjust these paths to point to your true test set
TEST_DIR = "test_images"          # folder with subfolders per class
CLASSES = ["DA", "IDA", "SCA", "TH", "VB12"]
# -------------------------------------------------

y_true, y_pred = [], []

for cls in CLASSES:
    cls_dir = os.path.join(TEST_DIR, cls)
    for fname in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, fname)
        pred = predict_image(img_path)   # returns class label string
        y_true.append(cls)
        y_pred.append(pred)

print("Overall accuracy:", accuracy_score(y_true, y_pred))
print("\nPer‑class report:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))