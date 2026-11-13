import os, json
from model_service import predict_anemia_sample, reset_model_cache

# Directory where the zip was extracted
base_dir = os.path.dirname(__file__)
test_dir = os.path.join(base_dir, "test_dataset")

results = []
for root, _, files in os.walk(test_dir):
    for fname in files:
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, fname)
            pred = predict_anemia_sample(img_path, h5_path='M_Anemia_finetuned.h5')
            results.append({
                "file": fname,
                "predicted_class": pred["predicted_class"],
                "confidence": pred["confidence"]
            })

print(json.dumps(results, indent=2))

