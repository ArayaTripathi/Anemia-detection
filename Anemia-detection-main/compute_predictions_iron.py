import os, json
from model_service import predict_anemia_sample

base_dir = os.path.dirname(__file__)
test_dir = os.path.join(base_dir, "test_iron")

results = []
if os.path.exists(test_dir):
    for root, _, files in os.walk(test_dir):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, fname)
                try:
                    pred = predict_anemia_sample(img_path)
                    results.append({
                        "file": fname,
                        "predicted_class": pred["predicted_class"],
                        "confidence": pred["confidence"]
                    })
                except Exception as e:
                    results.append({"file": fname, "error": str(e)})

print(json.dumps(results, indent=2))
