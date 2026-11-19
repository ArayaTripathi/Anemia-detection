import json, os
from model_service import predict_anemia_sample

base = os.path.join(os.path.dirname(__file__), 'test_dataset')
files = ['Thalassemia_6.JPG', 'Thalassemia_7.JPG', 'Thalassemia_8.JPG']
results = []
for f in files:
    path = os.path.join(base, f)
    if os.path.exists(path):
        try:
            r = predict_anemia_sample(path)
            results.append({'file': f, 'predicted_class': r['predicted_class'], 'confidence': r['confidence']})
        except Exception as e:
            results.append({'file': f, 'error': str(e)})
    else:
        results.append({'file': f, 'error': 'not found'})
print(json.dumps(results, indent=2))
