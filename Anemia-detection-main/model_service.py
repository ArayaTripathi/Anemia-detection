import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CLASSES = [
    'Dimorphic Anemia',
    'Iron Deficiency Anemia',
    'Sickle Cell Anemia',
    'Thalassemia',
    'Vitamin B12 Deficiency (Megaloblastic)'
]

ANEMIA_INFO = {
    'Dimorphic Anemia': {
        'description': 'Characterized by the presence of two distinct populations of red blood cells (microcytic hypochromic and normocytic or macrocytic cells).',
        'morphology': 'Dual erythrocyte population, anisocytosis, poikilocytosis, mixed cell sizing.',
        'recommendations': 'Evaluate combined nutritional status (Serum Ferritin, Vitamin B12, and Serum Folate). Investigate underlying malabsorption or secondary causes.',
        'severity': 'Moderate to High'
    },
    'Iron Deficiency Anemia': {
        'description': 'The most common type of anemia worldwide, resulting from insufficient iron stores required for adequate hemoglobin synthesis.',
        'morphology': 'Microcytic hypochromic red blood cells with expanded central pallor, pencil cells (elliptocytes), and marked anisocytosis.',
        'recommendations': 'Assess Serum Ferritin, Total Iron Binding Capacity (TIBC), and Transferrin Saturation. Initiate oral/IV iron supplementation and identify etiology of blood loss.',
        'severity': 'Mild to Moderate'
    },
    'Sickle Cell Anemia': {
        'description': 'A hereditary hemoglobinopathy caused by HbS mutation, causing red blood cells to deform into rigid sickle/crescent shapes under low oxygen tension.',
        'morphology': 'Sickle cells (drepanocytes), boat-shaped cells, target cells, polychromasia, and Howell-Jolly bodies.',
        'recommendations': 'Confirm with Hemoglobin Electrophoresis or HPLC. Ensure hydration, pain management protocol, hydroxyurea evaluation, and hematology consultation.',
        'severity': 'Critical / Chronic'
    },
    'Thalassemia': {
        'description': 'An inherited genetic blood disorder characterized by defective or reduced synthesis of one or more globin polypeptide chains of hemoglobin.',
        'morphology': 'Marked microcytosis, target cells (codocytes), basophilic stippling, nucleated red blood cells, and teardrop cells.',
        'recommendations': 'Perform Hemoglobin Variant Analysis (HPLC), genetic testing, and iron studies to avoid mistaken iron overload from misdiagnosis.',
        'severity': 'Moderate to Severe'
    },
    'Vitamin B12 Deficiency (Megaloblastic)': {
        'description': 'Macrocytic anemia resulting from impaired DNA synthesis in erythrocyte precursors due to insufficient cobalamin (Vitamin B12) or folate.',
        'morphology': 'Oval macrocytes, hypersegmented neutrophils (>= 5 lobes), anisopoikilocytosis, and elevated MCV (> 100 fL).',
        'recommendations': 'Confirm with Serum B12, Methylmalonic Acid (MMA), and Homocysteine levels. Initiate intramuscular or high-dose oral cyanocobalamin / methylcobalamin.',
        'severity': 'Moderate to High'
    }
}

class AnemiaVGGModel(nn.Module):
    """
    Exact VGG16 architecture trained in M_Anemia.h5:
    13 Conv2D layers with ReLU activations, 5 MaxPooling2D layers,
    followed by Flatten, Dense(128, ReLU), and Dense(5, Softmax).
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(512 * 7 * 7, 128)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        feat = self.features(x)
        # Permute (N, C, H, W) -> (N, H, W, C) to match Keras Flatten row-major layout
        feat = feat.permute(0, 2, 3, 1).contiguous()
        flat = feat.view(feat.size(0), -1)
        x = self.relu(self.fc1(flat))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


_model_instance = None
_model_path = None

def reset_model_cache():
    """Clear the cached model and path so the next call reloads weights."""
    global _model_instance, _model_path
    _model_instance = None
    _model_path = None


def load_anemia_model(h5_path='M_Anemia.h5'):
    """Loads weights from an H5 (Keras) file or a PyTorch state_dict.
    Caches the model, but reloads if a different path is requested.
    """
    global _model_instance, _model_path
    # If we already have a loaded model for the requested path, reuse it
    if _model_instance is not None and _model_path == h5_path:
        return _model_instance
    # Otherwise load a new model (or reload existing one)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file '{h5_path}' not found!")

    model = AnemiaVGGModel()
    conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    conv_names = [
        'block1_conv1', 'block1_conv2',
        'block2_conv1', 'block2_conv2',
        'block3_conv1', 'block3_conv2', 'block3_conv3',
        'block4_conv1', 'block4_conv2', 'block4_conv3',
        'block5_conv1', 'block5_conv2', 'block5_conv3'
    ]

    # Try loading as a Keras‑style H5 first
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            w = f['model_weights']
            for idx, name in zip(conv_indices, conv_names):
                k = np.array(w[name][name]['kernel'])
                b = np.array(w[name][name]['bias'])
                model.features[idx].weight.data.copy_(torch.from_numpy(k.transpose(3, 2, 0, 1)))
                model.features[idx].bias.data.copy_(torch.from_numpy(b))

            d1_k = np.array(w['dense']['dense']['kernel'])
            d1_b = np.array(w['dense']['dense']['bias'])
            model.fc1.weight.data.copy_(torch.from_numpy(d1_k.T))
            model.fc1.bias.data.copy_(torch.from_numpy(d1_b))

            d2_k = np.array(w['dense_1']['dense_1']['kernel'])
            d2_b = np.array(w['dense_1']['dense_1']['bias'])
            model.fc2.weight.data.copy_(torch.from_numpy(d2_k.T))
            model.fc2.bias.data.copy_(torch.from_numpy(d2_b))
    except Exception as h5_err:
        # Fallback: assume the file is a PyTorch state_dict
        try:
            state_dict = torch.load(h5_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)
        except Exception as torch_err:
            raise RuntimeError(
                f"Failed to load model weights from '{h5_path}'. "
                f"H5 error: {h5_err}; Torch error: {torch_err}"
            )

    model.eval()
    _model_instance = model
    _model_path = h5_path
    return _model_instance


def preprocess_image_tensor(img_path_or_pil):
    """Opens and preprocesses image to (1, 3, 224, 224) normalized tensor."""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil)
    else:
        img = img_path_or_pil

    img = img.convert('RGB')
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.Resampling.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    # (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def predict_anemia_sample(img_path_or_pil, hemoglobin=None, mcv=None, rbc_count=None, mch=None, h5_path='M_Anemia.h5'):
    """
    Runs model prediction on image and contextualizes with CBC indices.
    Returns comprehensive diagnostic dictionary.
    """
    model = load_anemia_model(h5_path)
    tensor = preprocess_image_tensor(img_path_or_pil)

    with torch.no_grad():
        probs = model(tensor).numpy()[0]

    top_idx = int(np.argmax(probs))
    predicted_class = CLASSES[top_idx]
    confidence_val = float(probs[top_idx]) * 100.0

    all_probabilities = [
        {
            'name': cls_name,
            'probability': round(float(probs[i]) * 100.0, 2),
            'raw': float(probs[i])
        }
        for i, cls_name in enumerate(CLASSES)
    ]
    # Sort by probability descending
    all_probabilities.sort(key=lambda x: x['probability'], reverse=True)

    info = ANEMIA_INFO.get(predicted_class, {
        'description': 'Anemia detected via microscopic blood smear morphology.',
        'morphology': 'Abnormal red blood cell morphology.',
        'recommendations': 'Clinical review and comprehensive laboratory workup recommended.',
        'severity': 'Moderate'
    })

    # CBC correlation analysis
    cbc_corroboration = []
    cbc_warnings = []

    try:
        if hemoglobin is not None and str(hemoglobin).strip():
            hb_val = float(hemoglobin)
            if hb_val < 12.0:
                cbc_corroboration.append(f"Low Hemoglobin ({hb_val} g/dL) confirms anemic state.")
            else:
                cbc_warnings.append(f"Hemoglobin ({hb_val} g/dL) is within normal range; review smear for early-stage or trait conditions.")

        if mcv is not None and str(mcv).strip():
            mcv_val = float(mcv)
            if predicted_class in ['Iron Deficiency Anemia', 'Thalassemia']:
                if mcv_val < 80.0:
                    cbc_corroboration.append(f"Low MCV ({mcv_val} fL) supports microcytic morphology.")
                elif mcv_val > 96.0:
                    cbc_warnings.append(f"Elevated MCV ({mcv_val} fL) typically points to macrocytosis rather than microcytic anemia.")
            elif predicted_class == 'Vitamin B12 Deficiency (Megaloblastic)':
                if mcv_val > 100.0:
                    cbc_corroboration.append(f"High MCV ({mcv_val} fL) strongly supports megaloblastic/macrocytic anemia.")
                elif mcv_val < 80.0:
                    cbc_warnings.append(f"Low MCV ({mcv_val} fL) is atypical for isolated B12 deficiency; consider co-occurring iron deficiency.")

        if rbc_count is not None and str(rbc_count).strip():
            rbc_val = float(rbc_count)
            if predicted_class == 'Thalassemia' and rbc_val > 5.0 and (mcv is not None and float(mcv) < 75.0):
                cbc_corroboration.append(f"High RBC count ({rbc_val} 10^6/μL) relative to low MCV is characteristic of Thalassemia trait.")
            elif predicted_class == 'Sickle Cell Anemia' and rbc_val < 3.5:
                cbc_corroboration.append(f"Depressed RBC count ({rbc_val} 10^6/μL) is consistent with sickle cell hemolytic turnover.")

    except (ValueError, TypeError):
        pass

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence_val, 2),
        'confidence_str': f"{confidence_val:.2f}%",
        'all_probabilities': all_probabilities,
        'description': info['description'],
        'morphology': info['morphology'],
        'recommendations': info['recommendations'],
        'severity': info['severity'],
        'cbc_corroboration': cbc_corroboration,
        'cbc_warnings': cbc_warnings
    }

