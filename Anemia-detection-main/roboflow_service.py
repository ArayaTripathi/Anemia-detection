"""
Roboflow Hosted Inference Service for Anemia Detection.

Replaces local .h5 / PyTorch models with hosted cloud inference:
- Model ID: tri-blood-iw3al/2
- API URL: https://serverless.roboflow.com
- SDK: inference-sdk
- Inference method: InferenceHTTPClient.infer()
"""

import os
import json
import logging
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# Attempt to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Search in current directory, file's directory, and parent directory
    curr_dir = Path(__file__).resolve().parent
    for env_path in [curr_dir / ".env", curr_dir.parent / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
except ImportError:
    pass

logger = logging.getLogger(__name__)

ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "tri-blood-iw3al/2")

# Cache client instance
_CLIENT = None
_CACHED_KEY = None


def get_roboflow_client(api_key: str = None):
    """
    Returns a configured InferenceHTTPClient.
    Uses ROBOFLOW_API_KEY from environment or passed argument.
    """
    global _CLIENT, _CACHED_KEY

    key = api_key or os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Roboflow API key is not configured. "
            "Please set the ROBOFLOW_API_KEY environment variable or create a .env file with ROBOFLOW_API_KEY=your_key."
        )

    if _CLIENT is not None and _CACHED_KEY == key:
        return _CLIENT

    from inference_sdk import InferenceHTTPClient, InferenceConfiguration

    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=key
    ).configure(
        InferenceConfiguration(
            api_key_transport="header"
        )
    )

    _CLIENT = client
    _CACHED_KEY = key
    return _CLIENT


# Mapping cell morphology types to standard Anemia diagnosis categories
MORPHOLOGY_TO_ANEMIA_MAP = {
    # Sickle cell forms
    "sickle cell": "Sickle Cell Anemia",
    "sickle": "Sickle Cell Anemia",
    "sickle cell anemia": "Sickle Cell Anemia",
    "drepanocyte": "Sickle Cell Anemia",
    "howell-jolly": "Sickle Cell Anemia",
    "howell jolly body": "Sickle Cell Anemia",

    # Thalassemia indicators
    "target cell": "Thalassemia",
    "target": "Thalassemia",
    "thalassemia": "Thalassemia",
    "codocyte": "Thalassemia",
    "teardrop cell": "Thalassemia",
    "dacrocyte": "Thalassemia",
    "basophilic stippling": "Thalassemia",
    "nucleated rbc": "Thalassemia",
    "nrbc": "Thalassemia",

    # Iron Deficiency Anemia indicators
    "iron deficiency": "Iron Deficiency Anemia",
    "iron deficiency anemia": "Iron Deficiency Anemia",
    "microcyte": "Iron Deficiency Anemia",
    "hypochromic": "Iron Deficiency Anemia",
    "hypochromia": "Iron Deficiency Anemia",
    "pencil cell": "Iron Deficiency Anemia",
    "elliptocyte": "Iron Deficiency Anemia",
    "ovalocyte": "Iron Deficiency Anemia",

    # Vitamin B12 / Megaloblastic indicators
    "vitamin b12 deficiency": "Vitamin B12 Deficiency (Megaloblastic)",
    "vitamin b12 deficiency (megaloblastic)": "Vitamin B12 Deficiency (Megaloblastic)",
    "megaloblastic": "Vitamin B12 Deficiency (Megaloblastic)",
    "macrocyte": "Vitamin B12 Deficiency (Megaloblastic)",
    "macrocytosis": "Vitamin B12 Deficiency (Megaloblastic)",
    "hypersegmented neutrophil": "Vitamin B12 Deficiency (Megaloblastic)",

    # Dimorphic indicators
    "dimorphic": "Dimorphic Anemia",
    "dimorphic anemia": "Dimorphic Anemia",
    "schistocyte": "Dimorphic Anemia",

    # Normal cells
    "normal rbc": "Normal RBC",
    "normal": "Normal RBC",
    "rbc": "Normal RBC",
    "platelet": "Platelet",
    "wbc": "White Blood Cell",
}


def normalize_class_name(raw_class_name: str) -> str:
    """Normalizes raw model class labels to standard clinical presentation names."""
    if not raw_class_name:
        return "Unknown"
    cleaned = raw_class_name.strip()
    lower_cleaned = cleaned.lower().replace("_", " ").replace("-", " ")
    
    # Exact or substring match in mapping
    if lower_cleaned in MORPHOLOGY_TO_ANEMIA_MAP:
        return MORPHOLOGY_TO_ANEMIA_MAP[lower_cleaned]
    
    for key, val in MORPHOLOGY_TO_ANEMIA_MAP.items():
        if key in lower_cleaned:
            return val
            
    # Default: Title Cased original label
    return cleaned.title()


def draw_bounding_boxes(image_path: str, predictions: list, output_path: str) -> str:
    """
    Draws bounding boxes, labels, and confidence tags on a copy of the image.
    Uses PIL ImageDraw.
    """
    try:
        with Image.open(image_path) as img:
            annotated = img.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            img_w, img_h = annotated.size

            # Distinct palette for class bounding boxes
            palette = [
                "#e63946", "#1d3557", "#2a9d8f", "#e76f51",
                "#457b9d", "#7209b7", "#3a0ca3", "#4cc9f0"
            ]
            class_colors = {}

            # Attempt to load a default font
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            for i, pred in enumerate(predictions):
                raw_cls = pred.get("class", "Detections")
                conf = pred.get("confidence", 0.0)
                conf_pct = conf * 100.0 if conf <= 1.0 else conf

                if raw_cls not in class_colors:
                    class_colors[raw_cls] = palette[len(class_colors) % len(palette)]
                color = class_colors[raw_cls]

                # Coordinates in Roboflow format: center x, center y, width, height
                x = pred.get("x", 0)
                y = pred.get("y", 0)
                w = pred.get("width", 0)
                h = pred.get("height", 0)

                x0 = max(0, x - w / 2)
                y0 = max(0, y - h / 2)
                x1 = min(img_w, x + w / 2)
                y1 = min(img_h, y + h / 2)

                # Draw outer box
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                # Label text
                label = f"{raw_cls}: {conf_pct:.1f}%"
                
                # Draw text background pill
                try:
                    text_bbox = draw.textbbox((x0, max(0, y0 - 15)), label, font=font)
                    draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1], fill=color)
                    draw.text((x0, max(0, y0 - 15)), label, fill="white", font=font)
                except Exception:
                    draw.text((x0, y0), label, fill=color, font=font)

            annotated.save(output_path, quality=95)
            return output_path
    except Exception as e:
        logger.warning(f"Failed to draw bounding boxes on image: {e}")
        return image_path


def infer_with_roboflow(image_path: str, model_id: str = None) -> dict:
    """
    Executes inference using the Roboflow InferenceHTTPClient.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    target_model_id = model_id or ROBOFLOW_MODEL_ID
    client = get_roboflow_client()

    try:
        result = client.infer(
            image_path,
            model_id=target_model_id
        )
        return result
    except Exception as e:
        err_str = str(e)
        logger.error(f"Roboflow inference error: {err_str}", exc_info=True)
        # Handle specific error scenarios gracefully
        if "401" in err_str or "unauthorized" in err_str.lower() or "forbidden" in err_str.lower():
            raise RuntimeError("Roboflow Authentication Failed: Invalid or unauthorized ROBOFLOW_API_KEY. Please verify your API key.")
        elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
            raise TimeoutError("Roboflow API Timeout: The serverless inference request timed out. Please try again.")
        elif "connection" in err_str.lower() or "failed to establish a new connection" in err_str.lower():
            raise ConnectionError(f"Roboflow Network Error: Could not connect to {ROBOFLOW_API_URL}. Check your internet connection.")
        else:
            raise RuntimeError(f"Roboflow Inference Error: {err_str}")


def parse_roboflow_predictions(result: dict, image_path: str = None, generate_annotated_image: bool = True) -> dict:
    """
    Dynamically parses Roboflow responses (Object Detection, Classification, or Keypoint/Segmentation).
    Returns standardized format expected by the frontend.
    """
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected prediction result type: {type(result)}")

    raw_preds = result.get("predictions")
    detections = []
    detection_counts = Counter()
    all_probabilities = []
    has_bounding_boxes = False
    annotated_image_path = None

    # Case 1: Object Detection (predictions is a list of dicts with bounding boxes)
    if isinstance(raw_preds, list) and len(raw_preds) > 0 and isinstance(raw_preds[0], dict) and "x" in raw_preds[0]:
        has_bounding_boxes = True
        total_detections = len(raw_preds)
        
        # Accumulate per-class confidences and counts
        class_confidences = {}
        for p in raw_preds:
            raw_cls = p.get("class", "Cell")
            conf = float(p.get("confidence", 0.0))
            norm_cls = normalize_class_name(raw_cls)
            detection_counts[norm_cls] += 1
            if norm_cls not in class_confidences:
                class_confidences[norm_cls] = []
            class_confidences[norm_cls].append(conf)

            detections.append({
                "class": raw_cls,
                "normalized_class": norm_cls,
                "confidence": round(conf * 100.0 if conf <= 1.0 else conf, 2),
                "x": p.get("x", 0),
                "y": p.get("y", 0),
                "width": p.get("width", 0),
                "height": p.get("height", 0)
            })

        # Calculate primary class
        # Filter for abnormal cell types first if present
        abnormal_counts = {k: v for k, v in detection_counts.items() if k not in ["Normal RBC", "Platelet", "White Blood Cell"]}
        if abnormal_counts:
            dominant_class = max(abnormal_counts.keys(), key=lambda k: abnormal_counts[k])
        else:
            dominant_class = detection_counts.most_common(1)[0][0]

        dominant_conf_list = class_confidences.get(dominant_class, [1.0])
        avg_conf = (sum(dominant_conf_list) / len(dominant_conf_list))
        confidence_val = avg_conf * 100.0 if avg_conf <= 1.0 else avg_conf

        # Build all_probabilities breakdown based on detected proportions
        for cls_name, count in detection_counts.most_common():
            proportion = (count / total_detections) * 100.0
            avg_c = sum(class_confidences[cls_name]) / count
            all_probabilities.append({
                "name": cls_name,
                "probability": round(proportion, 2),
                "raw": round(avg_c if avg_c <= 1.0 else avg_c / 100.0, 4),
                "count": count
            })

        # Generate annotated image if requested
        if generate_annotated_image and image_path and os.path.exists(image_path):
            base_dir, filename = os.path.split(image_path)
            name_part, ext = os.path.splitext(filename)
            out_filename = f"{name_part}_annotated{ext}"
            out_path = os.path.join(base_dir, out_filename)
            annotated_image_path = draw_bounding_boxes(image_path, raw_preds, out_path)

    # Case 2: Classification (predictions is list of class/confidence dicts)
    elif isinstance(raw_preds, list) and len(raw_preds) > 0 and isinstance(raw_preds[0], dict):
        total_detections = len(raw_preds)
        for p in raw_preds:
            cls_name = normalize_class_name(p.get("class", "Unknown"))
            conf = float(p.get("confidence", 0.0))
            conf_pct = conf * 100.0 if conf <= 1.0 else conf
            all_probabilities.append({
                "name": cls_name,
                "probability": round(conf_pct, 2),
                "raw": round(conf if conf <= 1.0 else conf / 100.0, 4)
            })

        all_probabilities.sort(key=lambda x: x["probability"], reverse=True)
        dominant_class = all_probabilities[0]["name"]
        confidence_val = all_probabilities[0]["probability"]

    # Case 3: Classification dict {ClassName: {confidence: 0.95}}
    elif isinstance(raw_preds, dict):
        total_detections = len(raw_preds)
        for raw_cls, details in raw_preds.items():
            conf = float(details.get("confidence", 0.0) if isinstance(details, dict) else details)
            conf_pct = conf * 100.0 if conf <= 1.0 else conf
            all_probabilities.append({
                "name": normalize_class_name(raw_cls),
                "probability": round(conf_pct, 2),
                "raw": round(conf if conf <= 1.0 else conf / 100.0, 4)
            })
        all_probabilities.sort(key=lambda x: x["probability"], reverse=True)
        dominant_class = all_probabilities[0]["name"] if all_probabilities else "Normal RBC"
        confidence_val = all_probabilities[0]["probability"] if all_probabilities else 100.0

    # Case 4: Top-level classification or empty predictions
    else:
        top_name = result.get("top")
        top_conf = result.get("confidence")
        if top_name:
            dominant_class = normalize_class_name(str(top_name))
            c = float(top_conf) if top_conf is not None else 1.0
            confidence_val = c * 100.0 if c <= 1.0 else c
            all_probabilities = [{
                "name": dominant_class,
                "probability": round(confidence_val, 2),
                "raw": round(c if c <= 1.0 else c / 100.0, 4)
            }]
            total_detections = 1
        else:
            # Empty predictions: no abnormal cells detected
            dominant_class = "Normal (No Abnormalities Detected)"
            confidence_val = 100.0
            all_probabilities = [{
                "name": "Normal Erythrocytes",
                "probability": 100.0,
                "raw": 1.0
            }]
            total_detections = 0

    return {
        "predicted_class": dominant_class,
        "confidence": round(confidence_val, 2),
        "confidence_str": f"{confidence_val:.2f}%",
        "all_probabilities": all_probabilities,
        "detections": detections,
        "total_detections": total_detections,
        "detection_counts": dict(detection_counts),
        "has_bounding_boxes": has_bounding_boxes,
        "annotated_image_path": annotated_image_path,
        "raw_response": result
    }

