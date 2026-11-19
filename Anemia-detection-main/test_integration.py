"""
Automated Integration and Verification Test Suite for Roboflow Anemia Detection.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roboflow_service import (
    get_roboflow_client,
    parse_roboflow_predictions,
    draw_bounding_boxes,
    normalize_class_name
)
import model_service
from app import app


class TestRoboflowIntegration(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        # Create a sample test image
        self.test_img_path = "uploads/test_sample_smear.jpg"
        os.makedirs("uploads", exist_ok=True)
        img = Image.new("RGB", (300, 300), color=(220, 200, 210))
        img.save(self.test_img_path)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
        annotated = "uploads/test_sample_smear_annotated.jpg"
        if os.path.exists(annotated):
            os.remove(annotated)

    def test_missing_api_key_error(self):
        """Verify that missing API key raises informative ValueError."""
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                get_roboflow_client()
            self.assertIn("ROBOFLOW_API_KEY", str(ctx.exception))

    def test_class_normalization(self):
        """Verify morphology terms map to standardized clinical categories."""
        self.assertEqual(normalize_class_name("sickle cell"), "Sickle Cell Anemia")
        self.assertEqual(normalize_class_name("target_cell"), "Thalassemia")
        self.assertEqual(normalize_class_name("microcyte"), "Iron Deficiency Anemia")
        self.assertEqual(normalize_class_name("macrocyte"), "Vitamin B12 Deficiency (Megaloblastic)")
        self.assertEqual(normalize_class_name("normal rbc"), "Normal RBC")

    def test_parse_object_detection_response(self):
        """Verify dynamic parsing of object detection responses with bounding boxes."""
        mock_detection_result = {
            "time": 0.04,
            "image": {"width": 640, "height": 640},
            "predictions": [
                {"x": 100, "y": 100, "width": 40, "height": 40, "class": "Sickle Cell", "confidence": 0.95},
                {"x": 200, "y": 200, "width": 45, "height": 45, "class": "Sickle Cell", "confidence": 0.91},
                {"x": 300, "y": 300, "width": 50, "height": 50, "class": "Normal RBC", "confidence": 0.88}
            ]
        }
        parsed = parse_roboflow_predictions(mock_detection_result, image_path=self.test_img_path)
        self.assertEqual(parsed["predicted_class"], "Sickle Cell Anemia")
        self.assertEqual(parsed["total_detections"], 3)
        self.assertTrue(parsed["has_bounding_boxes"])
        self.assertIsNotNone(parsed["annotated_image_path"])
        self.assertTrue(os.path.exists(parsed["annotated_image_path"]))
        self.assertEqual(len(parsed["all_probabilities"]), 2)

    def test_parse_classification_response(self):
        """Verify dynamic parsing of classification responses."""
        mock_classification_result = {
            "predictions": [
                {"class": "Thalassemia", "confidence": 0.89},
                {"class": "Iron Deficiency Anemia", "confidence": 0.08},
                {"class": "Normal RBC", "confidence": 0.03}
            ]
        }
        parsed = parse_roboflow_predictions(mock_classification_result, image_path=self.test_img_path)
        self.assertEqual(parsed["predicted_class"], "Thalassemia")
        self.assertEqual(parsed["confidence"], 89.0)
        self.assertFalse(parsed["has_bounding_boxes"])

    def test_no_h5_file_dependency_during_prediction(self):
        """Verify that no .h5 file is accessed during prediction."""
        mock_roboflow_response = {
            "predictions": [
                {"class": "Iron Deficiency Anemia", "confidence": 0.92}
            ]
        }
        with patch("model_service.infer_with_roboflow", return_value=mock_roboflow_response):
            result = model_service.predict_anemia_sample(
                self.test_img_path,
                hemoglobin="8.5",
                mcv="72.0",
                rbc_count="3.8"
            )
            self.assertEqual(result["predicted_class"], "Iron Deficiency Anemia")
            self.assertEqual(result["confidence"], 92.0)
            self.assertTrue(len(result["cbc_corroboration"]) > 0)
            # Ensure no .h5 attributes exist on model_service
            self.assertFalse(hasattr(model_service, "load_anemia_model"))
            self.assertFalse(hasattr(model_service, "AnemiaVGGModel"))

    def test_frontend_routes_exist_and_render(self):
        """Verify frontend pages load with 200 OK."""
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"Anemia Detection System", res_index.data)
        self.assertIn(b"Patient & Blood Smear Input", res_index.data)

        res_home = self.client.get("/home")
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"Disease Guide", res_home.data)

        res_about = self.client.get("/about")
        self.assertEqual(res_about.status_code, 200)
        self.assertIn(b"tri-blood-iw3al/2", res_about.data)

    def test_predict_endpoint_with_mock_inference(self):
        """Test full prediction flow through POST /predict."""
        mock_roboflow_response = {
            "predictions": [
                {"x": 150, "y": 150, "width": 50, "height": 50, "class": "Sickle Cell", "confidence": 0.94}
            ]
        }
        with patch("model_service.infer_with_roboflow", return_value=mock_roboflow_response):
            with open(self.test_img_path, "rb") as f:
                response = self.client.post("/predict", data={
                    "name": "Jane Doe",
                    "age": "28",
                    "gender": "Female",
                    "blood_type": "B+",
                    "hemoglobin": "8.0",
                    "mcv": "85.0",
                    "rbc_count": "3.1",
                    "mch": "26.0",
                    "image": (f, "sample_smear.jpg")
                })
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Sickle Cell Anemia", response.data)
            self.assertIn(b"94.00%", response.data)
            self.assertIn(b"Roboflow", response.data)

    def test_predict_endpoint_missing_api_key_graceful_handling(self):
        """Test that missing API key shows clean alert instead of crashing."""
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": ""}, clear=True):
            with open(self.test_img_path, "rb") as f:
                response = self.client.post("/predict", data={
                    "name": "Jane Doe",
                    "age": "28",
                    "gender": "Female",
                    "blood_type": "B+",
                    "image": (f, "sample_smear.jpg")
                })
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Validation Alert:", response.data)
            self.assertIn(b"ROBOFLOW_API_KEY", response.data)

    def test_security_api_key_not_in_rendered_html(self):
        """Verify API key is never leaked into rendered HTML or client payload."""
        secret_key = "super_secret_rf_test_key_12345"
        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": secret_key}):
            mock_roboflow_response = {
                "predictions": [
                    {"class": "Thalassemia", "confidence": 0.91}
                ]
            }
            with patch("model_service.infer_with_roboflow", return_value=mock_roboflow_response):
                with open(self.test_img_path, "rb") as f:
                    response = self.client.post("/predict", data={
                        "name": "Test Patient",
                        "age": "30",
                        "gender": "Male",
                        "image": (f, "sample.jpg")
                    })
                self.assertNotIn(secret_key.encode("utf-8"), response.data)


if __name__ == "__main__":
    unittest.main()
