import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_service
from app import app


class TestClassSwaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.test_img = "uploads/test_swap_validation.jpg"
        os.makedirs("uploads", exist_ok=True)
        img = Image.new("RGB", (224, 224), color=(210, 190, 200))
        img.save(cls.test_img)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img):
            os.remove(cls.test_img)

    def test_internal_mapping_intact(self):
        """Ensure original internal CLASSES list and ordering are 100% intact."""
        expected_internal = [
            'Dimorphic Anemia',
            'Iron Deficiency Anemia',
            'Sickle Cell Anemia',
            'Thalassemia',
            'Vitamin B12 Deficiency (Megaloblastic)'
        ]
        self.assertEqual(model_service.CLASSES, expected_internal)

    def test_swap_1_dimorphic_to_thalassemia(self):
        """If model internally predicts Dimorphic Anemia (index 0), display Thalassemia."""
        # Simulated softmax output where index 0 has 92% confidence
        fake_probs = torch.tensor([[0.92, 0.02, 0.02, 0.02, 0.02]])
        with patch.object(model_service.AnemiaVGGModel, 'forward', return_value=fake_probs):
            res = model_service.predict_anemia_sample(self.test_img)
            self.assertEqual(res['internal_class'], 'Dimorphic Anemia')
            self.assertEqual(res['predicted_class'], 'Thalassemia')
            self.assertAlmostEqual(res['confidence'], 92.0, places=1)
            # Top probability in breakdown should be Thalassemia
            self.assertEqual(res['all_probabilities'][0]['name'], 'Thalassemia')
            self.assertAlmostEqual(res['all_probabilities'][0]['probability'], 92.0, places=1)

    def test_swap_2_thalassemia_to_dimorphic(self):
        """If model internally predicts Thalassemia (index 3), display Dimorphic Anemia."""
        fake_probs = torch.tensor([[0.01, 0.02, 0.03, 0.90, 0.04]])
        with patch.object(model_service.AnemiaVGGModel, 'forward', return_value=fake_probs):
            res = model_service.predict_anemia_sample(self.test_img)
            self.assertEqual(res['internal_class'], 'Thalassemia')
            self.assertEqual(res['predicted_class'], 'Dimorphic Anemia')
            self.assertAlmostEqual(res['confidence'], 90.0, places=1)
            self.assertEqual(res['all_probabilities'][0]['name'], 'Dimorphic Anemia')
            self.assertAlmostEqual(res['all_probabilities'][0]['probability'], 90.0, places=1)

    def test_swap_3_vitamin_b12_to_iron_deficiency(self):
        """If model internally predicts Vitamin B12 Deficiency (index 4), display Iron Deficiency Anemia."""
        fake_probs = torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.90]])
        with patch.object(model_service.AnemiaVGGModel, 'forward', return_value=fake_probs):
            res = model_service.predict_anemia_sample(self.test_img)
            self.assertEqual(res['internal_class'], 'Vitamin B12 Deficiency (Megaloblastic)')
            self.assertEqual(res['predicted_class'], 'Iron Deficiency Anemia')
            self.assertAlmostEqual(res['confidence'], 90.0, places=1)
            self.assertEqual(res['all_probabilities'][0]['name'], 'Iron Deficiency Anemia')
            self.assertAlmostEqual(res['all_probabilities'][0]['probability'], 90.0, places=1)

    def test_swap_4_iron_deficiency_to_vitamin_b12(self):
        """If model internally predicts Iron Deficiency Anemia (index 1), display Vitamin B12 Deficiency."""
        fake_probs = torch.tensor([[0.02, 0.88, 0.03, 0.03, 0.04]])
        with patch.object(model_service.AnemiaVGGModel, 'forward', return_value=fake_probs):
            res = model_service.predict_anemia_sample(self.test_img)
            self.assertEqual(res['internal_class'], 'Iron Deficiency Anemia')
            self.assertEqual(res['predicted_class'], 'Vitamin B12 Deficiency')
            self.assertAlmostEqual(res['confidence'], 88.0, places=1)
            self.assertEqual(res['all_probabilities'][0]['name'], 'Vitamin B12 Deficiency')
            self.assertAlmostEqual(res['all_probabilities'][0]['probability'], 88.0, places=1)

    def test_sickle_cell_unchanged(self):
        """Sickle Cell Anemia (index 2) must remain completely unchanged."""
        fake_probs = torch.tensor([[0.02, 0.03, 0.89, 0.03, 0.03]])
        with patch.object(model_service.AnemiaVGGModel, 'forward', return_value=fake_probs):
            res = model_service.predict_anemia_sample(self.test_img)
            self.assertEqual(res['internal_class'], 'Sickle Cell Anemia')
            self.assertEqual(res['predicted_class'], 'Sickle Cell Anemia')
            self.assertAlmostEqual(res['confidence'], 89.0, places=1)
            self.assertEqual(res['all_probabilities'][0]['name'], 'Sickle Cell Anemia')

    def test_api_json_and_html_rendering(self):
        """Verify Flask /predict API returns swapped class names in JSON & HTML."""
        fake_probs = torch.tensor([[0.02, 0.85, 0.03, 0.05, 0.05]])  # Internal Iron Deficiency
        with patch.object(model_service.AnemiaVGGModel, 'forward', return_value=fake_probs):
            # JSON request
            with open(self.test_img, 'rb') as f:
                resp = self.client.post('/predict', data={'image': (f, 'test.jpg')}, headers={'X-Requested-With': 'XMLHttpRequest'})
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data['anemiaType'], 'Vitamin B12 Deficiency')
            self.assertEqual(data['allProbabilities'][0]['name'], 'Vitamin B12 Deficiency')


if __name__ == '__main__':
    unittest.main()
