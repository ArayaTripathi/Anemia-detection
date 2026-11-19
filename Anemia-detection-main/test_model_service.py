"""
Comprehensive test suite verifying local model loading and prediction
with M_Anemia.h5 and M_Anemia_finetuned.h5.
"""

import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_service
from app import app


class TestModelService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.test_img = "uploads/test_unit_sample.jpg"
        img = Image.new("RGB", (224, 224), color=(200, 180, 190))
        img.save(cls.test_img)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img):
            os.remove(cls.test_img)

    def test_load_m_anemia_h5(self):
        """Verify M_Anemia.h5 loads properly without errors."""
        model = model_service.load_anemia_model('M_Anemia.h5')
        self.assertIsNotNone(model)
        self.assertIsInstance(model, model_service.AnemiaVGGModel)

    def test_load_m_anemia_finetuned_h5(self):
        """Verify M_Anemia_finetuned.h5 loads properly without errors."""
        model = model_service.load_anemia_model('M_Anemia_finetuned.h5')
        self.assertIsNotNone(model)
        self.assertIsInstance(model, model_service.AnemiaVGGModel)

    def test_predict_sample_with_m_anemia(self):
        """Verify inference runs cleanly with M_Anemia.h5."""
        res = model_service.predict_anemia_sample(
            self.test_img,
            hemoglobin="9.0",
            mcv="70.0",
            rbc_count="3.5",
            mch="23.0",
            h5_path='M_Anemia.h5'
        )
        self.assertIn('predicted_class', res)
        self.assertIn(res['predicted_class'], model_service.DISPLAY_CLASSES)
        self.assertIn('confidence', res)
        self.assertGreater(res['confidence'], 0.0)
        self.assertEqual(len(res['all_probabilities']), 5)
        self.assertIn('cbc_corroboration', res)

    def test_bidirectional_class_name_swaps(self):
        """Verify the exact bidirectional class name swap requirements:
        1. Dimorphic Anemia <-> Thalassemia
        2. Vitamin B12 Deficiency <-> Iron Deficiency Anemia
        3. Sickle Cell Anemia remains unchanged
        """
        mapping = model_service.SWAP_CLASS_MAP
        # 1. Dimorphic Anemia -> Thalassemia
        self.assertEqual(mapping['Dimorphic Anemia'], 'Thalassemia')
        # 2. Thalassemia -> Dimorphic Anemia
        self.assertEqual(mapping['Thalassemia'], 'Dimorphic Anemia')
        # 3. Vitamin B12 Deficiency -> Iron Deficiency Anemia
        self.assertEqual(mapping['Vitamin B12 Deficiency'], 'Iron Deficiency Anemia')
        self.assertEqual(mapping['Vitamin B12 Deficiency (Megaloblastic)'], 'Iron Deficiency Anemia')
        # 4. Iron Deficiency Anemia -> Vitamin B12 Deficiency
        self.assertEqual(mapping['Iron Deficiency Anemia'], 'Vitamin B12 Deficiency')
        # 5. Sickle Cell Anemia remains unchanged
        self.assertEqual(mapping['Sickle Cell Anemia'], 'Sickle Cell Anemia')

    def test_predict_endpoint_flask(self):
        """Verify POST /predict endpoint functions seamlessly."""
        with open(self.test_img, 'rb') as f:
            resp = self.client.post('/predict', data={
                'name': 'Integration Test Patient',
                'age': '32',
                'gender': 'Male',
                'blood_type': 'B+',
                'hemoglobin': '10.2',
                'mcv': '75.0',
                'rbc_count': '4.1',
                'mch': '25.0',
                'image': (f, 'test.jpg')
            })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Clinical Recommendations', resp.data)


if __name__ == '__main__':
    unittest.main()
