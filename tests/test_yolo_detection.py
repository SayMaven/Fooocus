import os
import sys
import unittest

# Sanitize sys.argv for args_manager before importing modules
_saved_argv = list(sys.argv)
sys.argv = [sys.argv[0]]

try:
    import modules.config
    import modules.flags
finally:
    sys.argv = _saved_argv


class TestYoloDetectionConfig(unittest.TestCase):
    def test_yolo_models_catalog_complete_and_excludes_face_yolov8s(self):
        catalog = modules.config.yolo_detection_models
        expected_models = [
            'yolov8n-animeface.onnx',
            'yolov8s-animeface.onnx',
            'face_yolov8n.onnx',
            'yolov8n-eyes.onnx',
            'hand_yolov8n.onnx',
            'person_yolov8n.onnx',
        ]
        for m in expected_models:
            self.assertIn(m, catalog, f"Expected {m} in yolo_detection_models catalog")
            self.assertTrue(catalog[m].startswith("https://huggingface.co/"), f"URL for {m} is invalid")

        # Must exclude face_yolov8s per explicit user instruction
        self.assertNotIn('face_yolov8s.onnx', catalog)
        self.assertNotIn('face_yolov8s.pt', catalog)

    def test_flags_inpaint_mask_models_includes_yolo(self):
        flags_models = modules.flags.inpaint_mask_models
        for m in modules.flags.yolo_detection_models:
            self.assertIn(m, flags_models)
        self.assertNotIn('face_yolov8s.onnx', flags_models)

    def test_get_all_mask_models(self):
        models = modules.config.get_all_mask_models()
        self.assertIn('yolov8n-animeface.onnx', models)
        self.assertIn('sam', models)
        self.assertNotIn('face_yolov8s.onnx', models)


if __name__ == '__main__':
    unittest.main()
