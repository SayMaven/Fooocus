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

    def test_defaults_configured_for_improve_detail_and_animeface(self):
        self.assertEqual(modules.config.default_inpaint_method, modules.flags.inpaint_option_detail)
        self.assertEqual(modules.config.default_enhance_inpaint_mask_model, 'yolov8n-animeface.onnx')
        self.assertEqual(modules.config.default_inpaint_mask_model, 'yolov8n-animeface.onnx')

    def test_prompt_lora_persistence_for_enhance(self):
        from unittest.mock import MagicMock
        if 'cv2' not in sys.modules:
            sys.modules['cv2'] = MagicMock()
        from modules.util import parse_lora_references_from_prompt
        base_prompt = "masterpiece, 1girl, <lora:8in1BangDreamYumeMita_ANIMA:1.0>, looking at viewer"
        task_loras = []
        parsed_loras, _ = parse_lora_references_from_prompt(
            base_prompt, task_loras, skip_file_check=True
        )
        task_loras = list(parsed_loras)
        self.assertEqual(len(task_loras), 1)
        self.assertEqual(task_loras[0][0], '8in1BangDreamYumeMita_ANIMA.safetensors')
        self.assertEqual(task_loras[0][1], 1.0)

        # User fills Enhancement positive prompt with custom text (e.g. 'beautiful eyes')
        enhance_prompt = "beautiful eyes"
        enhance_parsed_loras, _ = parse_lora_references_from_prompt(
            enhance_prompt, task_loras, skip_file_check=True
        )
        self.assertEqual(len(enhance_parsed_loras), 1)
        self.assertEqual(enhance_parsed_loras[0][0], '8in1BangDreamYumeMita_ANIMA.safetensors')


if __name__ == '__main__':
    unittest.main()
