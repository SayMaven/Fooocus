import unittest
import numpy as np
from PIL import Image
from modules.util import to_numpy_image, extract_inpaint_image_and_mask, HWC3


class TestGradioAdapter(unittest.TestCase):
    def test_to_numpy_image_none(self):
        self.assertIsNone(to_numpy_image(None))

    def test_to_numpy_image_ndarray(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        res = to_numpy_image(arr)
        self.assertIs(res, arr)

    def test_to_numpy_image_pil_rgb(self):
        pil_img = Image.new("RGB", (32, 32), color=(255, 128, 0))
        res = to_numpy_image(pil_img)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (32, 32, 3))
        self.assertEqual(res[0, 0, 0], 255)
        self.assertEqual(res[0, 0, 1], 128)
        self.assertEqual(res[0, 0, 2], 0)

    def test_to_numpy_image_pil_rgba(self):
        pil_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
        res = to_numpy_image(pil_img)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (32, 32, 3))  # to_numpy_image converts to RGB

    def test_hwc3_with_pil(self):
        pil_img = Image.new("RGB", (40, 40), color=(10, 20, 30))
        res = HWC3(pil_img)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (40, 40, 3))
        self.assertEqual(res.dtype, np.uint8)

    def test_extract_inpaint_gradio3_sketch(self):
        bg = np.ones((80, 80, 3), dtype=np.uint8) * 100
        mask_raw = np.zeros((80, 80, 4), dtype=np.uint8)
        mask_raw[10:30, 10:30, 0] = 255  # Legacy Gradio 3 sketch put mask in channel 0

        data = {'image': bg, 'mask': mask_raw}
        img, mask = extract_inpaint_image_and_mask(data)

        self.assertIsNotNone(img)
        self.assertIsNotNone(mask)
        self.assertEqual(img.shape, (80, 80, 3))
        self.assertEqual(mask.shape, (80, 80))
        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_extract_inpaint_gradio5_image_editor_rgba_layer(self):
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 50
        # Layer is RGBA, brush stroke is on alpha channel
        layer = np.zeros((100, 100, 4), dtype=np.uint8)
        layer[20:40, 20:40, 0:3] = 255
        layer[20:40, 20:40, 3] = 255  # alpha = 255

        data = {
            'background': bg,
            'layers': [layer],
            'composite': bg
        }
        img, mask = extract_inpaint_image_and_mask(data)

        self.assertIsNotNone(img)
        self.assertIsNotNone(mask)
        self.assertEqual(img.shape, (100, 100, 3))
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask[30, 30], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_extract_inpaint_gradio5_multilayer(self):
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 50
        layer1 = np.zeros((100, 100, 4), dtype=np.uint8)
        layer1[10:20, 10:20, 3] = 255

        layer2 = np.zeros((100, 100, 4), dtype=np.uint8)
        layer2[50:60, 50:60, 3] = 255

        data = {
            'background': bg,
            'layers': [layer1, layer2],
            'composite': bg
        }
        img, mask = extract_inpaint_image_and_mask(data)

        self.assertEqual(mask[15, 15], 255)
        self.assertEqual(mask[55, 55], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_extract_inpaint_gradio5_empty_layers(self):
        bg = np.ones((64, 64, 3), dtype=np.uint8) * 200
        data = {
            'background': bg,
            'layers': [],
            'composite': bg
        }
        # Inpaint mode: empty layers -> all zeros mask
        img, mask = extract_inpaint_image_and_mask(data, is_mask_upload=False)
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(np.all(mask == 0))

        # Mask upload mode: empty layers -> binarized background
        img, mask = extract_inpaint_image_and_mask(data, is_mask_upload=True)
        self.assertTrue(np.all(mask == 255))

    def test_extract_inpaint_tuple(self):
        bg = np.ones((50, 50, 3), dtype=np.uint8) * 100
        mask_raw = np.zeros((50, 50), dtype=np.uint8)
        mask_raw[10:20, 10:20] = 255

        img, mask = extract_inpaint_image_and_mask((bg, mask_raw))
        self.assertEqual(img.shape, (50, 50, 3))
        self.assertEqual(mask.shape, (50, 50))
        self.assertEqual(mask[15, 15], 255)

    def test_extract_inpaint_layer_resize_mismatch(self):
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 50
        # Mismatched layer size: 50x50 vs 100x100 background
        layer = np.zeros((50, 50, 4), dtype=np.uint8)
        layer[:, :, 3] = 255

        data = {
            'background': bg,
            'layers': [layer],
            'composite': bg
        }
        img, mask = extract_inpaint_image_and_mask(data)
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(np.any(mask == 255))

    def test_gradio_hijack_factory_mapping(self):
        from unittest.mock import MagicMock
        from modules import gradio_hijack

        if not gradio_hijack.IS_LEGACY_GRADIO_3:
            mock_gr = MagicMock()
            mock_gr.Image = MagicMock(return_value="mock_image")
            mock_gr.ImageEditor = MagicMock(return_value="mock_image_editor")
            mock_gr.Brush = MagicMock(side_effect=lambda **kw: kw)
            mock_gr.Eraser = MagicMock(return_value="mock_eraser")

            orig_gr = gradio_hijack.gr
            gradio_hijack.gr = mock_gr
            try:
                # Normal image upload: forwards to gr.Image with sources=['upload']
                res1 = gradio_hijack.Image(label='Upload', source='upload', type='numpy')
                self.assertEqual(res1, "mock_image")
                mock_gr.Image.assert_called_with(label='Upload', type='numpy', sources=['upload'])

                # Sketch image upload: forwards to gr.ImageEditor with brush and eraser
                res2 = gradio_hijack.Image(label='Inpaint', source='upload', type='numpy', tool='sketch', brush_color='#FFFFFF')
                self.assertEqual(res2, "mock_image_editor")
                mock_gr.ImageEditor.assert_called()
                call_kwargs = mock_gr.ImageEditor.call_args[1]
                self.assertIn('brush', call_kwargs)
                self.assertEqual(call_kwargs['sources'], ['upload'])
            finally:
                gradio_hijack.gr = orig_gr

    def test_legacy_event_listener_js_bridge(self):
        class MockEventListenerMethod:
            def __call__(self, *args, **kwargs):
                return kwargs

        method = MockEventListenerMethod()
        orig_call = method.__call__
        def safe_elm_call(self, *args, **kwargs):
            if 'js' in kwargs and '_js' not in kwargs:
                kwargs['_js'] = kwargs.pop('js')
            return orig_call(*args, **kwargs)

        res = safe_elm_call(method, inputs=None, js='cancelGenerateForever')
        self.assertNotIn('js', res)
        self.assertIn('_js', res)
        self.assertEqual(res['_js'], 'cancelGenerateForever')


if __name__ == '__main__':
    unittest.main()
