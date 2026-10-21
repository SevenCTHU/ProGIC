"""Regression tests using the actual ProGIC checkpoints (CPU, small inputs)."""
import os
from pathlib import Path
import unittest

import torch

from ProGIC import ProGIC
from test import pack_indices, unpack_indices, pad_image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path(os.environ.get("PROGIC_CHECKPOINT_DIR", ROOT / "ckpt"))


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_checkpoints_rates_and_independent_decoder(self):
        for variant, filename in (("base", "ckpt.pth.tar"), ("small", "ckpt_s.pth.tar")):
            with self.subTest(variant=variant):
                state = torch.load(CHECKPOINTS / filename, map_location="cpu", weights_only=True)["state_dict"]
                encoder = ProGIC(variant).eval()
                decoder = ProGIC(variant).eval()
                encoder.load_state_dict(state, strict=True)
                decoder.load_state_dict(state, strict=True)
                image = torch.rand(1, 3, 31, 47) * 2 - 1
                padded = pad_image(image)
                full = encoder.compress(padded, 4)
                self.assertEqual(len(full), 5)
                for index in range(5):
                    direct = encoder.compress(padded, index)
                    self.assertEqual(len(direct), index + 1)
                    for expected, actual in zip(full[:index + 1], direct):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    payload, shape = pack_indices(direct)
                    restored = unpack_indices(payload, shape, torch.device("cpu"))
                    for expected, actual in zip(direct, restored):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(len(payload), (sum(t.numel() for t in direct) * 10 + 7) // 8)
                    result = decoder.decompress(restored)
                    self.assertEqual(result.shape, padded.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result, encoder.decompress(full, index))
                    self.assertEqual(result[..., :31, :47].shape, image.shape)
                # Encode another shape before decoding the first image again.
                encoder.compress(torch.zeros(1, 3, 16, 16), 0)
                torch.testing.assert_close(encoder.decompress(full), decoder.decompress(full))

    def test_bit_packing_extremes_and_byte_padding(self):
        indices = [torch.tensor([[[0, 1, 511, 1023, 17]]])]
        payload, shape = pack_indices(indices)
        self.assertEqual(len(payload), 7)
        torch.testing.assert_close(unpack_indices(payload, shape, torch.device("cpu"))[0], indices[0])


if __name__ == "__main__":
    unittest.main()
