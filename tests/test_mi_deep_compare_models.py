from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mi_low_shot_deep_compare import build_model  # noqa: E402


class DeepComparisonModelTests(unittest.TestCase):
    def test_model_output_and_embedding_shapes(self) -> None:
        cases = {
            "eegnet": torch.randn(3, 1, 21, 480),
            "fbcnet": torch.randn(3, 5, 21, 480),
            "eegconformer": torch.randn(3, 1, 21, 480),
        }
        for kind, inputs in cases.items():
            with self.subTest(kind=kind):
                model = build_model(kind, channels=21, samples=480, fs=160)
                logits = model(inputs)
                embedding = model.embed(inputs)
                self.assertEqual(tuple(logits.shape), (3, 2))
                self.assertEqual(embedding.shape[0], 3)
                self.assertEqual(embedding.ndim, 2)


if __name__ == "__main__":
    unittest.main()
