from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mi_group_filterbank_csp import fisher_scores  # noqa: E402


class GroupFilterBankCspTests(unittest.TestCase):
    def test_fisher_score_ranks_separated_feature_first(self) -> None:
        rng = np.random.default_rng(4)
        labels = np.repeat([0, 1], 20)
        features = rng.normal(size=(40, 3))
        features[labels == 1, 1] += 4.0
        scores = fisher_scores(features, labels)
        self.assertEqual(int(np.argmax(scores)), 1)

    def test_fisher_scores_are_scale_invariant(self) -> None:
        rng = np.random.default_rng(9)
        labels = np.repeat([0, 1], 15)
        features = rng.normal(size=(30, 4))
        features[labels == 1] += np.asarray([0.2, 0.5, 1.0, 2.0])
        scaled = features * np.asarray([0.1, 2.0, 10.0, 50.0])
        np.testing.assert_allclose(fisher_scores(features, labels), fisher_scores(scaled, labels))


if __name__ == "__main__":
    unittest.main()
