from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mi_group_graph_csp import (  # noqa: E402
    bilateral_group_projection,
    motor_graph_laplacian,
    reflection_matrix,
)
from mi_physionet_benchmark import MOTOR_CHANNELS  # noqa: E402


class GroupGraphCspTests(unittest.TestCase):
    def test_reflection_is_involutive_orthogonal_permutation(self) -> None:
        reflection = reflection_matrix(MOTOR_CHANNELS)
        identity = np.eye(len(MOTOR_CHANNELS))
        np.testing.assert_allclose(reflection @ reflection, identity)
        np.testing.assert_allclose(reflection.T @ reflection, identity)

    def test_graph_laplacian_is_symmetric_positive_semidefinite(self) -> None:
        laplacian = motor_graph_laplacian(MOTOR_CHANNELS)
        np.testing.assert_allclose(laplacian, laplacian.T)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(laplacian))), -1e-12)

    def test_full_group_projection_swaps_classes_under_reflection(self) -> None:
        rng = np.random.default_rng(7)
        reflection = reflection_matrix(MOTOR_CHANNELS)
        raw = rng.normal(size=(2, len(MOTOR_CHANNELS), len(MOTOR_CHANNELS)))
        covariances = np.stack([matrix @ matrix.T for matrix in raw])
        projected = bilateral_group_projection(covariances, reflection, eta=1.0)
        np.testing.assert_allclose(projected[1], reflection @ projected[0] @ reflection.T)


if __name__ == "__main__":
    unittest.main()
