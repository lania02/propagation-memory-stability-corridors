"""Regression tests for the analytical bounds and numerical constructions."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from validation import (  # noqa: E402
    cluster_diagnostics,
    general_system,
    spectral_radius,
    theoretical_bounds,
)


class StabilityBoundTests(unittest.TestCase):
    def test_q_zero_is_block_triangular_boundary(self) -> None:
        M, c, eps, q = 0.2, 1.0, 0.1, 0.0
        bounds = theoretical_bounds(M, c, eps, q)
        self.assertAlmostEqual(bounds.upper, max(M, c + eps))
        self.assertTrue(bounds.separated)
        self.assertAlmostEqual(bounds.inner_radius, eps)

    def test_general_rectangular_ensemble(self) -> None:
        rng = np.random.default_rng(1701)
        M, c, eps, q = 0.45, 0.84, 0.04, 0.0064
        bounds = theoretical_bounds(M, c, eps, q)
        self.assertTrue(bounds.separated)
        for _ in range(300):
            n = int(rng.integers(4, 15))
            p = int(rng.integers(2, 10))
            if p == n:
                p = 2 if n != 2 else 3
            ratio = float(10.0 ** rng.uniform(-1.0, 1.0))
            J, _, _, _, _ = general_system(n, p, M, c, eps, q, rng, ratio)
            eigenvalues = np.linalg.eigvals(J)
            rho = spectral_radius(J)
            diagnostics = cluster_diagnostics(eigenvalues, c, bounds)
            self.assertEqual(diagnostics["inner_count"], p)
            self.assertEqual(diagnostics["outer_count"], n)
            self.assertEqual(diagnostics["annulus_count"], 0)
            self.assertGreaterEqual(rho + 1e-10, bounds.lower)
            self.assertLessEqual(rho, bounds.upper + 1e-10)

    def test_unconditional_upper_without_separation(self) -> None:
        rng = np.random.default_rng(1905)
        M, c, eps, q = 0.72, 0.68, 0.08, 0.09
        bounds = theoretical_bounds(M, c, eps, q)
        self.assertFalse(bounds.separated)
        for _ in range(200):
            J, _, _, _, _ = general_system(8, 3, M, c, eps, q, rng)
            self.assertLessEqual(spectral_radius(J), bounds.upper + 1e-10)


if __name__ == "__main__":
    unittest.main()
