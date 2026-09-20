"""Regression tests for the analytical bounds and numerical constructions."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from validation import (  # noqa: E402
    assemble,
    cluster_diagnostics,
    general_system,
    inf_norm,
    interpolated_witness_system,
    perturbation_norms,
    permutation_shift,
    spectral_radius,
    theoretical_bounds,
    weighted_block_envelope,
)


class StabilityBoundTests(unittest.TestCase):
    def test_weighted_block_optimization_matches_family_upper(self) -> None:
        for M, c, eps, q in ((0.45, 0.84, 0.04, 0.0064), (0.8, 0.4, 0.1, 0.09)):
            for scale in (0.01, 1.0, 100.0):
                b, d = np.sqrt(q) / scale, np.sqrt(q) * scale
                value = weighted_block_envelope(M, c + eps, b, d)
                self.assertAlmostEqual(value, theoretical_bounds(M, c, eps, q).upper)
                t = (value - M) / b
                self.assertAlmostEqual(M + b * t, c + eps + d / t)
                self.assertAlmostEqual(max(M + b * t, c + eps + d / t), value)
                for factor in (0.3, 3.0):
                    self.assertGreaterEqual(max(M + b * t * factor,
                                                c + eps + d / (t * factor)), value)

    def test_weighted_block_zero_and_unsaturated_product(self) -> None:
        for b, d in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)):
            self.assertAlmostEqual(weighted_block_envelope(0.4, 0.8, b, d), 0.8)
        self.assertLess(weighted_block_envelope(0.45, 0.88, 0.04, 0.04),
                        theoretical_bounds(0.45, 0.84, 0.04, 0.0064).upper)

    def test_interpolation_preserves_family_and_cluster_count(self) -> None:
        bounds = theoretical_bounds(0.65, 0.982, 0.0, 0.01)
        J_zero, _ = interpolated_witness_system(0.0)
        for theta in np.linspace(0, 1, 41):
            J, W = interpolated_witness_system(float(theta))
            self.assertTrue(np.all(W >= 0))
            np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-14)
            np.testing.assert_allclose(np.diag(W), 0.0)
            self.assertAlmostEqual(inf_norm(J[:16, :16]), 0.65)
            np.testing.assert_array_equal(J[:16, 16:], J_zero[:16, 16:])
            np.testing.assert_array_equal(J[16:, :], J_zero[16:, :])
            diag = cluster_diagnostics(np.linalg.eigvals(J), 0.982, bounds)
            self.assertEqual(diag["inner_count"], 16)
            self.assertEqual(diag["outer_count"], 16)
            self.assertEqual(diag["annulus_count"], 0)
            self.assertLessEqual(spectral_radius(J), bounds.upper)
            self.assertGreaterEqual(spectral_radius(J), bounds.lower)
        self.assertLess(spectral_radius(interpolated_witness_system(0.96)[0]), 1.0)
        self.assertGreater(spectral_radius(interpolated_witness_system(0.98)[0]), 1.0)

    def test_fixed_network_trajectory_matches_matrix_powers(self) -> None:
        initial = np.ones(32) / np.sqrt(32.0)
        endpoints = []
        for theta in (0.0, 1.0):
            J, _ = interpolated_witness_system(theta)
            norms = perturbation_norms(J, initial, 4000)
            self.assertAlmostEqual(norms[0], 1.0)
            for step in (1, 20, 100, 4000):
                expected = np.linalg.norm(np.linalg.matrix_power(J, step) @ initial)
                np.testing.assert_allclose(norms[step], expected, rtol=1e-10)
            endpoints.append(norms[-1])
        self.assertLess(endpoints[0], 1e-8)
        self.assertGreater(endpoints[1], 1.0)

    def test_uniform_corridors_along_interpolation(self) -> None:
        recovery_edge = 1.0 - 0.01 / (1.0 - 0.65)
        instability_edge = 1.0 + 0.01 / (1.0 - 0.65)
        self.assertAlmostEqual(theoretical_bounds(0.65, recovery_edge, 0.0, 0.01).upper, 1.0)
        self.assertAlmostEqual(theoretical_bounds(0.65, instability_edge, 0.0, 0.01).lower, 1.0)
        for c in (0.96, 1.04):
            bounds = theoretical_bounds(0.65, c, 0.0, 0.01)
            if c < 1:
                self.assertLess(bounds.upper, 1.0)
            else:
                self.assertGreater(bounds.lower, 1.0)
            for theta in np.linspace(0, 1, 11):
                rho = spectral_radius(interpolated_witness_system(float(theta), c)[0])
                self.assertGreaterEqual(rho, bounds.lower)
                self.assertLessEqual(rho, bounds.upper)

    def test_two_state_comparison_matrix_attains_upper_edge(self) -> None:
        M, c, eps, q = 0.45, 0.84, 0.04, 0.0064
        bounds = theoretical_bounds(M, c, eps, q)
        comparison = np.array(
            [[M, np.sqrt(q)], [np.sqrt(q), c + eps]]
        )
        self.assertAlmostEqual(spectral_radius(comparison), bounds.upper, places=14)

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

    def test_topology_only_stability_witness(self) -> None:
        n = 16
        M, c, eps = 0.65, 0.982, 0.0
        alpha, b_max = 0.1, 0.1
        q = alpha * b_max
        bounds = theoretical_bounds(M, c, eps, q)
        self.assertTrue(bounds.separated)
        self.assertLess(bounds.lower, 1.0)
        self.assertGreater(bounds.upper, 1.0)

        A = np.diag(np.linspace(0.35, M, n))
        B = np.diag(np.linspace(0.024, b_max, n))
        C = c * np.eye(n)
        D = alpha * np.eye(n)
        radii = {}
        for shift in (9, 8):
            W = permutation_shift(n, shift)
            radii[shift] = spectral_radius(assemble(A @ W, B, D, C))

        self.assertAlmostEqual(radii[9], 0.9951593871107954, places=12)
        self.assertAlmostEqual(radii[8], 1.0007141342930006, places=12)
        self.assertLess(radii[9], 1.0)
        self.assertGreater(radii[8], 1.0)


if __name__ == "__main__":
    unittest.main()
