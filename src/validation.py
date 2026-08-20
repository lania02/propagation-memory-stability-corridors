"""Numerical constructions for the propagation--memory stability bounds.

All experiments use the induced infinity norm.  The routines deliberately
include unequal block dimensions, rectangular couplings, and nonnormal memory
operators so that the validation exercises the general theorem rather than its
homogeneous square specialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """The family bounds for one parameter quadruple."""

    upper: float
    separated: bool
    gap: float
    lower: float | None = None
    symmetric_upper: float | None = None
    inner_radius: float | None = None
    outer_radius: float | None = None


def inf_norm(matrix: np.ndarray) -> float:
    """Return the induced infinity norm."""

    return float(np.linalg.norm(matrix, ord=np.inf))


def scale_to_inf_norm(matrix: np.ndarray, target: float) -> np.ndarray:
    """Scale a nonzero matrix to a requested induced infinity norm."""

    if target < 0:
        raise ValueError("target norm must be nonnegative")
    if target == 0:
        return np.zeros_like(matrix, dtype=float)
    current = inf_norm(matrix)
    if current == 0:
        raise ValueError("cannot scale the zero matrix to a positive norm")
    return np.asarray(matrix, dtype=float) * (target / current)


def theoretical_bounds(M: float, c: float, eps: float, q: float) -> Bounds:
    """Compute the unconditional upper bound and, when available, the band."""

    if min(M, c, eps, q) < 0:
        raise ValueError("M, c, eps, and q must be nonnegative")

    a = c + eps - M
    upper = c + eps + 0.5 * (np.sqrt(a * a + 4.0 * q) - a)
    gap = c - M - eps
    separated = gap > 2.0 * np.sqrt(q)
    if not separated:
        return Bounds(upper=float(upper), separated=False, gap=float(gap))

    root = np.sqrt(gap * gap - 4.0 * q)
    r_minus = 0.5 * (gap - root)
    r_plus = 0.5 * (gap + root)
    R_minus = eps + r_minus
    R_plus = eps + r_plus
    return Bounds(
        upper=float(upper),
        separated=True,
        gap=float(gap),
        lower=float(c - R_minus),
        symmetric_upper=float(c + R_minus),
        inner_radius=float(R_minus),
        outer_radius=float(R_plus),
    )


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius of a square matrix."""

    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def nonnormality(matrix: np.ndarray) -> float:
    """Dimensionless Frobenius commutator measure."""

    denominator = np.linalg.norm(matrix, ord="fro") ** 2
    if denominator == 0:
        return 0.0
    commutator = matrix.T.conj() @ matrix - matrix @ matrix.T.conj()
    return float(np.linalg.norm(commutator, ord="fro") / denominator)


def assemble(P: np.ndarray, B: np.ndarray, D: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Assemble the two-block Jacobian."""

    return np.block([[P, B], [D, C]])


def random_row_stochastic(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a dense directed row-stochastic matrix."""

    raw = rng.gamma(shape=1.3, scale=1.0, size=(n, n))
    np.fill_diagonal(raw, 0.0)
    return raw / raw.sum(axis=1, keepdims=True)


def permutation_shift(n: int, shift: int) -> np.ndarray:
    """Return the row-stochastic cyclic permutation with a nonzero shift."""

    if n < 2:
        raise ValueError("n must be at least two")
    if shift % n == 0:
        raise ValueError("shift must be nonzero modulo n")
    return np.roll(np.eye(n), shift % n, axis=1)


def topology_family(n: int, count: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Generate structured and random row-stochastic directed topologies."""

    topologies: list[tuple[str, np.ndarray]] = []
    for shift in (1, 2, 3):
        topologies.append((f"shift_{shift}", permutation_shift(n, shift)))

    sink = np.zeros((n, n))
    sink[:, 0] = 1.0
    topologies.append(("rank_one_sink", sink))

    ring = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), 2, axis=1)
    topologies.append(("two_neighbor_ring", ring))

    while len(topologies) < count:
        random_part = random_row_stochastic(n, rng)
        if len(topologies) % 3 == 0:
            shift = np.roll(np.eye(n), 1 + len(topologies) % (n - 1), axis=1)
            matrix = 0.9 * shift + 0.1 * random_part
            label = "near_permutation"
        else:
            matrix = random_part
            label = "random_directed"
        topologies.append((f"{label}_{len(topologies)}", matrix))
    return topologies


def _random_nonnormal_square(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return a nonzero, generally nonnormal square matrix."""

    upper = np.triu(rng.normal(size=(dim, dim)))
    dense = 0.15 * rng.normal(size=(dim, dim))
    return upper + dense


def general_system(
    n: int,
    p: int,
    M: float,
    c: float,
    eps: float,
    q: float,
    rng: np.random.Generator,
    coupling_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct one general admissible member with rectangular couplings."""

    if coupling_ratio <= 0:
        raise ValueError("coupling_ratio must be positive")

    P = scale_to_inf_norm(_random_nonnormal_square(n, rng), M)
    E = scale_to_inf_norm(_random_nonnormal_square(p, rng), eps)
    C = c * np.eye(p) + E

    base = np.sqrt(q)
    B = scale_to_inf_norm(rng.normal(size=(n, p)), base / coupling_ratio)
    D = scale_to_inf_norm(rng.normal(size=(p, n)), base * coupling_ratio)
    J = assemble(P, B, D, C)
    return J, P, B, D, C


def worked_network_system(
    W: np.ndarray,
    c: float,
    M: float,
    eps: float,
    q: float,
    B: np.ndarray,
    D: np.ndarray,
    memory_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the limited-channel network example for one topology."""

    n = W.shape[0]
    susceptibilities = np.linspace(0.55 * M, M, n)
    P = np.diag(susceptibilities) @ W
    C = c * np.eye(memory_shape.shape[0]) + eps * memory_shape
    J = assemble(P, B, D, C)
    if inf_norm(P) > M + 1e-12:
        raise AssertionError("constructed propagation block violates M")
    if inf_norm(C - c * np.eye(C.shape[0])) > eps + 1e-12:
        raise AssertionError("constructed memory block violates eps")
    if inf_norm(B) * inf_norm(D) > q + 1e-12:
        raise AssertionError("constructed feedback loop violates q")
    return J, P, C


def fixed_worked_components(
    n: int,
    p: int,
    eps: float,
    q: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed rectangular couplings and a nonnormal memory shape."""

    memory_shape = np.diag(np.linspace(-1.0, 1.0, p))
    memory_shape += 0.65 * np.diag(np.ones(p - 1), k=1)
    if p > 2:
        memory_shape += 0.20 * np.diag(np.ones(p - 2), k=2)
    memory_shape = scale_to_inf_norm(memory_shape, 1.0)

    # An asymmetric factorization makes the ordinary full-matrix norm loose,
    # while the closed-loop product remains q.
    B = scale_to_inf_norm(rng.normal(size=(n, p)), 0.25 * np.sqrt(q))
    D = scale_to_inf_norm(rng.normal(size=(p, n)), 4.0 * np.sqrt(q))
    return B, D, memory_shape


def cluster_diagnostics(eigenvalues: np.ndarray, c: float, bounds: Bounds) -> dict[str, float | int]:
    """Summarize the separated spectral clusters."""

    if not bounds.separated:
        raise ValueError("cluster diagnostics require separated bounds")
    distances = np.abs(eigenvalues - c)
    inner = distances <= bounds.inner_radius + 1e-10
    outer = distances >= bounds.outer_radius - 1e-10
    annulus = ~(inner | outer)
    return {
        "inner_count": int(np.count_nonzero(inner)),
        "outer_count": int(np.count_nonzero(outer)),
        "annulus_count": int(np.count_nonzero(annulus)),
        "inner_max_modulus": float(np.max(np.abs(eigenvalues[inner]))),
        "outer_max_modulus": float(np.max(np.abs(eigenvalues[outer]))),
    }
