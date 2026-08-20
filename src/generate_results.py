"""Generate all data and publication figures for the APS Open Science paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import numpy as np

from validation import (
    assemble,
    cluster_diagnostics,
    fixed_worked_components,
    general_system,
    inf_norm,
    nonnormality,
    permutation_shift,
    spectral_radius,
    theoretical_bounds,
    topology_family,
    worked_network_system,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
SEED = 26032026

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with a stable column order."""

    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    """Use a compact APS-compatible visual style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "lines.linewidth": 1.5,
        }
    )


def generate_validation_data() -> dict[str, object]:
    """Create deterministic theorem-validation datasets."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    M, c, eps, q = 0.45, 0.84, 0.04, 0.0064
    bounds = theoretical_bounds(M, c, eps, q)
    if not bounds.separated or bounds.upper >= 1.0:
        raise AssertionError("worked example should lie in universal recovery")

    n_example, p_example = 12, 4
    B, D, memory_shape = fixed_worked_components(n_example, p_example, eps, q, rng)
    example_topology = topology_family(n_example, 12, rng)[7][1]
    J_example, P_example, C_example = worked_network_system(
        example_topology, c, M, eps, q, B, D, memory_shape
    )
    example_eigs = np.linalg.eigvals(J_example)
    diagnostics = cluster_diagnostics(example_eigs, c, bounds)
    if diagnostics["inner_count"] != p_example or diagnostics["annulus_count"] != 0:
        raise AssertionError("worked example violates cluster count")

    spectrum_rows: list[dict[str, object]] = []
    for index, value in enumerate(example_eigs):
        distance = abs(value - c)
        cluster = "inner" if distance <= bounds.inner_radius + 1e-10 else "outer"
        spectrum_rows.append(
            {
                "index": index,
                "real": float(value.real),
                "imag": float(value.imag),
                "modulus": float(abs(value)),
                "distance_from_c": float(distance),
                "cluster": cluster,
            }
        )
    write_csv(DATA_DIR / "spectrum_example.csv", spectrum_rows)

    ensemble_rows: list[dict[str, object]] = []
    dimension_pairs = [(8, 3), (12, 5), (16, 7), (20, 4)]
    for n, p in dimension_pairs:
        for sample in range(150):
            ratio = float(10.0 ** rng.uniform(-0.8, 0.8))
            J, P, B_i, D_i, C_i = general_system(
                n, p, M, c, eps, q, rng, coupling_ratio=ratio
            )
            eigenvalues = np.linalg.eigvals(J)
            diag = cluster_diagnostics(eigenvalues, c, bounds)
            if diag["inner_count"] != p or diag["annulus_count"] != 0:
                raise AssertionError("ensemble member violates separated theorem")
            rho = float(np.max(np.abs(eigenvalues)))
            if not bounds.lower - 1e-10 <= rho <= bounds.upper + 1e-10:
                raise AssertionError("ensemble member violates spectral-radius band")
            ensemble_rows.append(
                {
                    "n": n,
                    "p": p,
                    "sample": sample,
                    "rho": rho,
                    "lower": bounds.lower,
                    "upper": bounds.upper,
                    "symmetric_upper": bounds.symmetric_upper,
                    "full_inf_norm": inf_norm(J),
                    "P_inf_norm": inf_norm(P),
                    "C_deviation_inf_norm": inf_norm(C_i - c * np.eye(p)),
                    "feedback_product": inf_norm(B_i) * inf_norm(D_i),
                    "C_deviation_nonnormality": nonnormality(C_i - c * np.eye(p)),
                    "inner_count": diag["inner_count"],
                    "outer_count": diag["outer_count"],
                    "annulus_count": diag["annulus_count"],
                }
            )
    write_csv(DATA_DIR / "general_ensemble.csv", ensemble_rows)
    norm_inconclusive_count = sum(row["full_inf_norm"] > 1.0 for row in ensemble_rows)
    if norm_inconclusive_count != 193:
        raise AssertionError("ordinary-norm regression count has changed")

    corridor_rows: list[dict[str, object]] = []
    topologies = topology_family(n_example, 60, rng)
    c_values = np.linspace(0.58, 1.18, 121)
    for c_value in c_values:
        current = theoretical_bounds(M, float(c_value), eps, q)
        values = []
        for _, W in topologies:
            J, _, _ = worked_network_system(
                W, float(c_value), M, eps, q, B, D, memory_shape
            )
            rho = spectral_radius(J)
            if rho > current.upper + 1e-9:
                raise AssertionError("topology member violates unconditional upper bound")
            values.append(rho)
        if current.separated and (
            min(values) < current.lower - 1e-9 or max(values) > current.upper + 1e-9
        ):
            raise AssertionError("topology family violates separated band")
        if current.upper < 1.0:
            regime = "universal_recovery"
        elif current.separated and current.lower > 1.0:
            regime = "universal_instability"
        elif current.separated:
            regime = "intermediate"
        else:
            regime = "unseparated"
        corridor_rows.append(
            {
                "c": float(c_value),
                "rho_min": float(min(values)),
                "rho_max": float(max(values)),
                "upper": current.upper,
                "lower": current.lower if current.separated else "",
                "symmetric_upper": current.symmetric_upper if current.separated else "",
                "separated": int(current.separated),
                "regime": regime,
            }
        )
    write_csv(DATA_DIR / "corridor_scan.csv", corridor_rows)

    witness_n = 16
    witness_M = 0.65
    witness_c = 0.982
    witness_eps = 0.0
    witness_alpha = 0.1
    witness_b_max = 0.1
    witness_q = witness_alpha * witness_b_max
    witness_bounds = theoretical_bounds(
        witness_M, witness_c, witness_eps, witness_q
    )
    witness_A = np.diag(np.linspace(0.35, witness_M, witness_n))
    witness_B = np.diag(np.linspace(0.024, witness_b_max, witness_n))
    witness_C = witness_c * np.eye(witness_n)
    witness_D = witness_alpha * np.eye(witness_n)
    witness_rows: list[dict[str, object]] = []
    for shift in (9, 8):
        witness_W = permutation_shift(witness_n, shift)
        witness_J = assemble(witness_A @ witness_W, witness_B, witness_D, witness_C)
        witness_rho = spectral_radius(witness_J)
        witness_rows.append(
            {
                "n": witness_n,
                "M": witness_M,
                "c": witness_c,
                "eps": witness_eps,
                "q": witness_q,
                "alpha": witness_alpha,
                "b_min": 0.024,
                "b_max": witness_b_max,
                "permutation_shift": shift,
                "rho": witness_rho,
                "stable": int(witness_rho < 1.0),
                "family_lower": witness_bounds.lower,
                "family_upper": witness_bounds.upper,
            }
        )
    if not witness_rows[0]["rho"] < 1.0 < witness_rows[1]["rho"]:
        raise AssertionError("topology-only witness should straddle the unit circle")
    write_csv(DATA_DIR / "topology_witness.csv", witness_rows)

    scaling_rows: list[dict[str, object]] = []
    rho_reference = spectral_radius(J_example)
    for scale in np.logspace(-1.5, 1.5, 81):
        B_scaled = B / scale
        D_scaled = D * scale
        J_scaled = assemble(P_example, B_scaled, D_scaled, C_example)
        rho = spectral_radius(J_scaled)
        if abs(rho - rho_reference) > 1e-9:
            raise AssertionError("state rescaling should preserve the spectrum")
        scaling_rows.append(
            {
                "scale": float(scale),
                "rho": rho,
                "upper": bounds.upper,
                "full_inf_norm": inf_norm(J_scaled),
                "feedback_product": inf_norm(B_scaled) * inf_norm(D_scaled),
            }
        )
    write_csv(DATA_DIR / "state_scaling.csv", scaling_rows)

    coupling_rows: list[dict[str, object]] = []
    gap = c - M - eps
    q_values = np.linspace(0.0002, 0.92 * gap * gap / 4.0, 45)
    for q_value in q_values:
        current = theoretical_bounds(M, c, eps, float(q_value))
        if not current.separated:
            raise AssertionError("coupling benchmark should remain separated")
        comparison_matrix = np.array(
            [[M, np.sqrt(q_value)], [np.sqrt(q_value), c + eps]]
        )
        scalar_extremizer_rho = spectral_radius(comparison_matrix)
        if abs(scalar_extremizer_rho - current.upper) > 1e-12:
            raise AssertionError("comparison-matrix Perron root should equal U")
        rho_values = []
        for _ in range(40):
            n = int(rng.choice([8, 12, 16]))
            p = int(rng.choice([3, 5, 7]))
            J, _, _, _, _ = general_system(n, p, M, c, eps, float(q_value), rng)
            rho_values.append(spectral_radius(J))
        coupling_rows.append(
            {
                "q": float(q_value),
                "q_over_gap_squared": float(q_value / (gap * gap)),
                "rho_max": float(max(rho_values)),
                "scalar_extremizer_rho": scalar_extremizer_rho,
                "lower": current.lower,
                "upper": current.upper,
                "symmetric_upper": current.symmetric_upper,
            }
        )
    write_csv(DATA_DIR / "coupling_benchmark.csv", coupling_rows)

    return {
        "parameters": (M, c, eps, q),
        "bounds": bounds,
        "example_eigs": example_eigs,
        "diagnostics": diagnostics,
        "ensemble_rows": ensemble_rows,
        "norm_inconclusive_count": norm_inconclusive_count,
        "corridor_rows": corridor_rows,
        "witness_rows": witness_rows,
        "scaling_rows": scaling_rows,
        "coupling_rows": coupling_rows,
    }


def create_figure_one(results: dict[str, object]) -> None:
    """Create the spectral geometry, ensemble, and corridor figure."""

    M, c, eps, q = results["parameters"]
    bounds = results["bounds"]
    eigenvalues = results["example_eigs"]
    ensemble_rows = results["ensemble_rows"]
    corridor_rows = results["corridor_rows"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.15))

    ax = axes[0]
    ax.add_patch(
        Wedge(
            (c, 0.0),
            bounds.outer_radius,
            0,
            360,
            width=bounds.outer_radius - bounds.inner_radius,
            facecolor=LIGHT_GRAY,
            edgecolor="none",
            alpha=0.45,
            zorder=0,
        )
    )
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=GRAY, linestyle="--", linewidth=1.0))
    ax.add_patch(Circle((c, 0), bounds.inner_radius, fill=False, color=ORANGE, linewidth=1.5))
    ax.add_patch(Circle((c, 0), bounds.outer_radius, fill=False, color=PURPLE, linestyle=":", linewidth=1.5))
    inner = np.abs(eigenvalues - c) <= bounds.inner_radius + 1e-10
    ax.scatter(eigenvalues[~inner].real, eigenvalues[~inner].imag, marker="o", s=28,
               facecolors="none", edgecolors=BLUE, label="propagation cluster")
    ax.scatter(eigenvalues[inner].real, eigenvalues[inner].imag, marker="s", s=32,
               color=ORANGE, label="memory cluster")
    ax.plot([c], [0], marker="x", color="black", markersize=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.58, 1.25)
    ax.set_ylim(-0.78, 0.78)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    ax.set_title(r"(a) Empty annulus for $n=12$, $p=4$")
    ax.legend(loc="lower left", frameon=False)
    inset = ax.inset_axes([0.61, 0.63, 0.34, 0.29])
    inset.scatter(
        eigenvalues[inner].real,
        eigenvalues[inner].imag,
        marker="s",
        s=24,
        color=ORANGE,
    )
    inset.add_patch(
        Circle((c, 0), bounds.inner_radius, fill=False, color=ORANGE, linewidth=1.0)
    )
    inset.set_xlim(c - 1.12 * bounds.inner_radius, c + 1.12 * bounds.inner_radius)
    inset.set_ylim(-1.12 * bounds.inner_radius, 1.12 * bounds.inner_radius)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("4 memory eigenvalues", fontsize=6.5)

    ax = axes[1]
    pairs = [(8, 3), (12, 5), (16, 7), (20, 4)]
    jitter_rng = np.random.default_rng(SEED + 1)
    for index, pair in enumerate(pairs):
        values = [row["rho"] for row in ensemble_rows if (row["n"], row["p"]) == pair]
        x = index + jitter_rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(x, values, s=9, alpha=0.38, color=BLUE, edgecolors="none")
    ax.axhline(bounds.lower, color=ORANGE, linestyle="--")
    ax.axhline(bounds.upper, color=GREEN, linestyle="-")
    ax.axhline(bounds.symmetric_upper, color=PURPLE, linestyle=":")
    ax.set_xticks(range(len(pairs)), [f"{n},{p}" for n, p in pairs])
    ax.set_xlabel(r"block dimensions $(n,p)$")
    ax.set_ylabel(r"spectral radius $\rho(J)$")
    ax.set_title("(b) 600 rectangular nonnormal systems")
    ax.set_ylim(bounds.lower - 0.008, bounds.symmetric_upper + 0.008)
    ax.text(3.25, bounds.lower + 0.001, r"$c-R_-$", color=ORANGE, ha="right", va="bottom")
    ax.text(3.25, bounds.upper - 0.001, r"$U$", color=GREEN, ha="right", va="top")
    ax.text(3.25, bounds.symmetric_upper + 0.001, r"$c+R_-$", color=PURPLE, ha="right", va="bottom")

    ax = axes[2]
    c_values = np.array([row["c"] for row in corridor_rows], dtype=float)
    rho_min = np.array([row["rho_min"] for row in corridor_rows], dtype=float)
    rho_max = np.array([row["rho_max"] for row in corridor_rows], dtype=float)
    upper = np.array([row["upper"] for row in corridor_rows], dtype=float)
    lower = np.array(
        [float(row["lower"]) if row["lower"] != "" else np.nan for row in corridor_rows]
    )
    regimes = np.array([row["regime"] for row in corridor_rows])
    y_low, y_high = 0.55, 1.22
    ax.fill_between(c_values, y_low, y_high, where=regimes == "universal_recovery",
                    color="#D9F0E6", alpha=0.65, linewidth=0)
    ax.fill_between(c_values, y_low, y_high, where=regimes == "intermediate",
                    color="#FFF2CC", alpha=0.65, linewidth=0)
    ax.fill_between(c_values, y_low, y_high, where=regimes == "universal_instability",
                    color="#FADBD8", alpha=0.65, linewidth=0)
    ax.fill_between(c_values, rho_min, rho_max, color=BLUE, alpha=0.24,
                    label="60-topology range")
    ax.plot(c_values, rho_max, color=BLUE, linewidth=1.0)
    ax.plot(c_values, upper, color=GREEN, label=r"upper bound $U$")
    ax.plot(c_values, lower, color=ORANGE, linestyle="--", label=r"lower bound $c-R_-$")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="unit circle")
    ax.set_xlim(c_values.min(), c_values.max())
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"memory center $c$")
    ax.set_ylabel(r"spectral radius $\rho(J)$")
    ax.set_title("(c) Family-level stability corridors")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIGURE_DIR / "generalized_validation.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "generalized_validation.png", bbox_inches="tight")
    plt.close(fig)


def create_figure_two(results: dict[str, object]) -> None:
    """Create the coordinate-scaling and upper-bound comparison figure."""

    scaling_rows = results["scaling_rows"]
    coupling_rows = results["coupling_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))

    ax = axes[0]
    scales = np.array([row["scale"] for row in scaling_rows], dtype=float)
    rho = np.array([row["rho"] for row in scaling_rows], dtype=float)
    upper = np.array([row["upper"] for row in scaling_rows], dtype=float)
    full_norm = np.array([row["full_inf_norm"] for row in scaling_rows], dtype=float)
    ax.semilogx(scales, full_norm, color=GRAY, linestyle="--", label=r"ordinary $\|J\|_\infty$")
    ax.semilogx(scales, upper, color=GREEN, label=r"product bound $U$")
    ax.semilogx(scales, rho, color=BLUE, linewidth=2.0, label=r"actual $\rho(J)$")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"memory-coordinate scale $s$")
    ax.set_ylabel("bound or spectral radius (log scale)")
    ax.set_title("(a) Product bound survives state rescaling")
    ax.legend(frameon=False)

    ax = axes[1]
    x = np.array([row["q_over_gap_squared"] for row in coupling_rows], dtype=float)
    empirical = np.array([row["rho_max"] for row in coupling_rows], dtype=float)
    scalar = np.array([row["scalar_extremizer_rho"] for row in coupling_rows], dtype=float)
    upper = np.array([row["upper"] for row in coupling_rows], dtype=float)
    symmetric = np.array([row["symmetric_upper"] for row in coupling_rows], dtype=float)
    ax.plot(x, symmetric, color=PURPLE, linestyle=":", label=r"symmetric edge $c+R_-$")
    ax.plot(x, upper, color=GREEN, label=r"exact family upper edge $U$")
    ax.plot(x, scalar, color="black", linestyle="none", marker="o", fillstyle="none",
            markersize=3.6, markevery=4, label="scalar extremizer")
    ax.plot(x, empirical, color=BLUE, marker="o", markersize=2.5,
            markevery=4, label="ensemble maximum")
    ax.set_xlabel(r"normalized coupling $q/g^2$")
    ax.set_ylabel("upper edge or spectral radius")
    ax.set_title("(b) Exact family edge and random realizations")
    ax.legend(frameon=False)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIGURE_DIR / "bound_benchmark.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "bound_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    results = generate_validation_data()
    create_figure_one(results)
    create_figure_two(results)
    diagnostics = results["diagnostics"]
    bounds = results["bounds"]
    print(f"seed={SEED}")
    print(
        "worked example: "
        f"inner={diagnostics['inner_count']}, outer={diagnostics['outer_count']}, "
        f"annulus={diagnostics['annulus_count']}"
    )
    print(
        f"family band: [{bounds.lower:.6f}, {bounds.upper:.6f}], "
        f"symmetric upper={bounds.symmetric_upper:.6f}"
    )
    print(f"ordinary norm inconclusive: {results['norm_inconclusive_count']}/600")
    witness_rows = results["witness_rows"]
    print(
        "topology witness: "
        f"shift 9 rho={witness_rows[0]['rho']:.6f}, "
        f"shift 8 rho={witness_rows[1]['rho']:.6f}"
    )
    print("wrote data to ./data")
    print("wrote figures to ./figures")


if __name__ == "__main__":
    main()
