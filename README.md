# Reproducibility package: propagation--memory stability corridors

This directory contains the complete numerical workflow for the manuscript
**“Family-Uniform Stability Corridors in Coupled Propagation--Memory Systems.”**
The calculations validate the analytical bounds; they are not used as a
substitute for the proofs.

## What is reproduced

- A worked directed network with `n=12` propagation states, `p=4` persistent
  states, rectangular cross-couplings, and a heterogeneous nonnormal memory
  block.
- A 600-member general block ensemble with unequal dimensions.
- A 60-topology row-stochastic sweep across the three stability corridors.
- A fixed-parameter pair of permutation topologies that lie on opposite sides
  of the unit-circle stability boundary, their 201-point row-stochastic
  interpolation, and 4000-step perturbation responses on the two fixed networks.
- A coordinate-rescaling comparison of the raw full-matrix norm with the
  classical optimized weighted block-norm envelope. The latter equals `U`
  when `||B|| ||D|| = q` and is invariant under reciprocal block rescaling.
- A comparison between the exact family upper edge `U`, the symmetric reference
  edge, and the largest spectral radius observed in a generalized ensemble.
- A count of cases in which the ordinary full-matrix norm cannot certify
  recovery although the optimized block envelope can.

All random calculations use the fixed seed `26032026` and the induced infinity
norm.

## Repository layout

```text
reproducibility/
├── data/                    # generated CSV files
├── figures/                 # generated PDF and PNG figures
├── src/
│   ├── generate_results.py  # data and figure generation
│   └── validation.py        # model construction and theorem diagnostics
├── tests/
│   └── test_validation.py   # regression and randomized theorem checks
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Run from a clean environment

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/generate_results.py
python -m unittest discover -s tests -v
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/generate_results.py
python -m unittest discover -s tests -v
```

The script overwrites the generated CSV and figure files deterministically.

## Expected checks

The worked example must contain exactly `p=4` eigenvalues in the inner memory
disk, `n=12` eigenvalues outside the outer exclusion radius, and no eigenvalues
in the forbidden annulus. Every separated ensemble member must satisfy

```text
c - R_- <= rho(J) <= U <= c + R_-.
```

The test suite also covers the degenerate `q=0` case and an unseparated family,
for which only the unconditional upper bound is invoked. It also verifies that
the shift-9 and shift-8 permutation topologies reproduce the topology-only
stability witness reported in `data/topology_witness.csv`.

The interpolation fixes `A`, `B`, `C`, and `D` and changes only
`W_theta = (1-theta) W_shift9 + theta W_shift8`. Each value of `theta` defines
a separate autonomous system. It does not vary during a trajectory. Both
endpoint trajectories start from `ones(32)/sqrt(32)`; the exported response is
`||e_t||_2 / ||e_0||_2`. The unstable endpoint initially declines, so the
4000-step horizon is important for interpreting its asymptotic growth.

Additional outputs:

- `data/state_scaling.csv`: includes `optimized_block_envelope` alongside `U`.
- `data/topology_interpolation.csv`: all 201 spectra summarized by spectral
  radius, cluster counts, row-sum error, and family bounds.
- `data/topology_crossing.csv`: one bracketed numerical crossing of `rho=1`.
- `data/topology_trajectories.csv`: endpoint perturbation norms for steps 0--4000.
- `figures/topology_dynamics.pdf` and `.png`: interpolation and recovery/growth.

Regression checks cover weighted-norm optimization (including zero coupling),
the interpolation's fixed parameters and spectral counts, and agreement of
iterated trajectories with direct matrix powers. The full-family bounds are
sharp; the topology path is a particular subfamily and need not attain them.

## Archiving

The public source repository is
<https://github.com/lania02/propagation-memory-stability-corridors>.
For a permanent version, archive a tagged release with Zenodo or another
DOI-granting repository and add the resulting DOI to `CITATION.cff` and the
manuscript Data Availability statement.
