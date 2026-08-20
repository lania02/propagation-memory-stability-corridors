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
  of the unit-circle stability boundary.
- A coordinate-rescaling benchmark showing why the feedback product
  `q = ||B|| ||D||` is more informative than a raw full-matrix norm.
- A comparison between the exact family upper edge `U`, the symmetric reference
  edge, and the largest spectral radius observed in a generalized ensemble.
- A count of cases in which the ordinary full-matrix norm cannot certify
  recovery although the product bound can.

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

## Archiving

The public source repository is
<https://github.com/lania02/propagation-memory-stability-corridors>.
For a permanent version, archive a tagged release with Zenodo or another
DOI-granting repository and add the resulting DOI to `CITATION.cff` and the
manuscript Data Availability statement.
