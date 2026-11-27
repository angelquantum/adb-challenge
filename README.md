# ADB Quantum Challenge — Joint-Loss Modelling for Natural Catastrophe Risk

Phase 1 code: exact classical reproduction of the brief's reference values, and Iterative Quantum Amplitude Estimation (IQAE) for an n-country catastrophe-bond portfolio, both monolithic and distributed.

```
classical/   exact enumeration of the brief's model + classical sampling baselines
quantum/     n-country model, IQAE, Qiskit circuits, distributed IQAE, benchmarks
results/     benchmark CSVs, BENCHMARK.md, figures/
tests/       pytest suite
```

## Setup

Python 3.11+. Run everything from the repo root.

```bash
git clone https://github.com/angelquantum/adb-challenge.git && cd adb-challenge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest && ruff check .
```

## Targets

| Target (three-year term) | Meaning |
|---|---|
| `expected_loss` | E[L], total principal loss |
| `all_pay` | P(every bond pays out) |
| `exceedance` | P(L > T), T = 80m at n = 2 (half the total principal) |
| `stop_loss` | E[max(L − T, 0)] |
| `all_exhausted` | P(every bond loses its full principal) |

Scenarios: `adb-base` (the brief), `adb-independent` (no cross-country dependence), `adb-common-shock` (shared quake rate ×2), `adb-rain-stress` (rain intensity ×2), `adb-rain-cover` (limits EQ 60m / rain 20m).

## Classical — `classical/`

Reproduces Table 5 of the brief exactly for every scenario, then runs the baselines (Monte Carlo, scrambled Sobol, stratified, importance sampling) with 95% CIs.

```bash
python -m classical --scenario adb-base --n 65536
```

Modules: `model.py` (rates, EQ Markov chain, precipitation, term distributions), `targets.py` (payoffs, `exact_targets()`), `sampling.py` (`estimate()`).

## IQAE — `quantum/`

First at n=2 countries, we did a reproduction of 
brief's exact enumeration results.
Given the shared shock state, countries are independent, so the joint law is a mixture of 10 product laws.
Both the exact solver and the quantum state preparation use this. IQAE needs no phase estimation, so its circuits are shallower than canonical QAE's.
We also did a projection for case n>2 countries, to estimate the limitation as the app scales.

```bash
# IQAE vs exact, with resource counts
python -m quantum 
(--n [number_of_countries] --target exceedance --eps 0.01) 

python -m quantum.benchmark                              # n = 2, 3, 5, 10 -> results/ (~1 h, 12 cores)
python -m quantum.exact_scaling                          # exact solver scaling, P(all exhausted) cost
python -m quantum.report                                 # figures + docs/IQAE_benchmark_and_limits_2-10_countries.pdf
```

Options: `--n`, `--scenario`, `--target`, `--eps` (relative CI half-width), `--alpha`, `--shots`, `--backend oracle|aer`, `--shots-mode`, `--seed`.

| Module | Contents |
|---|---|
| `portfolio.py` | n-country model, shock factorisation, exact solver, brute-force check |
| `baselines.py` | n-country classical samplers |
| `iqae.py` | IQAE with Clopper-Pearson intervals; exact Grover statistics (optionally noisy) or circuit backend |
| `circuits.py` | Qiskit oracles, Grover operator, gate / Clifford+T counts, `CircuitSampler` on Aer |
| `resources.py` | fault-tolerant estimate (T count, code distance, physical qubits, run time) |
| `benchmark.py`, `exact_scaling.py` | benchmark runs writing to `results/` |
| `analysis.py`, `figures.py`, `report.py` | tables, figures, PDF report |

## Distributed IQAE — `quantum/distributed_iqae.py`

Country groups run independent local IQAE circuits, each conditioned on a shared-shock stratum fixed by a classical host. The host combines the local estimates: it adds expected losses, multiplies attachment/exhaustion probabilities, convolves group loss distributions for exceedance and stop loss, then averages over the shock weights. Two single-country QPUs need five logical qubits each.

```bash
python -m quantum.distributed_iqae --scenario adb-base --capacities 5 6 --seed 1729
python -m quantum.distributed_iqae --countries 3 --groups 0,1 2 --parallel
python -m quantum.distributed_benchmark                  # vs monolithic IQAE and classical -> results/distributed.csv
```

Outputs JSON with the five estimates, their intervals, per-QPU shots, preparations and timings. Options: `--groups` (comma-separated country indices per QPU), `--capacities` (qubits per QPU), `--epsilon`, `--alpha`, `--shots`, `--max-rounds`, `--parallel`.
