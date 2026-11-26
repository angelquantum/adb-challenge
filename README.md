# ADB Quantum Challenge — Joint-Loss Modelling for Natural Catastrophe Risk

Code for our Phase 1 submission: classical reproduction of the reference values, a first quantum amplitude estimation (QAE) prototype, and the Iterative QAE (IQAE) version used in the proposal.

## Repository structure

```
classical/      Classical reproduction: exact enumeration + sampling baselines
quantum/        QAE, first version
└── new/        Iterative QAE (shared-shock loading)
tests/          pytest suite for classical/
README.md
pyproject.toml  ruff + pytest config
requirements.txt
.gitignore
```

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/angelquantum/adb-challenge.git
cd adb-challenge

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 1. Classical reproduction — `classical/`

Reproduces the exact reference values of the brief (Table 5) for all scenarios (`adb-base`, `adb-independent`, `adb-common-shock`, `adb-rain-stress`, `adb-rain-cover`) by exact enumeration, and runs the classical baselines the quantum methods are compared against: plain Monte Carlo, scrambled Sobol, stratified sampling and importance sampling.

```bash
python -m classical                                   # from the repo root
python -m classical --scenario adb-independent --n 65536 --seed 0
```

Prints the exact Table 5 values for all scenarios, then each sampling baseline for the chosen scenario as estimate ± 95% CI and relative error against the exact value. Options: `--scenario` (default `adb-base`), `--n` term paths per method (default 65536), `--seed`.

| Scenario | Change from base |
|---|---|
| `adb-base` | Brief's base case |
| `adb-independent` | No earthquake or rain dependence between countries (`eq_m = 0`, `c_r = 0`) |
| `adb-common-shock` | Shared earthquake rate doubled (`eq_m = 2`) |
| `adb-rain-stress` | Rain intensity doubled (`pr_m = 2`) |
| `adb-rain-cover` | Limits shifted from earthquake to rain: EQ 60m, rain 20m |

| Target (three-year term) | Meaning |
|---|---|
| `term_expected_principal_loss_usd` | E[L1 + L2] |
| `term_joint_attachment_probability` | P(both bonds pay out) |
| `term_portfolio_loss_exceedance_probability` | P(L1 + L2 > T), T = 80m |
| `term_portfolio_stop_loss_usd` | E[max(L1 + L2 − T, 0)] |
| `term_both_bonds_exhausted_probability` | P(both bonds lose their full principal) |

The package is importable from other code. Run that code from the repo root as a module (e.g. `python -m quantum.new.<module>`) so `classical` is on the import path:

```python
from classical import term_distribution, annual_states, TARGETS, exact_targets
```

| Module | Contents |
|---|---|
| `model.py` | Scenarios, compound Poisson rates, EQ Markov chain, precipitation, annual and term distributions, losses |
| `targets.py` | The five focus targets as payoff functions `f(state, scenario)`, `exact_targets()` |
| `sampling.py` | Monte Carlo, scrambled Sobol, stratified, importance sampling, via `estimate()` |

Tests and lint (from the repo root):

```bash
pytest                  # normalisation, input-tail reproduction, sampling vs exact
ruff check .
```

## 2. QAE, first version — `quantum/`

First prototype: loads the joint loss distribution into amplitudes, computes the payout indicator into an objective qubit, and estimates the amplitude with canonical (phase-estimation-based) QAE. Runs on the Qiskit Aer simulator.

> Code not yet committed.

## 3. Iterative QAE — `quantum/new/`

Current method: shared-shock state loading combined with Iterative QAE (Grinko et al., 2021). No phase estimation, so circuits are shallower and closer to what current hardware can run. Reports estimates against the exact values, plus resource counts (qubits, depth, two-qubit gates, shots).

The brief's model is generalised to n countries (alternating Kyrgyz-like and Tajik-like parameters, one regional earthquake stream, one regional rain stream, T = half the total principal). At n = 2 it is exactly the brief's model and reproduces Table 5 through the `classical` package. Given the shared shock state z = (K regional quakes, J regional rain), the countries are independent, so the joint law is a mixture of 10 product laws. The exact solver and the quantum state preparation both use this.

Run from the repo root:

```bash
python -m quantum.new --n 5 --target exceedance --eps 0.01         # one IQAE run per target, with resources
python -m quantum.new --n 2 --backend aer                          # IQAE on the actual Qiskit circuits (Aer)
python -m quantum.new.benchmark                                    # 2, 3, 5, 10 countries -> quantum/new/results/ (~1 h, 12 cores)
python -m quantum.new.report                                       # figures + docs/IQAE_benchmark_and_limits_2-10_countries.pdf
```

`python -m quantum.new` options: `--n`, `--scenario`, `--target` (`all`, `expected_loss`, `all_pay`, `exceedance`, `stop_loss`, `all_exhausted`), `--eps` (relative CI half-width), `--alpha`, `--shots` (per round), `--backend oracle|aer`, `--shots-mode` (literal measured shots on Aer).

| Module | Contents |
|---|---|
| `portfolio.py` | n-country model, shock-state factorisation, exact solver, brute-force check |
| `baselines.py` | n-country Monte Carlo, scrambled Sobol, stratified (on the shock state), importance sampling |
| `iqae.py` | IQAE with Clopper-Pearson intervals and a pluggable backend: exact Grover statistics (with optional depolarising noise) or the circuits |
| `circuits.py` | Qiskit oracles (shared-shock loader, Fourier-adder loss register, all-pay / all-exhausted objectives, lookup oracle), Grover operator, gate and Clifford+T counts, `CircuitSampler` on Aer |
| `resources.py` | fault-tolerant estimate (T counts, code distance, physical qubits, run time) |
| `benchmark.py` | exact, classical baselines, IQAE, circuit counts, noise, depth/query trade-offs, Aer validation and end-to-end runs |
| `analysis.py`, `figures.py`, `report.py` | result tables, figures and the PDF |

Outputs in `quantum/new/results/`: `BENCHMARK.md` (summary tables), `exact.csv`, `classical.csv`, `iqae.csv`, `counts.csv`, `noise.csv`, `noisy_circuit.csv`, `tradeoff.csv`, `validation.csv`, `e2e.csv`, `figures/`. The report [docs/IQAE_benchmark_and_limits_2-10_countries.pdf](docs/IQAE_benchmark_and_limits_2-10_countries.pdf) covers the benchmark at 2, 3, 5 and 10 countries and the factors that limit IQAE: depth growing as the event gets rarer, gate-error requirements, noise bias, the depth–query trade-off, the confidence level, fault-tolerant cost, and the classical shortcut this model allows.

Tests: `pytest tests/test_iqae.py` checks Table 5 at n = 2, brute force at n = 3 and 4, the samplers, IQAE coverage, and that the oracles load the exact amplitudes on Aer. It also runs an end-to-end IQAE on the circuits.
