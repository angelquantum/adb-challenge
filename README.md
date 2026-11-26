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

> Code not yet committed.