# ADB Quantum Challenge — Joint-Loss Modelling for Natural Catastrophe Risk

Code for our Phase 1 submission: classical reproduction of the reference values, a first quantum amplitude estimation (QAE) prototype, and the Iterative QAE (IQAE) version used in the proposal.

## Repository structure

```
classical/      Classical reproduction: exact enumeration + sampling baselines
quantum/        QAE, first version
└── new/        Iterative QAE (shared-shock loading)
README.md
requirements.txt
.gitignore
```

## Installation

Requires Python 3.11+ (tested on 3.13).

```bash
git clone <repo-url> adb-challenge
cd adb-challenge

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 1. Classical reproduction — `classical/`

Reproduces the exact reference values of the brief (Table 5) for all scenarios (`adb-base`, `adb-independent`, `adb-common-shock`, `adb-rain-stress`, `adb-rain-cover`) by exact enumeration, and runs the classical baselines the quantum methods are compared against: plain Monte Carlo, scrambled Sobol, stratified sampling and importance sampling.

```bash
cd classical
python <main_script>.py
```

## 2. QAE, first version — `quantum/`

First prototype: loads the joint loss distribution into amplitudes, computes the payout indicator into an objective qubit, and estimates the amplitude with canonical (phase-estimation-based) QAE. Runs on the Qiskit Aer simulator.

```bash
cd quantum
python <main_script>.py
```

## 3. Iterative QAE — `quantum/new/`

Current method: shared-shock state loading combined with Iterative QAE (Grinko et al., 2021). No phase estimation, so circuits are shallower and closer to what current hardware can run. Reports estimates against the exact values, plus resource counts (qubits, depth, two-qubit gates, shots).

```bash
cd quantum/new
python <main_script>.py
```