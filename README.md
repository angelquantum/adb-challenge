# ADB Quantum Challenge — Joint-Loss Modelling for Natural Catastrophe Risk

This repository is our Phase 1 prototype. It does three things:

1. **Reproduces the brief exactly.** Table 5 (all five scenarios, all five targets) comes out of an exact classical solver, alongside the classical sampling baselines.
2. **Estimates the same targets with Iterative Quantum Amplitude Estimation (IQAE)**, simulated on Qiskit, and reports the quantum resources it needs (qubits, depth, two-qubit gates, shots).
3. **Scales the problem beyond two countries** (3, 5 and 10) to show where IQAE wins against classical sampling, and where it doesn't.

Everything below runs on a laptop. Apart from the full benchmark, each step finishes in seconds.

---

## 1. Set up

You need Python 3.11 or newer. Run every command from the repository root.

```bash
git clone https://github.com/angelquantum/adb-challenge.git
cd adb-challenge
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check that everything works:

```bash
pytest
```

All 66 tests should pass in about 30 seconds. Among other things, they check that our code reproduces Table 5, that the quantum circuits load the exact probabilities, and that IQAE's confidence intervals hold.

---

## 2. Reproduce Table 5 of the brief

```bash
python -m classical
```

This takes about one second. The first block it prints is Table 5:

```
scenario                   EL (USD)         Both pay            L > T  Stop-loss (USD)   Both exhausted
adb-base                 10,063,513          4.4138%          1.2047%          282,355          0.0032%
adb-independent          10,063,513          2.7156%          0.7186%          171,266          0.0004%
adb-common-shock         10,063,513          5.0526%          1.6066%          385,043          0.0044%
adb-rain-stress          11,170,506          7.0464%          1.3215%          306,286          0.0069%
adb-rain-cover            9,970,051          4.4138%          0.6997%          170,927          0.0032%
```

Below it, the four classical baselines from the brief (Monte Carlo, scrambled Sobol, stratified and importance sampling) estimate the same targets. Each estimate comes with a 95% confidence interval and its error against the exact value. Note how all of them struggle with "both bonds exhausted", the 3-in-100,000 event.

Try another scenario or sample size with `python -m classical --scenario adb-rain-stress --n 100000`.

---

## 3. Run the quantum estimator

```bash
python -m quantum --n 2
```

This takes about two seconds. For each of the five targets, it prints the exact value, the IQAE estimate with its confidence interval, the number of oracle calls it took, and the size of the circuit. Where it applies, it also prints how many Monte Carlo samples would be needed for the same accuracy. For example:

```
P(all exhausted)  (all_exhausted)
  exact      3.1633e-05
  IQAE       3.15795e-05   CI [3.14765e-05, 3.16826e-05]   error -0.169%
  cost       2,132,300 oracle calls, ...   (Monte Carlo, same 95% half-width: ~1.3e+09 paths)
```

Your IQAE numbers will differ slightly from run to run, because each run draws new random shots.

By default this uses the **oracle backend**: it samples measurement outcomes from the exact statistics of the ideal circuit, so it is fast at any size. To run the **actual Qiskit circuits** on the Aer simulator instead (up to 3 countries):

```bash
python -m quantum --n 2 --backend aer --target all_pay --eps 0.05
```

This takes about 10 seconds. A tighter `--eps` or `--target all` also works, but takes 10+ minutes.

Useful options:

| Option | What it does | Default |
|---|---|---|
| `--n` | number of countries | 2 |
| `--scenario` | any scenario from Table 5 | `adb-base` |
| `--target` | `expected_loss`, `all_pay`, `exceedance`, `stop_loss`, `all_exhausted`, or `all` | `all` |
| `--eps` | target accuracy, as a fraction of the value (0.01 = ±1%) | 0.01 |
| `--alpha` | 1 − confidence level | 0.05 |
| `--backend` | `oracle` (fast) or `aer` (real circuits, n ≤ 3) | `oracle` |
| `--seed` | fix the random seed for a repeatable run | none |

### More than two countries

The brief notes that exact enumeration blows up as countries are added (10ⁿ states). We extend the model to *n* countries by alternating the Kyrgyz-like and Tajik-like parameters, with one regional earthquake stream and one regional rain stream. At n = 2 it is exactly the brief's model.

Once you condition on the shared regional events, the countries are independent. This makes exact answers cheap at any size, which lets us check every quantum estimate against the truth. It is also how the quantum circuit loads the distribution.

```bash
python -m quantum --n 10 --target exceedance
```

---

## 4. Distributed IQAE

Instead of one large circuit, each country (or group of countries) runs its own small circuit, as if on a separate small quantum processor. A classical computer then combines the results. With two countries, each processor needs only 5 qubits.

```bash
python -m quantum.distributed_iqae --capacities 5 6
```

This takes about five seconds and prints JSON with the five estimates, their intervals, and the shots and timings for each processor. To split three countries across two processors, use `--countries 3 --groups 0,1 2`.

---

## 5. Regenerate the full benchmark

The files in `results/` are the ones our report is based on. To regenerate them:

```bash
python -m quantum.benchmark                  # about 1 hour on 12 cores; set --workers to your core count
python -m quantum.exact_scaling              # a few minutes
python -m quantum.distributed_benchmark      # distributed vs. single-circuit IQAE vs. classical
python -m quantum.report                     # figures and docs/IQAE_benchmark_and_limits_2-10_countries.pdf
```

To check the pipeline end to end before committing an hour, run `python -m quantum.benchmark --quick` first.

| File in `results/` | What it holds |
|---|---|
| `BENCHMARK.md` | Summary tables. **Start here.** |
| `exact.csv` | Exact values for 2, 3, 5 and 10 countries, checked by brute force up to 5 |
| `classical.csv`, `iqae.csv` | Accuracy vs. cost for every classical method and for IQAE |
| `counts.csv` | Qubits, depth, two-qubit gates and T gates for each circuit |
| `noise.csv`, `noisy_circuit.csv` | How IQAE degrades under hardware noise |
| `tradeoff.csv` | Circuit depth vs. number of queries |
| `validation.csv`, `e2e.csv` | The real Qiskit circuits checked against exact values on Aer |
| `distributed.csv` | Distributed IQAE results |
| `exact_scaling.csv`, `exhausted_cost.csv` | How the exact solver and each method scale |
| `figures/` | All plots used in the report |

---

## Where the brief's Phase 1 deliverables are

| The brief asks for | Where to find it |
|---|---|
| Exact reference values (Table 5) | Section 2; [classical/](classical/) |
| Classical baselines | `python -m classical`; `results/classical.csv` |
| Quantum estimates | Sections 3 and 4; `results/iqae.csv`, `results/distributed.csv` |
| Resource estimation (qubits, depth, two-qubit gates, shots) | printed by `python -m quantum`; `results/counts.csv`; fault-tolerant estimate in [quantum/resources.py](quantum/resources.py) |
| Scaling to larger portfolios | `results/BENCHMARK.md`, `results/exact_scaling.csv` |

## Environment

| | |
|---|---|
| Hardware | Laptop, Intel Core i5-1135G7 (4 cores / 8 threads), 32 GB RAM. No GPU and no quantum hardware was used. |
| OS | Ubuntu Linux |
| Python | 3.12.3 |
| Quantum stack | Qiskit 2.6.2, Qiskit Aer 0.18.4 (statevector simulator), qiskit-algorithms 0.4.0 (cross-check only) |
| Exact pinned versions | [requirements.txt](requirements.txt) |

## AI use

**TODO — describe how AI tools were used (the brief requires this).**
