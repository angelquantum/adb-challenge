"""IQAE benchmark on portfolios of 2, 3, 5 and 10 countries, plus the factors that limit it.

python -m benchmarks.benchmark [--ns 2 3 5 10] [--workers 12] [--quick] [--only STEP ...]

Steps (the default run renders PNG plots from the in-memory results):
    exact      factorised exact solver (any n), brute force on 5^n x 2^n states (n <= 5)
    classical  Monte Carlo, scrambled Sobol, stratified, importance sampling; budgets
               2^10 .. 2^20 paths, 100 replications per cell
    iqae       oracle-level IQAE (exact Grover statistics, noiseless), 400 runs per accuracy
               level, alpha = 1e-4, 100 shots per round
    counts     qubits, CX, depth, T and rotation counts of the oracle A and Grover operator Q
    tradeoff   shots per round (depth vs. query count) and confidence level alpha
    distributed  oracle-level distributed IQAE
    circuits   PennyLane validation of the oracles (n = 2, 3), Grover statistics, end-to-end IQAE on
               the actual circuits (n = 2) with analytic and finite-shot simulation
"""
# ruff: noqa: E402
from __future__ import annotations

import os

# Limit numerical library threads before imports so worker processes can share cores.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from quantum.baselines import METHODS
from quantum.iqae import GroverModel, iqae
from quantum.portfolio import LABELS, TARGETS, PortfolioModel, adb_portfolio, brute_force, exact_targets

from . import analysis as an

NS = (2, 3, 5, 10)
BUDGETS = tuple(2**e for e in range(10, 21, 2))
CLASSICAL_REPS = 100
IQAE_REPS = 400
ALPHA = 1e-4
SHOTS = 100
EPS_REL = (3.0, 2.0, 1.4, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002)
MAX_QUERIES = 1e10
TRADEOFF_SHOTS = (10, 30, 100, 300, 1000, 3000, 10000)


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng([2026, zlib.crc32(repr(key).encode())])


def iqae_stats(a: float, eps_rel: float, reps: int, rng, alpha=ALPHA, shots=SHOTS) -> dict:
    """Run IQAE `reps` times on amplitude a and summarise accuracy, cost and depth."""
    model = GroverModel(a)
    eps = min(eps_rel * a, 0.49)
    runs = [iqae(model, eps, alpha=alpha, shots=shots, rng=rng) for _ in range(reps)]
    est = np.array([r.estimate for r in runs])
    err = est / a - 1
    q = np.array([r.queries for r in runs], float)
    km = np.array([r.k_max for r in runs], float)
    return dict(eps_rel=eps_rel, alpha=alpha, shots=shots, reps=reps, amplitude=a,
                queries=float(q.mean()), queries_p95=float(np.percentile(q, 95)),
                k_max_median=float(np.median(km)), k_max_max=float(km.max()),
                rounds=float(np.mean([r.rounds for r in runs])),
                rel_rmse=float(np.sqrt(np.mean(err**2))), rel_bias=float(err.mean()),
                rel_err_median=float(np.median(np.abs(err))),
                rel_err_p95=float(np.percentile(np.abs(err), 95)),
                ci_coverage=float(np.mean([r.ci[0] <= a <= r.ci[1] for r in runs])),
                ci_halfwidth_rel=float(np.mean([(r.ci[1] - r.ci[0]) / 2 for r in runs]) / a))


def task_exact(n: int) -> list[dict]:
    pf = adb_portfolio(n)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        ex = exact_targets(pf)
        best = min(best, time.perf_counter() - t0)
    t_bf, dev = np.nan, np.nan
    if n <= 5:
        t0 = time.perf_counter()
        bf = brute_force(pf)
        t_bf = time.perf_counter() - t0
        dev = max(abs(bf[k] / ex[k] - 1) for k in ex)
    m = PortfolioModel(pf)
    return [dict(n=n, target=t, label=LABELS[t], exact=ex[t], scale=m.scale(k), amplitude=m.amplitude(k),
                 total_principal=m.total, threshold=m.T, t_exact_s=best, t_brute_s=t_bf,
                 brute_rel_dev=dev, states=10.0**n) for k, t in enumerate(TARGETS)]


def task_classical(n: int, method: str, N: int, reps: int) -> list[dict]:
    m = PortfolioModel(adb_portfolio(n))
    rng = rng_for("classical", n, method, N)
    fn = METHODS[method]
    est, se, dof = [], [], np.inf
    t0 = time.perf_counter()
    for _ in range(reps):
        b = fn(m, N, rng)
        est.append(b.mean)
        se.append(b.stderr)
        dof = b.dof
    dt = (time.perf_counter() - t0) / reps
    est, se = np.array(est), np.array(se)
    err = est - m.exact
    z = student_t.ppf(0.975, dof) if np.isfinite(dof) else norm.ppf(0.975)
    return [dict(n=n, method=method, paths=N, target=t, exact=m.exact[k], reps=reps,
                 rel_rmse=float(np.sqrt(np.mean(err[:, k] ** 2)) / m.exact[k]),
                 rel_bias=float(err[:, k].mean() / m.exact[k]),
                 frac_zero=float(np.mean(est[:, k] == 0)),
                 ci95_coverage=float(np.mean(np.abs(err[:, k]) <= z * se[:, k])),
                 seconds_per_path=dt / N) for k, t in enumerate(TARGETS)]


def task_iqae(n: int, target: str, reps: int) -> list[dict]:
    m = PortfolioModel(adb_portfolio(n))
    k = TARGETS.index(target)
    rng = rng_for("iqae", n, target)
    rows = []
    for eps in EPS_REL:
        r = iqae_stats(m.amplitude(k), eps, reps, rng)
        rows.append(dict(n=n, target=target, method="iqae", **r))
        if r["queries"] > MAX_QUERIES:
            break
    return rows


def task_tradeoff(kind: str, n: int, target: str, value: float, eps_rel: float, reps: int) -> list[dict]:
    m = PortfolioModel(adb_portfolio(n))
    a = m.amplitude(TARGETS.index(target))
    rng = rng_for("tradeoff", kind, n, target, value, eps_rel)
    kw = {"shots": int(value)} if kind == "shots" else {"alpha": value}
    return [dict(study=kind, n=n, target=target, **iqae_stats(a, eps_rel, reps, rng, **kw))]


def task_counts(n: int, target: str) -> list[dict]:
    from quantum.circuits import ft_counts, gate_counts, grover_operator, oracle

    m = PortfolioModel(adb_portfolio(n))
    A = oracle(m, target)
    Q = grover_operator(A)
    ga, gq, fa, fq = gate_counts(A), gate_counts(Q), ft_counts(A), ft_counts(Q)
    return [dict(n=n, target=target, qubits_A=ga["qubits"], cx_A=ga["cx"], depth_A=ga["depth"],
                 qubits_Q=gq["qubits"], cx_Q=gq["cx"], depth_Q=gq["depth"],
                 t_A=fa["t_gates"], rot_A=fa["rotations"], t_Q=fq["t_gates"], rot_Q=fq["rotations"])]


def circuits_step(quick: bool) -> None:
    from quantum.circuits import CircuitSampler, grover_operator, objective_probability, oracle

    for n, kinds in ((2, ("lookup", "scalable")), (3, ("scalable",))):
        m = PortfolioModel(adb_portfolio(n))
        for k, t in enumerate(TARGETS):
            for kind in kinds:
                A = oracle(m, t, kind)
                v = objective_probability(A) * m.scale(k)
                dev = abs(v / m.exact[k] - 1)
                print(f"  validate n={n} {t:14s} {kind:8s} rel dev {dev:.1e}", flush=True)
    m = PortfolioModel(adb_portfolio(2))
    samplers = {}
    for k, t in enumerate(TARGETS):
        A = oracle(m, t, "lookup")
        samplers[t] = cs = CircuitSampler(A)
        theta = np.arcsin(np.sqrt(m.amplitude(k)))
        dev = max(abs(cs.probability(j) - np.sin((2 * j + 1) * theta) ** 2) for j in range(64))
        print(f"  validate n=2 {t:14s} Grover statistics max deviation {dev:.1e}", flush=True)

    e2e = []
    levels = (0.1,) if quick else (0.1, 0.05, 0.01)
    reps = 10 if quick else 50
    for k, t in enumerate(TARGETS):
        a = m.amplitude(k)
        for eps in levels:
            rng = rng_for("e2e", t, eps)
            t0 = time.perf_counter()
            runs = [iqae(samplers[t], min(eps * a, 0.49), alpha=ALPHA, shots=SHOTS, rng=rng)
                    for _ in range(reps)]
            est = np.array([r.estimate for r in runs]) * m.scale(k)
            e2e.append(dict(backend="PennyLane statevector (shots drawn from circuit probabilities)",
                            implementation="quantum.iqae", n=2, target=t, eps_rel=eps, alpha=ALPHA,
                            reps=reps, exact=m.exact[k], estimate_mean=float(est.mean()),
                            rel_rmse=float(np.sqrt(np.mean((est / m.exact[k] - 1) ** 2))),
                            ci_coverage=float(np.mean([r.ci[0] <= a <= r.ci[1] for r in runs])),
                            queries=float(np.mean([r.queries for r in runs])),
                            k_max=float(np.median([r.k_max for r in runs])),
                            seconds=time.perf_counter() - t0))
            print(f"  e2e {t:14s} eps {eps:5.2f}: rmse {e2e[-1]['rel_rmse']:.3%} ({e2e[-1]['seconds']:.0f}s)",
                  flush=True)
    for t in ("exceedance", "all_pay"):
        k = TARGETS.index(t)
        a = m.amplitude(k)
        A = oracle(m, t, "lookup")
        Q = grover_operator(A)
        lit = CircuitSampler(A, Q, mode="shots")
        t0 = time.perf_counter()
        r = iqae(lit, 0.05 * a, alpha=0.05, shots=SHOTS, rng=rng_for("literal", t))
        e2e.append(dict(backend="PennyLane, measured shots", implementation="quantum.iqae", n=2, target=t,
                        eps_rel=0.05, alpha=0.05, reps=1, exact=m.exact[k],
                        estimate_mean=r.estimate * m.scale(k), rel_rmse=abs(r.estimate / a - 1),
                        ci_coverage=float(r.ci[0] <= a <= r.ci[1]), queries=r.queries, k_max=r.k_max,
                        seconds=time.perf_counter() - t0))
        print(f"  finite-shot IQAE {t}: done", flush=True)
    for row in e2e:
        print(f"  {row['target']:14s} {row['backend']}: estimate={row['estimate_mean']:.6g}, "
              f"exact={row['exact']:.6g}, RMSE/error={row['rel_rmse']:.3%}, "
              f"queries={row['queries']:,.0f}, coverage={row['ci_coverage']:.0%}", flush=True)


def run_pool(jobs, workers: int) -> list[dict]:
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *args): (fn.__name__, args) for fn, args in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 10 == 0 or i == len(jobs):
                name, args = futs[f]
                print(f"  [{i}/{len(jobs)}] {name}{args[:3]} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


PLOT_STEPS = ("exact", "classical", "iqae", "counts", "tradeoff", "distributed")
STEPS = (*PLOT_STEPS, "circuits")


def run_benchmark(ns, workers: int, quick: bool, steps, distributed_reps: int = 50) -> an.BenchmarkData:
    creps = 20 if quick else CLASSICAL_REPS
    qreps = 60 if quick else IQAE_REPS
    budgets = BUDGETS[:-1] if quick else BUDGETS
    data = an.BenchmarkData()

    if "exact" in steps:
        print("exact", flush=True)
        data.exact_rows = run_pool([(task_exact, (n,)) for n in ns], workers)
    if "counts" in steps:
        print("counts", flush=True)
        data.count_rows = run_pool([(task_counts, (n, t)) for n in ns for t in TARGETS], workers)
    if "iqae" in steps:
        print("iqae", flush=True)
        jobs = [(task_iqae, (n, t, qreps)) for n in ns for t in TARGETS]
        data.iqae = run_pool(jobs[::-1], workers)
    if "tradeoff" in steps:
        print("tradeoff", flush=True)
        jobs = [(task_tradeoff, ("shots", n, t, s, e, qreps // 2))
                for n, t in ((5, "exceedance"), (10, "all_exhausted"))
                for s in TRADEOFF_SHOTS for e in EPS_REL[:-1]]
        jobs += [(task_tradeoff, ("alpha", 5, "exceedance", al, e, qreps))
                 for al in (0.2, 0.05, 0.01, 1e-3, 1e-4, 1e-6)
                 for e in (0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005)]
        data.tradeoff = run_pool(jobs, workers)
    if "classical" in steps:
        print("classical", flush=True)
        jobs = [(task_classical, (n, m, N, creps)) for n in ns for m in METHODS for N in budgets]
        jobs.sort(key=lambda j: -j[1][0] * j[1][2])
        data.classical = run_pool(jobs, workers)
    if "circuits" in steps:
        print("circuits", flush=True)
        circuits_step(quick)
    if "distributed" in steps:
        from .distributed_benchmark import run

        print("distributed", flush=True)
        data.distributed = run(ns, distributed_reps, workers)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="fewer replications (smoke test)")
    ap.add_argument("--only", nargs="+", choices=STEPS, help="run selected stages without writing plots")
    args = ap.parse_args()
    t0 = time.perf_counter()
    data = run_benchmark(args.ns, args.workers, args.quick, args.only or STEPS,
                         distributed_reps=5 if args.quick else 50)
    if args.only is None:
        from .distributed_benchmark import figures as distributed_figures
        from .figures import generate

        generate(data)
        distributed_figures(data)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
