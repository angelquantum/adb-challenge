"""IQAE benchmark on portfolios of 2, 3, 5 and 10 countries, plus the factors that limit it.

python -m quantum.new.benchmark [--ns 2 3 5 10] [--workers 12] [--quick] [--only STEP ...]

Steps (all write CSVs to quantum/new/results/, then BENCHMARK.md is regenerated):
    exact      factorised exact solver (any n), brute force on 5^n x 2^n states (n <= 5)
    classical  Monte Carlo, scrambled Sobol, stratified, importance sampling; budgets
               2^10 .. 2^20 paths, 100 replications per cell
    iqae       oracle-level IQAE (exact Grover statistics, noiseless), 400 runs per accuracy
               level, alpha = 1e-4, 100 shots per round
    counts     qubits, CX, depth, T and rotation counts of the oracle A and Grover operator Q
    noise      noise-unaware IQAE under global depolarising noise, per-call rate
               lambda = p_2q x (CX per oracle call), p_2q = 1e-7 .. 1e-3
    tradeoff   shots per round (depth vs. query count) and confidence level alpha
    circuits   Aer validation of the oracles (n = 2, 3), Grover statistics, end-to-end IQAE on
               the actual circuits (n = 2), a qiskit-algorithms cross-check, and noisy Aer runs
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from . import analysis as an  # noqa: E402
from .baselines import METHODS  # noqa: E402
from .iqae import GroverModel, iqae  # noqa: E402
from .portfolio import LABELS, TARGETS, PortfolioModel, adb_portfolio, brute_force, exact_targets  # noqa: E402

NS = (2, 3, 5, 10)
BUDGETS = tuple(2**e for e in range(10, 21, 2))
CLASSICAL_REPS = 100
IQAE_REPS = 400
ALPHA = 1e-4
SHOTS = 100
EPS_REL = (3.0, 2.0, 1.4, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002)
MAX_QUERIES = 1e10
P2Q = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
TRADEOFF_SHOTS = (10, 30, 100, 300, 1000, 3000, 10000)
TRADEOFF_LEVEL = {"exceedance": 0.01, "all_exhausted": 0.10}   # accuracy at which settings are compared
NOISE_TARGETS = ("exceedance", "all_exhausted")


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng([2026, zlib.crc32(repr(key).encode())])


def iqae_stats(a: float, eps_rel: float, reps: int, rng, alpha=ALPHA, shots=SHOTS, noise=0.0) -> dict:
    """Run IQAE `reps` times on amplitude a and summarise accuracy, cost and depth."""
    model = GroverModel(a, noise)
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


# ---------------------------------------------------------------- tasks (run in worker processes)

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


def task_noise(n: int, target: str, p2q: float, cx_per_call: float, reps: int) -> list[dict]:
    m = PortfolioModel(adb_portfolio(n))
    a = m.amplitude(TARGETS.index(target))
    rng = rng_for("noise", n, target, p2q)
    lam = p2q * cx_per_call
    return [dict(n=n, target=target, p2q=p2q, lam=lam, cx_per_call=cx_per_call,
                 **iqae_stats(a, eps, reps, rng, noise=lam)) for eps in (0.1, 0.01)]


def task_tradeoff(kind: str, n: int, target: str, value: float, eps_rel: float, reps: int) -> list[dict]:
    m = PortfolioModel(adb_portfolio(n))
    a = m.amplitude(TARGETS.index(target))
    rng = rng_for("tradeoff", kind, n, target, value, eps_rel)
    kw = {"shots": int(value)} if kind == "shots" else {"alpha": value}
    return [dict(study=kind, n=n, target=target, **iqae_stats(a, eps_rel, reps, rng, **kw))]


def task_counts(n: int, target: str) -> list[dict]:
    from .circuits import ft_counts, gate_counts, grover_operator, oracle

    m = PortfolioModel(adb_portfolio(n))
    A = oracle(m, target)
    Q = grover_operator(A)
    ga, gq, fa, fq = gate_counts(A), gate_counts(Q), ft_counts(A), ft_counts(Q)
    return [dict(n=n, target=target, qubits_A=ga["qubits"], cx_A=ga["cx"], depth_A=ga["depth"],
                 qubits_Q=gq["qubits"], cx_Q=gq["cx"], depth_Q=gq["depth"],
                 t_A=fa["t_gates"], rot_A=fa["rotations"], t_Q=fq["t_gates"], rot_Q=fq["rotations"])]


def task_noisy_circuit(p2q: float, ks=(0, 1, 2, 3, 4), shots=2000) -> list[dict]:
    """n = 2, P(L > T), lookup oracle: measured hit rate of Q^k A under depolarising gate noise."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from .circuits import CircuitSampler, gate_counts, grover_operator, oracle

    m = PortfolioModel(adb_portfolio(2))
    k_t = TARGETS.index("exceedance")
    A = oracle(m, "exceedance", "lookup")
    Q = grover_operator(A)
    cs = CircuitSampler(A, Q)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q / 10, 1), ["u"])
    sim = AerSimulator(method="statevector", noise_model=nm)
    theta = np.arcsin(np.sqrt(m.amplitude(k_t)))
    cx_call = gate_counts(Q)["cx"] / 2
    rows = []
    for k in ks:
        qc = QuantumCircuit(cs.nq, 1)
        qc.compose(cs.At, qubits=range(cs.nA), inplace=True)
        for _ in range(k):
            qc.compose(cs.Qt, inplace=True)
        qc.measure(cs.obj, 0)
        t0 = time.perf_counter()
        hits = sim.run(qc, shots=shots, seed_simulator=100 + k).result().get_counts().get("1", 0)
        lam = p2q * cx_call
        d = np.exp(-lam * (2 * k + 1))
        rows.append(dict(p2q=p2q, k=k, shots=shots, hits=hits, measured=hits / shots,
                         ideal=float(np.sin((2 * k + 1) * theta) ** 2),
                         depolarising_model=float(d * np.sin((2 * k + 1) * theta) ** 2 + (1 - d) / 2),
                         cx_per_call=cx_call, seconds=time.perf_counter() - t0))
    return rows


# ---------------------------------------------------------------- circuit validation (main process)

def circuits_step(quick: bool) -> None:
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer.primitives import SamplerV2
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

    from .circuits import (
        BASIS,
        CircuitSampler,
        grover_operator,
        objective_index,
        objective_probability,
        oracle,
    )

    rows = []
    for n, kinds in ((2, ("lookup", "scalable")), (3, ("scalable",))):
        m = PortfolioModel(adb_portfolio(n))
        for k, t in enumerate(TARGETS):
            for kind in kinds:
                A = oracle(m, t, kind)
                t0 = time.perf_counter()
                v = objective_probability(A) * m.scale(k)
                rows.append(dict(check="amplitude, k = 0", n=n, target=t, oracle=kind, qubits=A.num_qubits,
                                 value=v, exact=m.exact[k], rel_dev=abs(v / m.exact[k] - 1),
                                 k_checked=0, seconds=time.perf_counter() - t0))
                print(f"  validate n={n} {t:14s} {kind:8s} rel dev {rows[-1]['rel_dev']:.1e}", flush=True)
    m = PortfolioModel(adb_portfolio(2))
    samplers = {}
    for k, t in enumerate(TARGETS):
        A = oracle(m, t, "lookup")
        samplers[t] = cs = CircuitSampler(A)
        theta = np.arcsin(np.sqrt(m.amplitude(k)))
        kk = 64
        t0 = time.perf_counter()
        dev = max(abs(cs.probability(j) - np.sin((2 * j + 1) * theta) ** 2) for j in range(kk))
        rows.append(dict(check="Grover statistics, max |P_k - sin^2((2k+1)theta)|", n=2, target=t,
                         oracle="lookup", qubits=cs.nq, value=dev, exact=0.0, rel_dev=dev,
                         k_checked=kk - 1, seconds=time.perf_counter() - t0))
    an.write(rows, "validation.csv")

    # End-to-end IQAE on the actual circuits (n = 2), same settings as the oracle-level runs.
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
            e2e.append(dict(backend="Aer statevector (shots drawn from circuit probabilities)",
                            implementation="quantum.new.iqae", n=2, target=t, eps_rel=eps, alpha=ALPHA,
                            reps=reps, exact=m.exact[k], estimate_mean=float(est.mean()),
                            rel_rmse=float(np.sqrt(np.mean((est / m.exact[k] - 1) ** 2))),
                            ci_coverage=float(np.mean([r.ci[0] <= a <= r.ci[1] for r in runs])),
                            queries=float(np.mean([r.queries for r in runs])),
                            k_max=float(np.median([r.k_max for r in runs])),
                            seconds=time.perf_counter() - t0))
            print(f"  e2e {t:14s} eps {eps:5.2f}: rmse {e2e[-1]['rel_rmse']:.3%} ({e2e[-1]['seconds']:.0f}s)",
                  flush=True)
    # Literal shots on Aer (circuit rebuilt per round), and the qiskit-algorithms reference IQAE.
    for t in ("exceedance", "all_pay"):
        k = TARGETS.index(t)
        a = m.amplitude(k)
        A = oracle(m, t, "lookup")
        Q = grover_operator(A)
        lit = CircuitSampler(A, Q, mode="shots")
        t0 = time.perf_counter()
        r = iqae(lit, 0.05 * a, alpha=0.05, shots=SHOTS, rng=rng_for("literal", t))
        e2e.append(dict(backend="Aer, measured shots", implementation="quantum.new.iqae", n=2, target=t,
                        eps_rel=0.05, alpha=0.05, reps=1, exact=m.exact[k],
                        estimate_mean=r.estimate * m.scale(k), rel_rmse=abs(r.estimate / a - 1),
                        ci_coverage=float(r.ci[0] <= a <= r.ci[1]), queries=r.queries, k_max=r.k_max,
                        seconds=time.perf_counter() - t0))
        prob = EstimationProblem(state_preparation=A, objective_qubits=[objective_index(A)],
                                 grover_operator=Q)
        pm = generate_preset_pass_manager(optimization_level=1, basis_gates=BASIS, seed_transpiler=0)
        ae = IterativeAmplitudeEstimation(epsilon_target=0.05 * a, alpha=0.05,
                                          sampler=SamplerV2(default_shots=SHOTS, seed=7), transpiler=pm)
        t0 = time.perf_counter()
        res = ae.estimate(prob)
        lo, hi = res.confidence_interval
        e2e.append(dict(backend="Aer SamplerV2, measured shots", implementation="qiskit_algorithms IQAE",
                        n=2, target=t, eps_rel=0.05, alpha=0.05, reps=1, exact=m.exact[k],
                        estimate_mean=res.estimation * m.scale(k), rel_rmse=abs(res.estimation / a - 1),
                        ci_coverage=float(lo <= a <= hi),
                        queries=float(sum(SHOTS * (2 * p + 1) for p in res.powers)),
                        k_max=float(max(res.powers)), seconds=time.perf_counter() - t0))
        print(f"  literal/qiskit-algorithms {t}: done", flush=True)
    an.write(e2e, "e2e.csv")


# ---------------------------------------------------------------- driver

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="fewer replications (smoke test)")
    ap.add_argument("--only", nargs="+", choices=["exact", "classical", "iqae", "counts", "noise",
                                                   "tradeoff", "circuits"], default=None)
    args = ap.parse_args()
    steps = args.only or ["exact", "classical", "iqae", "counts", "noise", "tradeoff", "circuits"]
    creps = 20 if args.quick else CLASSICAL_REPS
    qreps = 60 if args.quick else IQAE_REPS
    budgets = BUDGETS[:-1] if args.quick else BUDGETS
    t0 = time.perf_counter()

    if "exact" in steps:
        print("exact", flush=True)
        an.write(run_pool([(task_exact, (n,)) for n in args.ns], args.workers), "exact.csv")
    if "counts" in steps:
        print("counts", flush=True)
        an.write(run_pool([(task_counts, (n, t)) for n in args.ns for t in TARGETS], args.workers),
                 "counts.csv")
    if "iqae" in steps:
        print("iqae", flush=True)
        jobs = [(task_iqae, (n, t, qreps)) for n in args.ns for t in TARGETS]
        an.write(run_pool(jobs[::-1], args.workers), "iqae.csv")
    if "noise" in steps:
        print("noise", flush=True)
        an.clear_cache()
        cx = {(r["n"], r["target"]): r["cx_Q"] / 2 for r in an.table("counts.csv")}
        jobs = [(task_noise, (n, t, p, cx[(n, t)], qreps // 2)) for n in args.ns for t in NOISE_TARGETS
                for p in P2Q]
        jobs += [(task_noisy_circuit, (p,)) for p in (1e-4, 1e-3)]
        rows = run_pool(jobs, args.workers)
        an.write([r for r in rows if "eps_rel" in r], "noise.csv")
        an.write([r for r in rows if "k" in r], "noisy_circuit.csv")
    if "tradeoff" in steps:
        print("tradeoff", flush=True)
        # sweep eps for every setting: compare calls and depth at equal achieved RMSE
        jobs = [(task_tradeoff, ("shots", n, t, s, e, qreps // 2))
                for n, t in ((5, "exceedance"), (10, "all_exhausted"))
                for s in TRADEOFF_SHOTS for e in EPS_REL[:-1]]
        jobs += [(task_tradeoff, ("alpha", 5, "exceedance", al, e, qreps))
                 for al in (0.2, 0.05, 0.01, 1e-3, 1e-4, 1e-6)
                 for e in (0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005)]
        an.write(run_pool(jobs, args.workers), "tradeoff.csv")
    if "classical" in steps:
        print("classical", flush=True)
        jobs = [(task_classical, (n, m, N, creps)) for n in args.ns for m in METHODS for N in budgets]
        jobs.sort(key=lambda j: -j[1][0] * j[1][2])
        an.write(run_pool(jobs, args.workers), "classical.csv")
    if "circuits" in steps:
        print("circuits", flush=True)
        circuits_step(args.quick)
    an.clear_cache()
    write_markdown()
    print(f"done in {time.perf_counter() - t0:.0f}s -> {an.RESULTS}")


# ---------------------------------------------------------------- BENCHMARK.md

def write_markdown() -> None:
    from .resources import ft_estimate

    if not an.table("exact.csv"):
        return
    ns = an.ns()
    L = []
    L += ["# IQAE benchmark: 2, 3, 5 and 10 countries", "",
          "Generated by `python -m quantum.new.benchmark`. Model: the brief's two countries repeated "
          "(Kyrgyz-like, Tajik-like, alternating), one regional earthquake stream and one regional "
          "rain stream, T = half the total principal; identical to the brief at n = 2. Cost = term "
          "paths (classical) or oracle calls, i.e. applications of A or A^dagger (IQAE). IQAE is "
          f"simulated at oracle level on an ideal device (alpha = {ALPHA:g}, {SHOTS} shots per round).",
          "", "## Exact values", "",
          "| n | " + " | ".join(LABELS[t] for t in TARGETS) + " | exact solver | brute-force check |",
          "|---|" + "---|" * (len(TARGETS) + 2)]
    for n in ns:
        rr = {r["target"]: r for r in an.table("exact.csv") if r["n"] == n}
        vals = [f"USD {rr[t]['exact'] / 1e6:.2f}m" if t in ("expected_loss", "stop_loss")
                else an.fmt_sci(rr[t]["exact"], 3) for t in TARGETS]
        r0 = rr[TARGETS[0]]
        chk = f"{r0['brute_rel_dev']:.0e}" if np.isfinite(r0["brute_rel_dev"]) else "not run (10^n states)"
        L.append(f"| {n} | " + " | ".join(vals) + f" | {r0['t_exact_s'] * 1e3:.1f} ms | {chk} |")

    for level in (0.10, 0.01):
        L += ["", f"## Cost to reach {level:.0%} relative RMSE", "",
              "`~` extrapolated along 1/sqrt(N); `theory` = binomial formula for Monte Carlo runs "
              "that never saw a hit; `<=` met at the smallest budget tried; `>` not reached.", "",
              "| n | target | " + " | ".join(an.METHOD_LABEL[m] for m in an.CLASSICAL)
              + " | IQAE queries | IQAE k_max | best classical / IQAE |", "|---|---|" + "---|" * 7]
        for n in ns:
            for t in TARGETS:
                cells = []
                for m in an.CLASSICAL:
                    v, f, _ = an.cost_to(level, n, t, m)
                    cells.append({"theory": "~", "": ""}.get(f, f) + an.fmt_sci(v))
                qv, qf, km = an.cost_to(level, n, t, "iqae")
                r, meth, bound, _, _ = an.advantage(level, n, t)
                L.append(f"| {n} | {LABELS[t]} | " + " | ".join(cells)
                         + f" | {qf if qf in ('<=', '>') else ''}{an.fmt_sci(qv)} | {km:,.0f} | "
                         f"{bound}{r:.2g}x ({an.METHOD_LABEL.get(meth, meth)}) |")

    if an.table("counts.csv"):
        L += ["", "## Circuit resources (transpiled to CX + U, Qiskit optimisation level 1)", "",
              "| n | target | qubits A | CX per A | qubits Q | CX per Q | T gates Q | rotations Q |",
              "|---|---|---|---|---|---|---|---|"]
        for r in an.table("counts.csv"):
            L.append(f"| {r['n']:.0f} | {LABELS[r['target']]} | {r['qubits_A']:.0f} | {r['cx_A']:,.0f} | "
                     f"{r['qubits_Q']:.0f} | {r['cx_Q']:,.0f} | {r['t_Q']:,.0f} | {r['rot_Q']:,.0f} |")
        L += ["", "## Deepest circuit and fault-tolerant estimate", "",
              "| n | target | error | queries | k_max | CX deepest circuit | p_2q for 90% fidelity | "
              "logical qubits | physical qubits | time @ 1 MHz T | best classical (laptop) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, level in (("exceedance", 0.01), ("all_exhausted", 0.10)):
            for n in ns:
                c = an.counts(n, t)
                qv, _, km = an.cost_to(level, n, t, "iqae")
                if not np.isfinite(qv):
                    continue
                e = ft_estimate(int(c["qubits_Q"]), int(c["t_A"]), int(c["rot_A"]), int(c["t_Q"]),
                                int(c["rot_Q"]), qv, km)
                cxd = c["cx_A"] + km * c["cx_Q"]
                cv, meth, _ = an.best_classical(level, n, t)
                spp = np.median([r["seconds_per_path"] for r in an.table("classical.csv")
                                 if r["n"] == n and r["method"] == meth] or [np.nan])
                L.append(f"| {n} | {LABELS[t]} | {level:.0%} | {an.fmt_sci(qv)} | {km:,.0f} | "
                         f"{an.fmt_sci(cxd)} | {an.fmt_sci(0.105 / cxd)} | {e['logical_qubits']} | "
                         f"{an.fmt_sci(e['physical_qubits'])} | {e['runtime_s_at_1000000Hz']:.3g} s | "
                         f"{cv * spp:.3g} s ({an.METHOD_LABEL.get(meth, '')}) |")
    if an.table("e2e.csv"):
        L += ["", "## End-to-end IQAE on the Qiskit circuits (n = 2, Aer)", "",
              "| implementation | backend | target | eps | runs | estimate | exact | rel. RMSE / error | "
              "CI covers | queries | k_max | seconds |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in an.table("e2e.csv"):
            L.append(f"| {r['implementation']} | {r['backend']} | {LABELS[r['target']]} | "
                     f"{r['eps_rel']:.0%} | {r['reps']:.0f} | {r['estimate_mean']:.6g} | {r['exact']:.6g} | "
                     f"{r['rel_rmse']:.2%} | {r['ci_coverage']:.0%} | {an.fmt_sci(r['queries'])} | "
                     f"{r['k_max']:,.0f} | {r['seconds']:.0f} |")
    (an.RESULTS / "BENCHMARK.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
