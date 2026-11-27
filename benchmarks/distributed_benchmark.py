"""Benchmark of distributed IQAE (distributed_iqae.py) against monolithic IQAE and the classical samplers.

python -m benchmarks.distributed_benchmark [--ns 2 3 5 10] [--reps 50] [--workers 8] [--quick]

Simulates distributed IQAE over a sweep of local precisions and renders the
query-cost and runtime comparisons with the monolithic benchmark.

Cost and time accounting
    oracle calls   sum over every QPU and every local IQAE task of shots x (2k + 1), i.e. forward
                   plus inverse state preparations, the same unit as the monolithic `queries`.
    QPU time       logical runtime model of resources.py at a 1 MHz logical T rate: each task costs
                   queries x T_per_query / 1e6 s, with T_per_query from the Clifford+T counts of its
                   own local circuit (rotation synthesis precision set by its deepest circuit).
                   Distributed: QPUs run in parallel, so wall time = slowest QPU + measured classical
                   preprocessing and aggregation. Monolithic: the same model on the full oracle
                   and the monolithic queries / k_max. Classical: paths x measured
                   seconds per path on this laptop.
"""
# ruff: noqa: E402
from __future__ import annotations

import os

# Limit numerical library threads before imports so worker processes can share cores.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from time import perf_counter

import numpy as np

from quantum.distributed_iqae import (
    Estimate,
    _aggregate,
    _group_law,
    _groups,
    _local_oracle,
    _local_preparation,
    _tasks,
)
from quantum.iqae import GroverModel, iqae
from quantum.portfolio import TARGETS, adb_portfolio, exact_targets, loss_unit, outcome_losses, shock_model
from quantum.resources import logical_estimate, rotation_t_cost

from . import analysis as an

NS = (2, 3, 5, 10)
EPS = (0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4)
ALPHA, SHOTS, MAX_ROUNDS = 0.05, 256, 30  # distributed_iqae defaults, rounds raised so small eps converge
T_RATE = 1e6  # logical T gates per second (optimistic rate of resources.py)


@cache
def _setup(n: int):
    pf = adb_portfolio(n)
    assignment, widths = _groups(pf, None, None)
    pz, cond = shock_model(pf)
    occupied = np.flatnonzero(pz > 0)
    unit = loss_unit(pf)
    units = np.rint(outcome_losses(pf) / unit).astype(np.int64)
    laws = [tuple(_group_law(g, w, cond[:, z], units) for z in occupied)
            for g, w in zip(assignment, widths, strict=True)]
    return pf, unit, pz[occupied], laws


@cache
def _local_counts(n: int, q: int, z: int, task: int) -> tuple[int, int, int, int]:
    """(t_A, rot_A, t_Q, rot_Q) of the local circuit for task `task` of stratum z on QPU q."""
    from quantum.circuits import ft_counts, grover_operator

    law = _setup(n)[3][q][z]
    A = _local_oracle(law, list(_tasks(law))[task], _local_preparation(law))
    fa, fq = ft_counts(A), ft_counts(grover_operator(A, aux=0))
    return fa["t_gates"], fa["rotations"], fq["t_gates"], fq["rotations"]


def _task_seconds(n, q, z, task, queries, k_max) -> float:
    t_A, rot_A, _, rot_Q = _local_counts(n, q, z, task)
    return queries * (t_A + rot_A * rotation_t_cost(rot_A + k_max * rot_Q)) / T_RATE


def run_once(n: int, eps: float, seed: int) -> dict:
    t0 = perf_counter()
    pf, unit, weights, laws = _setup(n)
    max_tasks = sum(len(law.support) + 2 for group in laws for law in group)
    local_alpha = ALPHA / (max_tasks * MAX_ROUNDS * (MAX_ROUNDS + 1) / 2)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(laws))]
    t_pre = perf_counter() - t0
    estimates, qpu_seconds, queries, k_max = [], [], 0, 0
    for q, (group, rng) in enumerate(zip(laws, rngs, strict=True)):
        rows, sec = [], 0.0
        for z, law in enumerate(group):
            active = law.probabilities > 0
            p = law.probabilities / law.probabilities.sum()
            row = []
            for task, values in enumerate(_tasks(law)):
                possible = values[active]
                if np.all(possible == possible[0]):
                    v = float(possible[0])
                    row.append(Estimate(v, v, v))
                    continue
                fit = iqae(GroverModel(float(values @ p)), eps, alpha=local_alpha, shots=SHOTS, rng=rng,
                           max_rounds=MAX_ROUNDS)
                lo, hi = fit.ci
                if hi <= lo:
                    lo, hi = 0.0, 1.0
                row.append(Estimate(float(fit.estimate), lo, hi))
                queries += fit.queries
                k_max = max(k_max, fit.k_max)
                sec += _task_seconds(n, q, z, task, fit.queries, fit.k_max)
            rows.append(row)
        estimates.append(rows)
        qpu_seconds.append(sec)
    t1 = perf_counter()
    targets = _aggregate(pf, unit, weights, laws, estimates)
    t_agg = perf_counter() - t1
    return dict(targets={k: v.endpoints() for k, v in targets.items()}, queries=queries, k_max=k_max,
                qpu_parallel=max(qpu_seconds), qpu_serial=sum(qpu_seconds), classical=t_pre + t_agg)


def _cell(args) -> list[dict]:
    n, eps, reps = args
    ex = exact_targets(_setup(n)[0])
    runs = [run_once(n, eps, seed=hash((2026, n, eps, r)) % 2**32) for r in range(reps)]
    out = []
    for t in TARGETS:
        est = np.array([r["targets"][t] for r in runs])
        err = est[:, 0] / ex[t] - 1
        out.append(dict(
            n=n, target=t, method="distributed_iqae", eps=eps, alpha=ALPHA, shots=SHOTS, reps=reps,
            qpus=n, exact=ex[t],
            queries=float(np.mean([r["queries"] for r in runs])),
            k_max_median=float(np.median([r["k_max"] for r in runs])),
            rel_rmse=float(np.sqrt(np.mean(err**2))), rel_bias=float(err.mean()),
            ci_coverage=float(np.mean((est[:, 1] <= ex[t]) & (ex[t] <= est[:, 2]))),
            qpu_seconds_parallel=float(np.mean([r["qpu_parallel"] for r in runs])),
            qpu_seconds_serial=float(np.mean([r["qpu_serial"] for r in runs])),
            classical_seconds=float(np.mean([r["classical"] for r in runs])),
        ))
    return out


def run(ns, reps: int, workers: int) -> list[dict]:
    jobs = [(n, e, reps) for n in sorted(ns, reverse=True) for e in EPS]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, cell in enumerate(pool.map(_cell, jobs), 1):
            rows += cell
            print(f"[{i}/{len(jobs)}] n={cell[0]['n']} eps={cell[0]['eps']:g} "
                  f"queries={cell[0]['queries']:.3g}", flush=True)
    return rows


DIST = dict(color="#6a4ce2", marker="h", label="Distributed IQAE (1 QPU / country)")


def _mono_seconds(data, n, t, rr) -> np.ndarray:
    c = data.counts(n, t)
    runtime_key = f"runtime_s_at_{int(T_RATE)}Hz"
    return np.array([
        logical_estimate(int(c["qubits_Q"]), int(c["t_A"]), int(c["rot_A"]), int(c["t_Q"]),
                         int(c["rot_Q"]), r["queries"], r["k_max_median"])[runtime_key]
        for r in rr
    ])


def figures(data: an.BenchmarkData) -> None:
    from .figures import STYLE, plot_accuracy_grid, setup

    setup()
    ns = data.ns()
    mono = dict(STYLE["iqae"], label="IQAE (monolithic)")

    def by_calls(n, t):
        for m in an.CLASSICAL:
            q, y, _ = data.curve(n, t, m)
            yield q, y, STYLE[m], 3
        q, y, _ = data.curve(n, t, "iqae")
        yield q, y, mono, 4
        q, y, _ = data.curve(n, t, "distributed_iqae")
        yield q, y, DIST, 5

    def by_time(n, t):
        for m in an.CLASSICAL:
            _, y, rr = data.curve(n, t, m)
            yield np.array([r["paths"] * r["seconds_per_path"] for r in rr]), y, STYLE[m], 3
        _, y, rr = data.curve(n, t, "iqae")
        yield _mono_seconds(data, n, t, rr), y, mono, 4
        _, y, rr = data.curve(n, t, "distributed_iqae")
        sec = np.array([r["qpu_seconds_parallel"] + r["classical_seconds"] for r in rr])
        yield sec, y, DIST, 5

    all_pay_ylim = (1e-11, 1.0)
    plot_accuracy_grid(ns, "paths / oracle calls", (5e2, 5e8), by_calls,
          "distributed_accuracy.png", "Distributed IQAE accuracy vs. computational work",
          note="Monolithic IQAE: one circuit per target.  Distributed IQAE: one run of n parallel local "
          "circuits (all five targets at once); oracle calls summed over all QPUs.",
          all_pay_ylim=all_pay_ylim)
    plot_accuracy_grid(ns, "time to solution (s)", (1e-4, 1e7), by_time,
          "distributed_runtime.png", "Distributed IQAE accuracy vs. time to solution",
          note="Classical: measured laptop time (paths times seconds per path).  "
          "IQAE: logical runtime estimate at a 1 MHz "
          "logical T rate (total T count / rate).\nDistributed: slowest QPU (QPUs run in parallel) + "
          "measured classical preprocessing and aggregation.", all_pay_ylim=all_pay_ylim)
    print("figures written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="fewer monolithic benchmark replications")
    a = ap.parse_args()

    from .benchmark import run_benchmark

    data = run_benchmark(a.ns, a.workers, a.quick, ("exact", "counts", "classical", "iqae"))
    data.distributed = run(a.ns, a.reps, a.workers)
    figures(data)


if __name__ == "__main__":
    main()
