"""Benchmark of distributed IQAE (distributed_iqae.py) against monolithic IQAE and the classical samplers.

python -m quantum.distributed_benchmark [--ns 2 3 5 10] [--reps 50] [--workers 8] [--only run|figures]

    run      oracle-level distributed IQAE (exact Grover statistics of every local circuit, noiseless,
             as in iqae.csv) over a sweep of local precisions epsilon; one run estimates all five
             targets at once. Writes results/distributed.csv.
    figures  f12_distributed_benchmark.png  relative RMSE vs paths / oracle calls (layout of f2)
             f13_time_benchmark.png         relative RMSE vs time to solution

Cost and time accounting
    oracle calls   sum over every QPU and every local IQAE task of shots x (2k + 1), i.e. forward
                   plus inverse state preparations, the same unit as the monolithic `queries`.
    QPU time       fault-tolerant model of resources.py at a 1 MHz logical T rate: each task costs
                   queries x T_per_query / 1e6 s, with T_per_query from the Clifford+T counts of its
                   own local circuit (rotation synthesis precision set by its deepest circuit).
                   Distributed: QPUs run in parallel, so wall time = slowest QPU + measured classical
                   preprocessing and aggregation. Monolithic: the same model on the full oracle
                   (counts.csv) and the queries / k_max of iqae.csv. Classical: paths x measured
                   seconds per path on this laptop (classical.csv).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import cache  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402

from . import analysis as an  # noqa: E402
from .distributed_iqae import Estimate, _aggregate, _group_law, _groups, _tasks  # noqa: E402
from .iqae import GroverModel, iqae  # noqa: E402
from .portfolio import TARGETS, adb_portfolio, exact_targets, loss_unit, outcome_losses, shock_model  # noqa: E402
from .resources import ft_estimate, rotation_t_cost  # noqa: E402

NS = (2, 3, 5, 10)
EPS = (0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4)
ALPHA, SHOTS, MAX_ROUNDS = 0.05, 256, 30      # distributed_iqae defaults, rounds raised so small eps converge
T_RATE = 1e6                                   # logical T gates per second (optimistic rate of resources.py)


# ---------------------------------------------------------------- local circuits (for Clifford+T counts)

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
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UCRYGate

    from .circuits import _cascade, ft_counts, grover_operator

    law = _setup(n)[3][q][z]
    data, obj = QuantumRegister(law.qubits - 1, "country"), QuantumRegister(1, "obj")
    A = QuantumCircuit(data, obj)
    mass = np.zeros(1 << len(data))
    mass[: len(law.probabilities)] = law.probabilities / law.probabilities.sum()
    _cascade(A, [], list(data), mass[None, :])
    values = list(_tasks(law))[task]
    angles = np.zeros_like(mass)
    angles[: len(values)] = 2 * np.arcsin(np.sqrt(values))
    A.append(UCRYGate(angles.tolist()), [obj[0], *data])
    fa, fq = ft_counts(A), ft_counts(grover_operator(A, aux=0))
    return fa["t_gates"], fa["rotations"], fq["t_gates"], fq["rotations"]


def _task_seconds(n, q, z, task, queries, k_max) -> float:
    t_A, rot_A, _, rot_Q = _local_counts(n, q, z, task)
    return queries * (t_A + rot_A * rotation_t_cost(rot_A + k_max * rot_Q)) / T_RATE


# ---------------------------------------------------------------- one distributed run, oracle level

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


def run(ns, reps: int, workers: int) -> None:
    jobs = [(n, e, reps) for n in sorted(ns, reverse=True) for e in EPS]
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, cell in enumerate(pool.map(_cell, jobs), 1):
            rows += cell
            print(f"[{i}/{len(jobs)}] n={cell[0]['n']} eps={cell[0]['eps']:g} "
                  f"queries={cell[0]['queries']:.3g}", flush=True)
    print("wrote", an.write(rows, "distributed.csv"))


# ---------------------------------------------------------------- figures

ROWS = ("expected_loss", "exceedance", "all_exhausted")
DIST = dict(color="#6a4ce2", marker="h", label="Distributed IQAE (1 QPU / country)")
YLIM = {"expected_loss": (3e-4, 0.5), "exceedance": (1e-3, 1.5), "all_exhausted": (2e-3, 3.2)}


def _dist_curve(n, t, key):
    rr = sorted((r for r in an.table("distributed.csv") if r["n"] == n and r["target"] == t),
                key=lambda r: r["queries"])
    return np.array([r[key] for r in rr]), np.array([r["rel_rmse"] for r in rr])


def _mono_seconds(n, t, rr) -> np.ndarray:
    c = an.counts(n, t)
    return np.array([ft_estimate(int(c["qubits_Q"]), int(c["t_A"]), int(c["rot_A"]), int(c["t_Q"]),
                                 int(c["rot_Q"]), r["queries"], r["k_max_median"])[f"runtime_s_at_{int(T_RATE)}Hz"]
                     for r in rr])


def _grid(ns, xlabel, xlim, series, name, note=None):
    from .figures import AXIS, INK2, SHORT, _plot, _save, plt

    fig, axes = plt.subplots(len(ROWS), len(ns), figsize=(7.2, 6.0), sharex=True, sharey="row")
    for i, t in enumerate(ROWS):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            for lvl in (0.1, 0.01):
                ax.axhline(lvl, color=AXIS, lw=0.8, zorder=1)
            for x, y, st, z in series(n, t):
                keep = (x >= xlim[0]) & (x <= xlim[1])
                _plot(ax, x[keep], np.minimum(y[keep], 3.0), st["color"], st["marker"], st["label"],
                      lw=1.3, ms=3.6, zorder=z)
            ax.set(xscale="log", yscale="log", xlim=xlim)
            if i == 0:
                ax.set_title(f"{n} countries")
            if j == 0:
                ax.set_ylabel(f"{SHORT[t]}\nrelative RMSE")
            if i == len(ROWS) - 1:
                ax.set_xlabel(xlabel)
            if j == len(ns) - 1:
                for lvl, txt in ((0.1, "10%"), (0.01, "1%")):
                    ax.text(1.03, lvl, txt, transform=ax.get_yaxis_transform(), fontsize=6.6,
                            color=INK2, va="center")
        axes[i, 0].set_ylim(YLIM[t])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=6.6, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.8, w_pad=0.5)
    _save(fig, name)


def figures(ns) -> None:
    from .figures import STYLE, setup

    setup()
    an.clear_cache()
    mono = dict(STYLE["iqae"], label="IQAE (monolithic)")

    def by_calls(n, t):
        for m in an.CLASSICAL:
            q, y, _ = an.curve(n, t, m)
            yield q, y, STYLE[m], 3
        q, y, _ = an.curve(n, t, "iqae")
        yield q, y, mono, 4
        q, y = _dist_curve(n, t, "queries")
        yield q, y, DIST, 5

    def by_time(n, t):
        for m in an.CLASSICAL:
            _, y, rr = an.curve(n, t, m)
            yield np.array([r["paths"] * r["seconds_per_path"] for r in rr]), y, STYLE[m], 3
        _, y, rr = an.curve(n, t, "iqae")
        yield _mono_seconds(n, t, rr), y, mono, 4
        rr = sorted((r for r in an.table("distributed.csv") if r["n"] == n and r["target"] == t),
                    key=lambda r: r["queries"])
        sec = np.array([r["qpu_seconds_parallel"] + r["classical_seconds"] for r in rr])
        yield sec, np.array([r["rel_rmse"] for r in rr]), DIST, 5

    _grid(ns, "paths / oracle calls", (5e2, 5e8), by_calls, "f12_distributed_benchmark.png",
          "Monolithic IQAE: one circuit per target.  Distributed IQAE: one run of n parallel local "
          "circuits (all five targets at once); oracle calls summed over all QPUs.")
    _grid(ns, "time to solution (s)", (1e-4, 1e7), by_time, "f13_time_benchmark.png",
          "Classical: measured laptop time (paths × s/path).  IQAE: fault-tolerant estimate at a 1 MHz "
          "logical T rate (total T count / rate).\nDistributed: slowest QPU (QPUs run in parallel) + "
          "measured classical preprocessing and aggregation.")
    print("figures written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", choices=("run", "figures"))
    a = ap.parse_args()
    if a.only != "figures":
        run(a.ns, a.reps, a.workers)
    if a.only != "run":
        figures(sorted(a.ns))


if __name__ == "__main__":
    main()
