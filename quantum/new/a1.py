"""A1: independent conditional IQAE circuits and classical portfolio aggregation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UCRYGate

from .circuits import CircuitSampler, _cascade, grover_operator
from .iqae import iqae
from .portfolio import (
    N_OUTCOME,
    TARGETS,
    X_EXHAUSTED,
    Portfolio,
    adb_portfolio,
    loss_unit,
    outcome_losses,
    shock_model,
)


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def endpoints(self) -> tuple[float, float, float]:
        return self.value, self.low, self.high


@dataclass(frozen=True)
class QPUWork:
    countries: tuple[int, ...]
    qubits: int
    tasks: int
    shots: int
    forward_preparations: int
    inverse_preparations: int


@dataclass(frozen=True)
class A1Result:
    targets: dict[str, Estimate]
    qpus: tuple[QPUWork, ...]
    shared_strata: int
    preprocessing_seconds: float
    local_execution_seconds: float
    aggregation_seconds: float


@dataclass(frozen=True)
class _Group:
    countries: tuple[int, ...]
    qubits: int
    probabilities: np.ndarray
    losses: np.ndarray
    support: np.ndarray
    payoffs: np.ndarray
    scale: int


def _groups(portfolio, assignment, capacities):
    groups = (
        tuple((i,) for i in range(portfolio.n))
        if assignment is None
        else tuple(tuple(group) for group in assignment)
    )
    if not groups or any(not group for group in groups):
        raise ValueError("Each QPU must have at least one country")
    if sorted(i for group in groups for i in group) != list(range(portfolio.n)):
        raise ValueError("Assign each country index exactly once")
    widths = tuple((N_OUTCOME ** len(group) - 1).bit_length() + 1 for group in groups)
    if capacities is not None:
        if len(capacities) != len(groups):
            raise ValueError("Supply one capacity per QPU")
        for i, (width, capacity) in enumerate(zip(widths, capacities, strict=True)):
            if capacity < width:
                raise ValueError(f"QPU {i + 1} requires {width} qubits; capacity is {capacity}")
    return groups, widths


def _group_law(group, width, cond, units):
    outcomes = np.indices((N_OUTCOME,) * len(group)).reshape(len(group), -1)
    countries = np.asarray(group)[:, None]
    probabilities = np.prod(cond[countries, outcomes], axis=0)
    paid = units[countries, outcomes]
    losses = paid.sum(axis=0)
    scale = int(units[list(group), X_EXHAUSTED].sum())
    payoffs = np.vstack((losses / scale, np.all(paid > 0, axis=0), np.all(outcomes == X_EXHAUSTED, axis=0)))
    support = np.unique(losses[probabilities > 0])
    return _Group(group, width, probabilities, losses, support, payoffs, scale)


def _tasks(law):
    yield from law.payoffs
    for cutoff in law.support[:-1]:
        yield law.losses <= cutoff


def _run_group(laws, epsilon, alpha, shots, max_rounds, seed):
    rng = np.random.default_rng(seed)
    estimates = []
    paid_tasks = total_shots = queries = 0
    for law in laws:
        data = QuantumRegister(law.qubits - 1, "country")
        obj = QuantumRegister(1, "obj")
        prep = QuantumCircuit(data, obj)
        mass = np.zeros(1 << len(data))
        mass[: len(law.probabilities)] = law.probabilities / law.probabilities.sum()
        _cascade(prep, [], list(data), mass[None, :])
        active = law.probabilities > 0
        row = []
        for values in _tasks(law):
            possible = values[active]
            if np.all(possible == possible[0]):
                value = float(possible[0])
                row.append(Estimate(value, value, value))
                continue
            angles = np.zeros_like(mass)
            angles[: len(values)] = 2 * np.arcsin(np.sqrt(values))
            circuit = prep.copy()
            circuit.append(UCRYGate(angles.tolist()), [obj[0], *data])
            sampler = CircuitSampler(circuit, grover_operator(circuit, aux=0), chunk=4)
            fitted = iqae(sampler, epsilon, alpha=alpha, shots=shots, rng=rng, max_rounds=max_rounds)
            lo, hi = fitted.ci
            # The existing controller collapses its interval on inversion failure.
            if hi <= lo:
                lo, hi = 0.0, 1.0
            row.append(Estimate(float(fitted.estimate), lo, hi))
            paid_tasks += 1
            total_shots += fitted.shots
            queries += fitted.queries
        estimates.append(row)
    first = laws[0]
    work = QPUWork(
        first.countries,
        first.qubits,
        paid_tasks,
        total_shots,
        (queries + total_shots) // 2,
        (queries - total_shots) // 2,
    )
    return estimates, work


def _loss_masses(support, estimates):
    """Return point, stochastically smaller, and stochastically larger PMFs."""
    points, lower, upper = np.asarray([e.endpoints() for e in (*estimates, Estimate(1.0, 1.0, 1.0))]).T
    lower = np.maximum.accumulate(lower)
    upper = np.minimum.accumulate(upper[::-1])[::-1]
    if np.any(lower > upper):
        lower = np.r_[np.zeros(len(support) - 1), 1.0]
        upper = np.ones(len(support))
    points = np.maximum.accumulate(np.clip(points, lower, upper))
    masses = np.zeros((3, int(support[-1]) + 1))
    masses[:, support] = np.diff([points, upper, lower], axis=1, prepend=0.0)
    return masses


def _aggregate(portfolio, unit, weights, laws_by_qpu, estimates_by_qpu):
    totals = np.zeros((len(TARGETS), 3))
    for z, weight in enumerate(weights):
        laws = [laws[z] for laws in laws_by_qpu]
        estimates = [rows[z] for rows in estimates_by_qpu]
        loss = (
            np.sum(
                [np.array(row[0].endpoints()) * law.scale for law, row in zip(laws, estimates, strict=True)],
                axis=0,
            )
            * unit
        )
        attach = np.prod([row[1].endpoints() for row in estimates], axis=0)
        exhaust = np.prod([row[2].endpoints() for row in estimates], axis=0)
        distribution = np.ones((3, 1))
        for law, row in zip(laws, estimates, strict=True):
            masses = _loss_masses(law.support, row[3:])
            distribution = np.array([np.convolve(a, b) for a, b in zip(distribution, masses, strict=True)])
        values = unit * np.arange(distribution.shape[1])
        exceedance = distribution[:, values > portfolio.threshold + 0.5].sum(axis=1)
        stop = distribution @ np.maximum(values - portfolio.threshold, 0.0)
        totals += weight * np.array([loss, attach, exceedance, stop, exhaust])
    return {
        key: Estimate(float(value), float(min(value, lo)), float(max(value, hi)))
        for key, (value, lo, hi) in zip(TARGETS, totals, strict=True)
    }


def run_a1(
    portfolio: Portfolio,
    *,
    groups: tuple[tuple[int, ...], ...] | None = None,
    capacities: tuple[int, ...] | None = None,
    epsilon: float = 0.02,
    alpha: float = 0.05,
    shots: int = 256,
    max_rounds: int = 12,
    seed: int = 1729,
    parallel: bool = False,
) -> A1Result:
    """Estimate all targets using independent local circuits, with one QPU per group."""
    if not 0 < epsilon < 0.5 or not 0 < alpha < 1:
        raise ValueError("Require 0 < epsilon < 0.5 and 0 < alpha < 1")
    if shots < 1 or max_rounds < 1:
        raise ValueError("Shots and max_rounds must be positive")
    started = perf_counter()
    assignment, widths = _groups(portfolio, groups, capacities)
    pz, cond = shock_model(portfolio)
    occupied = np.flatnonzero(pz > 0)
    unit = loss_unit(portfolio)
    units = np.rint(outcome_losses(portfolio) / unit).astype(np.int64)
    laws_by_qpu = [
        tuple(_group_law(group, width, cond[:, z], units) for z in occupied)
        for group, width in zip(assignment, widths, strict=True)
    ]
    max_tasks = sum(len(law.support) + 2 for laws in laws_by_qpu for law in laws)
    # Union bound over tasks and possible pooled block starts and lengths.
    # The existing controller further divides this alpha internally.
    local_alpha = alpha / (max_tasks * max_rounds * (max_rounds + 1) / 2)
    seeds = np.random.SeedSequence(seed).spawn(len(assignment))
    ready = perf_counter()

    def run_device(job):
        laws, child_seed = job
        return _run_group(laws, epsilon, local_alpha, shots, max_rounds, child_seed)

    jobs = list(zip(laws_by_qpu, seeds, strict=True))
    if parallel:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            runs = list(pool.map(run_device, jobs))
    else:
        runs = [run_device(job) for job in jobs]
    measured = perf_counter()
    estimates, work = zip(*runs, strict=True)
    targets = _aggregate(portfolio, unit, pz[occupied], laws_by_qpu, estimates)
    return A1Result(
        targets, work, len(occupied), ready - started, measured - ready, perf_counter() - measured
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="adb-base")
    parser.add_argument("--countries", type=int, default=2)
    parser.add_argument("--groups", nargs="+", help="Zero-based country groups, comma-separated within a QPU")
    parser.add_argument("--capacities", nargs="+", type=int, help="Maximum qubits on each QPU")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Normalized local IQAE precision")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=256, help="Shots per IQAE batch")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    groups = (
        None if args.groups is None else tuple(tuple(map(int, group.split(","))) for group in args.groups)
    )
    result = run_a1(
        adb_portfolio(args.countries, args.scenario),
        groups=groups,
        capacities=args.capacities,
        epsilon=args.epsilon,
        alpha=args.alpha,
        shots=args.shots,
        max_rounds=args.max_rounds,
        seed=args.seed,
        parallel=args.parallel,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
