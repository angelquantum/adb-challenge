"""Classical sampling baselines (brief, Section 3b, step 9).

A term is a path of `years` annual states. Each method returns, per target,
an Estimate(mean, stderr, ci95) where ci95 is the 95% half-width.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
from scipy.stats import t as student_t

from .model import STATE_CAP, Scenario, annual_distribution, annual_states, get_scenario, losses
from .targets import TARGETS

_STATES = annual_states()
SOBOL_SCRAMBLES = 8


@dataclass(frozen=True)
class Estimate:
    mean: float
    stderr: float
    ci95: float


def _term_states(paths: np.ndarray) -> np.ndarray:
    """(n, years) annual-state indices -> (n, 4) capped term states."""
    return np.minimum(_STATES[paths].sum(axis=1), STATE_CAP)


def _payoffs(paths: np.ndarray, s: Scenario) -> dict[str, np.ndarray]:
    states = _term_states(paths)
    return {name: f(states, s) for name, f in TARGETS.items()}


def _inverse_cdf(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(p)
    return np.minimum(np.searchsorted(cdf, u * cdf[-1], side="right"), len(p) - 1)


def _normal(x: np.ndarray) -> Estimate:
    se = x.std(ddof=1) / np.sqrt(len(x))
    return Estimate(float(x.mean()), float(se), float(1.96 * se))


def monte_carlo(s: Scenario, n: int, rng: np.random.Generator) -> dict[str, Estimate]:
    p = annual_distribution(s).ravel()
    paths = _inverse_cdf(rng.random((n, s.years)), p)
    return {k: _normal(v) for k, v in _payoffs(paths, s).items()}


def sobol(s: Scenario, n: int, rng: np.random.Generator) -> dict[str, Estimate]:
    """8 independent scrambles of n/8 points each, t-interval on 7 dof."""
    p = annual_distribution(s).ravel()
    m = n // SOBOL_SCRAMBLES
    reps: dict[str, list[float]] = {k: [] for k in TARGETS}
    for _ in range(SOBOL_SCRAMBLES):
        engine = qmc.Sobol(d=s.years, scramble=True, seed=rng)
        u = engine.random_base2(int(np.log2(m))) if m & (m - 1) == 0 else engine.random(m)
        for k, v in _payoffs(_inverse_cdf(u, p), s).items():
            reps[k].append(v.mean())
    tq = student_t.ppf(0.975, SOBOL_SCRAMBLES - 1)
    out = {}
    for k, r in reps.items():
        r = np.asarray(r)
        se = r.std(ddof=1) / np.sqrt(SOBOL_SCRAMBLES)
        out[k] = Estimate(float(r.mean()), float(se), float(tq * se))
    return out


def stratified(s: Scenario, n: int, rng: np.random.Generator) -> dict[str, Estimate]:
    """Strata = first year with any payout; no-payout paths contribute exactly 0."""
    p = annual_distribution(s).ravel()
    p0 = p[0]  # state (0,0,0,0)
    p_pay = p.copy()
    p_pay[0] = 0.0  # annual dist conditional on a payout
    w = np.array([p0 ** (k - 1) * (1 - p0) for k in range(1, s.years + 1)])
    n_k = np.maximum(np.round(n * w / w.sum()).astype(int), 2)
    mean = {k: 0.0 for k in TARGETS}
    var = {k: 0.0 for k in TARGETS}
    for k in range(1, s.years + 1):
        paths = np.zeros((n_k[k - 1], s.years), dtype=int)
        paths[:, k - 1] = _inverse_cdf(rng.random(n_k[k - 1]), p_pay)
        if k < s.years:
            paths[:, k:] = _inverse_cdf(rng.random((n_k[k - 1], s.years - k)), p)
        for name, v in _payoffs(paths, s).items():
            mean[name] += w[k - 1] * v.mean()
            var[name] += w[k - 1] ** 2 * v.var(ddof=1) / len(v)
    return {k: Estimate(mean[k], float(np.sqrt(var[k])), float(1.96 * np.sqrt(var[k]))) for k in TARGETS}


def importance(s: Scenario, n: int, rng: np.random.Generator) -> dict[str, Estimate]:
    """Proposal per year: 0.5 * p + 0.5 * (p tilted by annual principal loss)."""
    p = annual_distribution(s).ravel()
    L1, L2 = losses(_STATES, s)
    tilt = p * (L1 + L2)
    q = 0.5 * p + 0.5 * tilt / tilt.sum()
    paths = _inverse_cdf(rng.random((n, s.years)), q)
    weight = np.prod(p[paths] / q[paths], axis=1)
    return {k: _normal(weight * v) for k, v in _payoffs(paths, s).items()}


METHODS: dict[str, Callable[[Scenario, int, np.random.Generator], dict[str, Estimate]]] = {
    "monte_carlo": monte_carlo,
    "sobol": sobol,
    "stratified": stratified,
    "importance": importance,
}


def estimate(scenario: str | Scenario, method: str, n: int, seed: int | None = None) -> dict[str, Estimate]:
    """Estimate all focus targets with `method` using a budget of `n` term paths."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    if n < 2 * SOBOL_SCRAMBLES:
        raise ValueError(f"n must be >= {2 * SOBOL_SCRAMBLES}")
    return METHODS[method](get_scenario(scenario), n, np.random.default_rng(seed))
