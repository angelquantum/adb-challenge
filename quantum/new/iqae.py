"""Iterative quantum amplitude estimation (Grinko, Gacon, Zoufal & Woerner, npj QI 7, 52, 2021).

IQAE estimates a = P(objective qubit = 1) after a state preparation A, without phase estimation.
Measuring the objective after Q^k A, with Q = A S_0 A^dagger S_chi, gives a hit with probability
sin^2((2k + 1) theta), a = sin^2(theta). Each round picks the largest k that keeps the current
confidence interval for theta inside one half-period of that oscillation, pools the shots taken
at that k, and narrows the interval with a Clopper-Pearson bound at level alpha / T. It stops
when the interval for a has half-width <= epsilon, which gives a rigorous 1 - alpha interval.

The algorithm only sees hit counts, so the backend is pluggable (a Sampler):

    GroverModel(a, noise)      exact statistics of the ideal circuit ("oracle level"), with
                               optional global depolarising noise per oracle call
    circuits.CircuitSampler    the actual Qiskit circuits, simulated with Qiskit Aer

Cost accounting: a round at Grover power k with N shots costs N (2k + 1) oracle calls
(applications of A or A^dagger). k_max, the deepest circuit, sets the coherence and
error-correction requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.special import betaincinv


class Sampler(Protocol):
    def __call__(self, k: int, shots: int, rng: np.random.Generator) -> int:
        """Number of objective = 1 outcomes in `shots` runs of Q^k A."""
        ...


def hit_probability(theta: float, k, noise: float = 0.0):
    """P(objective = 1) after Q^k A with global depolarising rate `noise` per oracle call.

    P = d sin^2((2k+1) theta) + (1 - d) / 2,  d = exp(-noise (2k + 1)).
    """
    f = 2 * np.asarray(k) + 1
    d = np.exp(-noise * f)
    return d * np.sin(f * theta) ** 2 + (1 - d) / 2


@dataclass(frozen=True)
class GroverModel:
    """Oracle-level backend: draws hits from the exact Grover statistics of amplitude a."""
    a: float
    noise: float = 0.0

    def __call__(self, k: int, shots: int, rng: np.random.Generator) -> int:
        theta = np.arcsin(np.sqrt(self.a))
        return int(rng.binomial(shots, hit_probability(theta, k, self.noise)))


@dataclass(frozen=True)
class IQAEResult:
    estimate: float
    ci: tuple[float, float]
    queries: int                    # oracle calls (A or A^dagger)
    shots: int                      # circuit executions
    k_max: int                      # deepest Grover power in a single circuit
    powers: tuple[int, ...] = field(repr=False)   # Grover power of every round

    @property
    def rounds(self) -> int:
        return len(self.powers)


def clopper_pearson(hits: int, n: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if hits == 0 else float(betaincinv(hits, n - hits + 1, alpha / 2))
    hi = 1.0 if hits == n else float(betaincinv(hits + 1, n - hits, 1 - alpha / 2))
    return lo, hi


def _next_power(k: int, th_lo: float, th_hi: float, up: bool, min_ratio: float = 2.0,
                chunk: int = 4096):
    """Largest K = 4k' + 2 in [min_ratio (4k + 2), pi / width] with [K th_lo, K th_hi] inside one
    half-period [m pi, (m + 1) pi]; "up" when m is even.

    Same answer as Grinko et al.'s loop over K = K_max, K_max - 4, ..., but searched over the
    half-period index m instead: for each m the valid K form the interval
    [m pi / th_lo, (m + 1) pi / th_hi], scanned from the top. The K loop needs ~pi / (4 width)
    steps when k cannot grow (e.g. under noise); this needs a handful.
    """
    w = th_hi - th_lo
    K_min = min_ratio * (4 * k + 2)
    K_max = np.floor(np.pi / max(w, 1e-300))
    if K_max < K_min:
        return k, up
    m_hi = 0 if th_lo <= 0 else int(np.floor(min(K_max * th_lo / np.pi, th_lo / w)))
    m_lo = max(0, int(np.ceil(K_min * th_hi / np.pi - 1)))
    for top in range(m_hi, m_lo - 1, -chunk):
        m = np.arange(top, max(m_lo, top - chunk + 1) - 1, -1, dtype=float)
        K = np.floor(np.minimum((m + 1) * np.pi / th_hi, K_max))
        K -= np.mod(K - 2, 4)
        lo = np.maximum(m * np.pi / th_lo if th_lo > 0 else 0.0, K_min)
        ok = K >= lo
        if ok.any():
            i = int(np.argmax(ok))
            return int(K[i] - 2) // 4, bool(int(m[i]) % 2 == 0)
    return k, up


def iqae(sampler: Sampler, epsilon: float, alpha: float = 0.05, shots: int = 100,
         rng: np.random.Generator | int | None = None, min_ratio: float = 2.0,
         max_rounds: int = 10_000) -> IQAEResult:
    """Run IQAE until the 1 - alpha interval for a has half-width <= epsilon (absolute)."""
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    rng = np.random.default_rng(rng)
    T = max(1, int(np.ceil(np.log2(np.pi / (8 * epsilon)))))
    th_lo, th_hi = 0.0, np.pi / 2
    k, up = 0, True
    pooled: dict[int, list[int]] = {}
    queries = n_shots = 0
    powers: list[int] = []
    for _ in range(max_rounds):
        if (np.sin(th_hi) ** 2 - np.sin(th_lo) ** 2) / 2 <= epsilon:
            break
        k, up = _next_power(k, th_lo, th_hi, up, min_ratio)
        K = 4 * k + 2
        h = sampler(k, shots, rng)
        powers.append(k)
        queries += shots * (2 * k + 1)
        n_shots += shots
        H, N = pooled.setdefault(k, [0, 0])
        pooled[k] = [H + h, N + shots]
        a_lo, a_hi = clopper_pearson(H + h, N + shots, alpha / T)
        if up:
            phi_lo, phi_hi = np.arccos(1 - 2 * a_lo), np.arccos(1 - 2 * a_hi)
        else:
            phi_lo, phi_hi = 2 * np.pi - np.arccos(1 - 2 * a_hi), 2 * np.pi - np.arccos(1 - 2 * a_lo)
        base = 2 * np.pi * np.floor(K * th_lo / (2 * np.pi))
        th_lo, th_hi = max(th_lo, (base + phi_lo) / K), min(th_hi, (base + phi_hi) / K)
        if th_lo > th_hi:        # interval miss (probability <= alpha): collapse and stop
            th_lo = th_hi = 0.5 * (th_lo + th_hi)
            break
    lo, hi = float(np.sin(th_lo) ** 2), float(np.sin(th_hi) ** 2)
    return IQAEResult(float(np.sin(0.5 * (th_lo + th_hi)) ** 2), (lo, hi), queries, n_shots,
                      max(powers, default=0), tuple(powers))
