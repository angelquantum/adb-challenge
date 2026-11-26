"""n-country generalisation of the brief's model and its exact (factorised) solver.

Each country i keeps its own compound-Poisson earthquake process (rates from its four tail
probabilities, brief eqs. 2-3) and its own precipitation rate mu_i. One regional earthquake
stream with rate c = m * mean_i(rho_i * Lambda_i) hits every country at once (independent
quarter-step sizes per country), and one regional rain stream with rate c_r = share * min_i mu_i
triggers every country. Each country keeps its total rates (own-only rate = total - shared).
At n = 2 this is exactly the brief's model (classical/model.py, eqs. 4-6).

Conditional independence: given the number K of regional earthquakes in the term and whether
the regional rain stream fired (J), the countries are independent. A regional event moves every
country by at least one quarter step, so K >= 4 exhausts every earthquake sub-limit and K can be
lumped at 4 without approximation. The joint term law is therefore an exact mixture of
5 x 2 = 10 product laws over the shock states z = 2K + J. The exact solver and the quantum state
preparation (circuits.py) both use this structure.

Country outcome over the term: x = 2q + r, q in 0..4 quarter steps, r in {0, 1} rain trigger.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

from classical import TARGETS as BRIEF_TARGETS
from classical import Scenario, compound_poisson_rates, get_scenario

K_CAP = 4           # K >= 4 regional quakes exhaust every earthquake sub-limit
N_SHOCK = 10        # shock states z = 2K + J
N_OUTCOME = 10      # country outcomes x = 2q + r
X_EXHAUSTED = 9     # q = 4, r = 1: the country's full principal is lost

TARGETS = ("expected_loss", "all_pay", "exceedance", "stop_loss", "all_exhausted")
LABELS = {
    "expected_loss": "E[L]",
    "all_pay": "P(all pay)",
    "exceedance": "P(L > T)",
    "stop_loss": "E[(L - T)+]",
    "all_exhausted": "P(all exhausted)",
}
# At n = 2 the generalised targets are the brief's five focus targets (Table 3).
TO_BRIEF = dict(zip(TARGETS, BRIEF_TARGETS, strict=True))
PROBABILITY_TARGETS = ("all_pay", "exceedance", "all_exhausted")


@dataclass(frozen=True)
class Country:
    name: str
    eq_tail: tuple[float, ...]     # annual P(EQ payout >= 25/50/75/100% of the sub-limit)
    rho: float                     # P(other country triggers | this one triggers), same quake
    rain_prob: float = 0.02        # annual precipitation trigger probability q
    eq_limit: float = 70e6
    rain_limit: float = 10e6

    @property
    def principal(self) -> float:
        return self.eq_limit + self.rain_limit


@dataclass(frozen=True)
class Portfolio:
    countries: tuple[Country, ...]
    eq_m: float = 1.0              # shared earthquake multiplier m (eq. 4)
    rain_m: float = 1.0            # precipitation rate multiplier
    rain_share: float = 0.25       # c_r = share * min_i mu_i
    years: int = 3
    threshold: float = 80e6        # T

    @property
    def n(self) -> int:
        return len(self.countries)

    @property
    def total_principal(self) -> float:
        return float(sum(c.principal for c in self.countries))

    def rates(self):
        """(lambda[i, k], Lambda[i], c, mu[i], c_r)."""
        lam, Lam = zip(*(compound_poisson_rates(c.eq_tail) for c in self.countries), strict=True)
        Lam = np.array(Lam)
        c = self.eq_m * float(np.mean([ci.rho * L for ci, L in zip(self.countries, Lam, strict=True)]))
        if not 0 <= c <= Lam.min():
            raise ValueError(f"shared EQ rate c={c:.4g} must lie in [0, min Lambda={Lam.min():.4g}]")
        mu = np.array([-np.log(1 - ci.rain_prob) * self.rain_m for ci in self.countries])
        return np.array(lam), Lam, c, mu, self.rain_share * mu.min()


def adb_portfolio(n: int, scenario: str | Scenario = "adb-base") -> Portfolio:
    """n countries alternating the brief's Kyrgyz Republic and Tajikistan parameters.

    The threshold scales with the portfolio: T = half the total principal (USD 80m at n = 2).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    s = get_scenario(scenario)
    proto = (Country("KGZ", s.eq_tail_1, s.cond_2_given_1, s.pr_annual_prob, s.eq_limit, s.pr_limit),
             Country("TJK", s.eq_tail_2, s.cond_1_given_2, s.pr_annual_prob, s.eq_limit, s.pr_limit))
    countries = tuple(replace(proto[i % 2], name=f"{proto[i % 2].name}{i // 2 + 1}") for i in range(n))
    return Portfolio(countries, eq_m=s.eq_m, rain_m=s.pr_m, rain_share=s.pr_cr, years=s.years,
                     threshold=s.threshold * n / 2)


# ---------------------------------------------------------------- shock-state factorisation

def _own_eq_pmf(lam_own: np.ndarray, years: int) -> np.ndarray:
    """Capped quarter steps (0..4) from the country's own-only compound-Poisson stream."""
    G = np.zeros((5, 5))
    for a in range(5):
        for k in range(1, 5):
            G[a, min(a + k, 4)] += lam_own[k - 1]
    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(G, -G.sum(axis=1))
    return expm(years * G)[0]


def _add_event(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Add one event of size k ~ w (k = 1..4) to a capped step count p over 0..4."""
    out = np.zeros(5)
    for a in range(5):
        for k in range(1, 5):
            out[min(a + k, 4)] += p[a] * w[k - 1]
    return out


def shock_model(pf: Portfolio) -> tuple[np.ndarray, np.ndarray]:
    """(P(z) for the 10 shock states z = 2K + J, cond[i, z, x] = P(country i outcome x | z))."""
    lam, Lam, c, mu, c_r = pf.rates()
    Y = pf.years
    pK = poisson.pmf(np.arange(K_CAP), c * Y)
    pK = np.append(pK, max(0.0, 1 - pK.sum()))
    pJ = np.array([np.exp(-c_r * Y), -np.expm1(-c_r * Y)])
    pz = np.outer(pK, pJ).ravel()

    cond = np.zeros((pf.n, 5, 2, 5, 2))            # [i, K, J, q, r]
    for i in range(pf.n):
        w = lam[i] / Lam[i]
        eq = [_own_eq_pmf((Lam[i] - c) * w, Y)]
        for _ in range(K_CAP):
            eq.append(_add_event(eq[-1], w))
        p_own = -np.expm1(-(mu[i] - c_r) * Y)
        for K in range(K_CAP + 1):
            cond[i, K, 0] = np.outer(eq[K], [1 - p_own, p_own])
            cond[i, K, 1] = np.outer(eq[K], [0.0, 1.0])
    return pz, cond.reshape(pf.n, N_SHOCK, N_OUTCOME)


def outcome_losses(pf: Portfolio) -> np.ndarray:
    """loss[i, x]: principal loss of country i for term outcome x = 2q + r (brief eq. 7)."""
    q, r = np.arange(N_OUTCOME) // 2, np.arange(N_OUTCOME) % 2
    return np.array([c.eq_limit * q / 4 + c.rain_limit * r for c in pf.countries])


def loss_unit(pf: Portfolio) -> float:
    """Largest unit u such that every country loss is an integer multiple of u (USD 2.5m)."""
    return float(math.gcd(*[round(c.eq_limit / 4) for c in pf.countries],
                          *[round(c.rain_limit) for c in pf.countries]))


def loss_distribution(pf: Portfolio) -> tuple[np.ndarray, np.ndarray]:
    """Exact portfolio loss law: (loss values in USD, probabilities), cost O(10 n^2)."""
    pz, cond = shock_model(pf)
    u = loss_unit(pf)
    units = np.rint(outcome_losses(pf) / u).astype(int)
    size = int(units[:, X_EXHAUSTED].sum()) + 1
    pmf = np.zeros(size)
    for z in np.flatnonzero(pz > 0):
        cur = np.ones(1)
        for i in range(pf.n):
            ci = np.zeros(units[i].max() + 1)
            np.add.at(ci, units[i], cond[i, z])
            cur = np.convolve(cur, ci)
        pmf[: len(cur)] += pz[z] * cur
    return np.arange(size) * u, pmf


def exact_targets(pf: Portfolio) -> dict[str, float]:
    """The five generalised targets, exactly (reduce to the brief's Table 5 at n = 2)."""
    pz, cond = shock_model(pf)
    L, pmf = loss_distribution(pf)
    T = pf.threshold
    return {
        "expected_loss": float(pmf @ L),
        "all_pay": float(pz @ np.prod(1 - cond[:, :, 0], axis=0)),
        "exceedance": float(pmf[L > T + 0.5].sum()),
        "stop_loss": float(pmf @ np.maximum(L - T, 0.0)),
        "all_exhausted": float(pz @ np.prod(cond[:, :, X_EXHAUSTED], axis=0)),
    }


class PortfolioModel:
    """Everything the samplers and the oracles need, computed once per portfolio."""

    def __init__(self, pf: Portfolio):
        self.pf = pf
        self.n = pf.n
        self.pz, self.cond = shock_model(pf)
        self.loss = outcome_losses(pf)
        self.T = pf.threshold
        self.total = pf.total_principal
        self.exact = np.array(list(exact_targets(pf).values()))

    def scale(self, k: int) -> float:
        """Normaliser mapping target k to an amplitude in [0, 1]: f_max of its payoff."""
        return {0: self.total, 3: self.total - self.T}.get(k, 1.0)

    def amplitude(self, k: int) -> float:
        return float(self.exact[k] / self.scale(k))

    def values(self, x: np.ndarray) -> np.ndarray:
        """x: (N, n) outcomes -> (N, 5) per-path target values."""
        Li = self.loss[np.arange(self.n)[None, :], x]
        L = Li.sum(axis=1)
        return np.column_stack([
            L,
            (Li > 0.5).all(axis=1),
            L > self.T + 0.5,
            np.maximum(L - self.T, 0.0),
            (x == X_EXHAUSTED).all(axis=1),
        ]).astype(float)


# ---------------------------------------------------------------- brute force (verification)

def brute_force(pf: Portfolio) -> dict[str, float]:
    """Targets from the full joint chains on 5^n x 2^n states (small n only)."""
    if pf.n > 5:
        raise ValueError("brute force is limited to n <= 5")
    lam, Lam, c, mu, c_r = pf.rates()
    n, Y = pf.n, pf.years
    w = lam / Lam[:, None]

    states = list(itertools.product(range(5), repeat=n))
    index = {s: k for k, s in enumerate(states)}
    G = np.zeros((len(states), len(states)))
    combos = list(itertools.product(range(1, 5), repeat=n))
    p_combo = [np.prod([w[i, k - 1] for i, k in enumerate(ks)]) for ks in combos]
    for s in states:
        i0 = index[s]
        for i in range(n):
            for k in range(1, 5):
                t = list(s)
                t[i] = min(t[i] + k, 4)
                G[i0, index[tuple(t)]] += (Lam[i] - c) * w[i, k - 1]
        if c > 0:
            for ks, pk in zip(combos, p_combo, strict=True):
                G[i0, index[tuple(min(a + k, 4) for a, k in zip(s, ks, strict=True))]] += c * pk
    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(G, -G.sum(axis=1))
    p_eq = expm(Y * G)[0]

    rstates = list(itertools.product(range(2), repeat=n))
    rindex = {s: k for k, s in enumerate(rstates)}
    R = np.zeros((len(rstates), len(rstates)))
    for s in rstates:
        for i in range(n):
            if s[i] == 0:
                t = list(s)
                t[i] = 1
                R[rindex[s], rindex[tuple(t)]] += mu[i] - c_r
        if c_r > 0 and 0 in s:
            R[rindex[s], rindex[(1,) * n]] += c_r
    np.fill_diagonal(R, -R.sum(axis=1))
    p_r = expm(Y * R)[0]

    A, B = np.array(states), np.array(rstates)
    eq = np.array([ci.eq_limit for ci in pf.countries])
    rn = np.array([ci.rain_limit for ci in pf.countries])
    Li = (eq * A / 4)[:, None, :] + (rn * B)[None, :, :]
    P = np.outer(p_eq, p_r)
    L = Li.sum(-1)
    full = np.array([ci.principal for ci in pf.countries])
    return {
        "expected_loss": float((P * L).sum()),
        "all_pay": float(P[(Li > 0.5).all(-1)].sum()),
        "exceedance": float(P[L > pf.threshold + 0.5].sum()),
        "stop_loss": float((P * np.maximum(L - pf.threshold, 0)).sum()),
        "all_exhausted": float(P[(Li >= full - 0.5).all(-1)].sum()),
    }
