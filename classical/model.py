"""Build the exact annual and three-year loss distributions for two countries.

An outcome (a, b, r1, r2) records earthquake payout quarters (0 to 4)
and the rain trigger (0 or 1) for each country.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

EARTHQUAKE_STEPS = 5
RAIN_STEPS = 2
ANNUAL_SHAPE = (EARTHQUAKE_STEPS, EARTHQUAKE_STEPS, RAIN_STEPS, RAIN_STEPS)
N_ANNUAL = int(np.prod(ANNUAL_SHAPE))
STATE_CAP = np.array(ANNUAL_SHAPE) - 1

@dataclass(frozen=True)
class Scenario:
    """All inputs of one benchmark scenario. Defaults are the base case."""
    name: str = "adb-base"
    eq_tail_1: tuple[float, ...] = (0.03883, 0.02331, 0.01790, 0.01184) # Kyrgyz Republic
    eq_tail_2: tuple[float, ...] = (0.03925, 0.01743, 0.01332, 0.01043) # Tajikistan
    cond_2_given_1: float = 0.075 # Taj given Kyr
    cond_1_given_2: float = 0.074
    eq_m: float = 1.0
    pr_annual_prob: float = 0.02 # q, per country
    pr_cr: float = 0.25 # c_r
    pr_m: float = 1.0
    eq_limit: float = 70e6 # 70m on earthquakes
    pr_limit: float = 10e6 # 10m on rain
    years: int = 3
    threshold: float = 80e6 # T for portfolio exceedance / stop-loss

    @property
    def principal(self) -> float:
        """Per-bond principal: EQ limit + precipitation limit."""
        return self.eq_limit + self.pr_limit

_BASE = Scenario()
SCENARIOS: dict[str, Scenario] = {
    "adb-base"          : _BASE,
    "adb-independent"   : replace(_BASE,
                                  name="adb-independent",
                                  eq_m=0.0,
                                  pr_cr=0.0),
    "adb-common-shock"  : replace(_BASE,
                                  name="adb-common-shock",
                                  eq_m=2.0),
    "adb-rain-stress"   : replace(_BASE,
                                  name="adb-rain-stress",
                                  pr_m=2.0),
    "adb-rain-cover"    : replace(_BASE,
                                  name="adb-rain-cover",
                                  eq_limit=60e6,
                                  pr_limit=20e6),
}

def get_scenario(scenario: str | Scenario) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    try:
        return SCENARIOS[scenario]
    except KeyError:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}") from None

def compound_poisson_rates(tail: tuple[float, ...]) -> tuple[np.ndarray, float]:
    """
    Step 2, eqs (2)-(3): rates (λ1...λ4) and total rate λ
    """
    if len(tail) != EARTHQUAKE_STEPS - 1:
        raise ValueError(f"expected {EARTHQUAKE_STEPS - 1} tail probabilities, got {len(tail)}")
    if not 1 > tail[0] >= tail[1] >= tail[2] >= tail[3] >= 0:
        raise ValueError(f"tail probabilities must be non-increasing in [0, 1): {tail}")
    t1, t2, t3, t4 = tail
    p0, p1, p2, p3 = 1 - t1, t1 - t2, t2 - t3, t3 - t4
    Lam = -np.log(p0)
    l1 = p1 / p0
    l2 = p2 / p0 - l1**2 / 2
    l3 = p3 / p0 - l1 * l2 - l1**3 / 6
    l4 = Lam - l1 - l2 - l3
    rates = np.array([l1, l2, l3, l4])
    if np.any(rates < 0):
        raise ValueError(f"tail {tail} implies negative compound-Poisson rates {rates}")
    return rates, Lam


def shared_eq_rate(s: Scenario) -> float:
    """
    Step 3: eq. (4): shared-event rate c.
    """
    _, L1 = compound_poisson_rates(s.eq_tail_1)
    _, L2 = compound_poisson_rates(s.eq_tail_2)
    return s.eq_m * 0.5 * (s.cond_2_given_1*L1 + s.cond_1_given_2*L2)


def eq_generator(s: Scenario) -> np.ndarray:
    """
    Step 4: 25x25 CTMC generator on (a, b), index = 5*a + b
    """
    lam1, L1 = compound_poisson_rates(s.eq_tail_1)
    lam2, L2 = compound_poisson_rates(s.eq_tail_2)
    w1, w2 = lam1 / L1, lam2 / L2
    c = shared_eq_rate(s)
    if not 0 <= c <= min(L1, L2):
        raise ValueError(f"shared rate c={c:.4g} must lie in [0, min(L1, L2)={min(L1, L2):.4g}]")
    n = EARTHQUAKE_STEPS
    top = n - 1
    G = np.zeros((n * n, n * n))
    for a in range(n):
        for b in range(n):
            i = n * a + b
            for k in range(1, n):
                G[i, n * min(a + k, top) + b] += (L1 - c) * w1[k - 1]
                G[i, n * a + min(b + k, top)] += (L2 - c) * w2[k - 1]
                for j in range(1, n):
                    G[i, n * min(a + k, top) + min(b + j, top)] += c * w1[k - 1] * w2[j - 1]
    np.fill_diagonal(G, 0.0) # self-loops at the cap are no-ops
    np.fill_diagonal(G, -G.sum(axis=1))
    return G

def eq_annual(scenario: str | Scenario) -> np.ndarray:
    """
    Annual joint EQ distribution, shape (5, 5): P[a, b]
    """
    s = get_scenario(scenario)
    return expm(eq_generator(s))[0].reshape(EARTHQUAKE_STEPS, EARTHQUAKE_STEPS)

def pr_annual(scenario: str | Scenario) -> np.ndarray:
    """
    Step 5, eqs. (5)-(6): annual precipitation distribution, shape (2, 2): P[r1, r2]
    """
    s = get_scenario(scenario)
    mu = -np.log(1 - s.pr_annual_prob) * s.pr_m
    mu1 = mu2 = mu
    cr = s.pr_cr * min(mu1, mu2)
    P00 = np.exp(-(mu1 + mu2 - cr))
    P01 = np.exp(-mu1) - P00
    P10 = np.exp(-mu2) - P00
    P11 = 1 - P00 - P01 - P10
    return np.array([[P00, P01], [P10, P11]])

def annual_distribution(scenario: str | Scenario) -> np.ndarray:
    """
    Step 6: annual joint distribution, shape (5, 5, 2, 2): P[a, b, r1, r2]
    """
    return np.einsum("ab,rs->abrs", eq_annual(scenario), pr_annual(scenario))


def annual_states() -> np.ndarray:
    """
    All 100 states of a year as rows (a, b, r1, r2) in flattened C order
    """
    return np.array(np.unravel_index(np.arange(N_ANNUAL), ANNUAL_SHAPE)).T


def add_capped(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Add state arrays (..., 4) and cap at (4, 4, 1, 1)
    """
    return np.minimum(x + y, STATE_CAP)


def term_distribution(scenario: str | Scenario) -> np.ndarray:
    """
    Step 7: exact term distribution, shape (5, 5, 2, 2)
    """
    s = get_scenario(scenario)
    annual = annual_distribution(s).ravel()
    states = annual_states()
    # Map each pair of annual states to their capped sum.
    summed = add_capped(states[:, None, :], states[None, :, :])
    next_idx = np.ravel_multi_index(tuple(summed.transpose(2, 0, 1)), ANNUAL_SHAPE)
    term = np.zeros(N_ANNUAL)
    term[0] = 1.0 # start at (0, 0, 0, 0)
    for _ in range(s.years):
        new = np.zeros(N_ANNUAL)
        np.add.at(new, next_idx, term[:, None] * annual[None, :])
        term = new
    return term.reshape(ANNUAL_SHAPE)

def losses(states: np.ndarray, scenario: str | Scenario) -> tuple[np.ndarray, np.ndarray]:
    """
    Eq. (7): principal losses (L1, L2)
    for state rows (..., 4) = (a, b, r1, r2).
    """
    s = get_scenario(scenario)
    a, b, r1, r2 = (states[..., i] for i in range(4))
    top = EARTHQUAKE_STEPS - 1
    L1 = s.eq_limit * a / top + s.pr_limit * r1
    L2 = s.eq_limit * b / top + s.pr_limit * r2
    return L1, L2
