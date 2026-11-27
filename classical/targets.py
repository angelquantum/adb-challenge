"""
Focus targets (brief, Table 3) as payoff functions f(term state) and exact values
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .model import Scenario, annual_states, get_scenario, losses, term_distribution

Payoff = Callable[[np.ndarray, Scenario], np.ndarray]


def expected_principal_loss(states: np.ndarray, s: Scenario) -> np.ndarray:
    L1, L2 = losses(states, s)
    return L1 + L2

def joint_attachment(states: np.ndarray, s: Scenario) -> np.ndarray:
    L1, L2 = losses(states, s)
    return ((L1 > 0) & (L2 > 0)).astype(float)

def portfolio_loss_exceedance(states: np.ndarray, s: Scenario) -> np.ndarray:
    L1, L2 = losses(states, s)
    return (L1 + L2 > s.threshold).astype(float)


def portfolio_stop_loss(states: np.ndarray, s: Scenario) -> np.ndarray:
    L1, L2 = losses(states, s)
    return np.maximum(L1 + L2 - s.threshold, 0.0)


def both_bonds_exhausted(states: np.ndarray, s: Scenario) -> np.ndarray:
    L1, L2 = losses(states, s)
    return ((L1 == s.principal) & (L2 == s.principal)).astype(float)


TARGETS: dict[str, Payoff] = {
    "term_expected_principal_loss_usd": expected_principal_loss,
    "term_joint_attachment_probability": joint_attachment,
    "term_portfolio_loss_exceedance_probability": portfolio_loss_exceedance,
    "term_portfolio_stop_loss_usd": portfolio_stop_loss,
    "term_both_bonds_exhausted_probability": both_bonds_exhausted,
}


def exact_targets(scenario: str | Scenario) -> dict[str, float]:
    """Exact enumeration: mu = sum_x p(x) f(x) over the 100 capped term states."""
    s = get_scenario(scenario)
    p = term_distribution(s).ravel()
    states = annual_states()  # term states share the annual state grid
    return {name: float(p @ f(states, s)) for name, f in TARGETS.items()}
