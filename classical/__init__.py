"""Classical reproduction of the ADB joint-loss benchmark.

Exact model and classical baselines for the two-country (Kyrgyz Republic,
Tajikistan) catastrophe bond portfolio of the ADB Quantum Challenge brief.
Reproduces the exact reference values of Table 5 for all five scenarios and
the four sampling baselines (Monte Carlo, scrambled Sobol, stratified,
importance sampling).

State x = (a, b, r1, r2): cumulative earthquake quarter steps per country
(0..4) and precipitation triggers (0/1), i.e. 100 states, flattened in C order
over shape (5, 5, 2, 2). Every target is mu = sum_x p(x) f(x).

Typical use (e.g. from the quantum code):

    from classical import term_distribution, annual_states, TARGETS, exact_targets

    p = term_distribution("adb-base").ravel()        # p(x), 100 probabilities
    x = annual_states()                              # (100, 4) state rows
    f = TARGETS["term_both_bonds_exhausted_probability"](x, SCENARIOS["adb-base"])
    mu = p @ f                                       # == exact_targets(...)[...]

Run `python -m classical` from the repo root to print the reproduction.
"""
from .model import (
    ANNUAL_SHAPE,
    N_ANNUAL,
    SCENARIOS,
    Scenario,
    add_capped,
    annual_distribution,
    annual_states,
    compound_poisson_rates,
    eq_annual,
    eq_generator,
    get_scenario,
    losses,
    pr_annual,
    shared_eq_rate,
    term_distribution,
)
from .sampling import METHODS, Estimate, estimate
from .targets import TARGETS, exact_targets

__all__ = [
    "ANNUAL_SHAPE",
    "METHODS",
    "N_ANNUAL",
    "SCENARIOS",
    "TARGETS",
    "Estimate",
    "Scenario",
    "add_capped",
    "annual_distribution",
    "annual_states",
    "compound_poisson_rates",
    "eq_annual",
    "eq_generator",
    "estimate",
    "exact_targets",
    "get_scenario",
    "losses",
    "pr_annual",
    "shared_eq_rate",
    "term_distribution",
]
