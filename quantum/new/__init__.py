"""Iterative quantum amplitude estimation (IQAE) for the n-country joint-loss model.

Modules
    portfolio   n-country generalisation of the brief's model and its exact factorised solver
    baselines   classical samplers on n countries (Monte Carlo, Sobol, stratified, importance)
    iqae        IQAE (Grinko et al. 2021) with a pluggable backend: exact Grover statistics
                (oracle level) or the actual circuits on Qiskit Aer
    circuits    Qiskit oracles (shared-shock state preparation, loss register), Grover
                operator, gate and Clifford+T counts, Aer-backed sampler
    resources   fault-tolerant resource estimate
    benchmark   2, 3, 5 and 10 countries: exact, classical, IQAE and its limiting factors
    report      figures and the PDF report (docs/)

Run from the repo root so that `classical` is importable:

    python -m quantum.new --n 5 --target exceedance --eps 0.01
    python -m quantum.new.benchmark
    python -m quantum.new.report
"""
from .iqae import GroverModel, IQAEResult, iqae
from .portfolio import (
    LABELS,
    TARGETS,
    Portfolio,
    PortfolioModel,
    adb_portfolio,
    exact_targets,
    loss_distribution,
    shock_model,
)

__all__ = [
    "LABELS",
    "TARGETS",
    "GroverModel",
    "IQAEResult",
    "Portfolio",
    "PortfolioModel",
    "adb_portfolio",
    "exact_targets",
    "iqae",
    "loss_distribution",
    "shock_model",
]
