"""Tests for quantum: n-country model, n-country baselines, IQAE and the Qiskit oracles."""
import numpy as np
import pytest

from classical import SCENARIOS
from classical import exact_targets as brief_exact
from quantum.baselines import METHODS, estimate
from quantum.iqae import GroverModel, clopper_pearson, hit_probability, iqae
from quantum.portfolio import (
    TARGETS,
    TO_BRIEF,
    PortfolioModel,
    adb_portfolio,
    brute_force,
    exact_targets,
    loss_distribution,
    shock_model,
)
from quantum.resources import ft_estimate


@pytest.mark.parametrize("sc", list(SCENARIOS))
def test_two_countries_reproduce_table5(sc):
    ex = exact_targets(adb_portfolio(2, sc))
    ref = brief_exact(sc)
    for t in TARGETS:
        assert ex[t] == pytest.approx(ref[TO_BRIEF[t]], rel=1e-10)


@pytest.mark.parametrize("n", [3, 4])
def test_factorised_solver_matches_brute_force(n):
    pf = adb_portfolio(n)
    ex, bf = exact_targets(pf), brute_force(pf)
    for t in TARGETS:
        assert ex[t] == pytest.approx(bf[t], rel=1e-10)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_shock_model_is_normalised(n):
    pz, cond = shock_model(adb_portfolio(n))
    assert pz.sum() == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(cond.sum(-1), 1.0, atol=1e-13)
    L, pmf = loss_distribution(adb_portfolio(n))
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(pmf >= -1e-16) and L[-1] == pytest.approx(80e6 * n)


@pytest.mark.parametrize("method", list(METHODS))
def test_n_country_samplers_are_unbiased(method):
    m = PortfolioModel(adb_portfolio(4))
    b = METHODS[method](m, 2**15, np.random.default_rng(3))
    z = (b.mean - m.exact) / np.where(b.stderr > 0, b.stderr, np.inf)
    assert np.all(np.abs(z[:4]) < 5), z


def test_estimate_api_matches_classical_shape():
    est = estimate(adb_portfolio(3), "monte_carlo", 2**12, seed=1)
    assert set(est) == set(TARGETS)
    assert est["exceedance"].ci95 == pytest.approx(1.96 * est["exceedance"].stderr)


def test_hit_probability_limits():
    th = 0.3
    assert hit_probability(th, 3) == pytest.approx(np.sin(7 * th) ** 2)
    assert hit_probability(th, 3, noise=50.0) == pytest.approx(0.5)


def test_clopper_pearson_brackets_the_rate():
    lo, hi = clopper_pearson(10, 100, 0.05)
    assert lo < 0.1 < hi
    assert clopper_pearson(0, 100, 0.05)[0] == 0.0 and clopper_pearson(100, 100, 0.05)[1] == 1.0


@pytest.mark.parametrize("a", [0.3, 0.012, 3e-5])
def test_iqae_accuracy_and_coverage(a):
    rng = np.random.default_rng(2)
    runs = [iqae(GroverModel(a), 0.05 * a, alpha=0.01, rng=rng) for _ in range(200)]
    assert np.mean([r.ci[0] <= a <= r.ci[1] for r in runs]) >= 0.97
    assert all(abs(r.estimate - a) <= 2 * 0.05 * a or not (r.ci[0] <= a <= r.ci[1]) for r in runs)
    assert all(r.queries == sum(100 * (2 * k + 1) for k in r.powers) for r in runs)


def test_iqae_beats_monte_carlo_scaling():
    # query cost grows ~ 1/eps (Heisenberg), not 1/eps^2: 10x smaller eps -> < 20x more queries
    a, rng = 0.01, np.random.default_rng(0)
    q = [np.mean([iqae(GroverModel(a), e * a, alpha=0.01, rng=rng).queries for _ in range(50)])
         for e in (0.1, 0.01)]
    assert q[1] / q[0] < 20


def test_ft_estimate_monotone():
    a = ft_estimate(30, 100, 500, 200, 1000, 1e5, 100)
    b = ft_estimate(30, 100, 500, 200, 1000, 1e7, 10000)
    assert b["T_total"] > a["T_total"] and b["code_distance"] >= a["code_distance"]


# ---------------------------------------------------------------- circuits (Qiskit Aer)

qiskit_aer = pytest.importorskip("qiskit_aer")


@pytest.mark.parametrize("kind", ["scalable", "lookup"])
@pytest.mark.parametrize("target", TARGETS)
def test_oracles_load_the_exact_amplitude(target, kind):
    from quantum.circuits import objective_probability, oracle

    m = PortfolioModel(adb_portfolio(2))
    k = TARGETS.index(target)
    p1 = objective_probability(oracle(m, target, kind))
    assert p1 * m.scale(k) == pytest.approx(m.exact[k], rel=1e-9)


def test_grover_statistics_on_the_circuit():
    from quantum.circuits import CircuitSampler, oracle

    m = PortfolioModel(adb_portfolio(2))
    k = TARGETS.index("exceedance")
    cs = CircuitSampler(oracle(m, "exceedance", "lookup"), chunk=8)
    theta = np.arcsin(np.sqrt(m.amplitude(k)))
    for j in range(17):
        assert cs.probability(j) == pytest.approx(np.sin((2 * j + 1) * theta) ** 2, abs=1e-9)


def test_end_to_end_iqae_on_aer():
    from quantum.circuits import CircuitSampler, oracle

    m = PortfolioModel(adb_portfolio(2))
    k = TARGETS.index("all_pay")
    a = m.amplitude(k)
    r = iqae(CircuitSampler(oracle(m, "all_pay", "lookup"), mode="shots"), 0.1 * a, alpha=0.05, rng=1)
    assert r.ci[0] <= a <= r.ci[1]
