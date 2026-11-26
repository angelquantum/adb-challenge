"""Sanity and consistency tests for the classical model and baselines."""
import numpy as np
import pytest

from classical import (
    SCENARIOS,
    TARGETS,
    annual_distribution,
    compound_poisson_rates,
    eq_annual,
    estimate,
    exact_targets,
    get_scenario,
    pr_annual,
    term_distribution,
)
from classical.sampling import METHODS

ALL = list(SCENARIOS)


@pytest.mark.parametrize("sc", ALL)
def test_distributions_are_normalised_and_nonnegative(sc):
    for P in (eq_annual(sc), pr_annual(sc), annual_distribution(sc), term_distribution(sc)):
        assert np.all(P >= -1e-15)
        assert P.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("sc", ALL)
def test_eq_marginals_reproduce_input_tails(sc):
    s = get_scenario(sc)
    P = eq_annual(s)
    tail1 = [P[k:].sum() for k in range(1, 5)]
    tail2 = [P[:, k:].sum() for k in range(1, 5)]
    np.testing.assert_allclose(tail1, s.eq_tail_1, rtol=1e-9)
    np.testing.assert_allclose(tail2, s.eq_tail_2, rtol=1e-9)


@pytest.mark.parametrize("sc", ALL)
def test_precipitation_marginals(sc):
    s = get_scenario(sc)
    P = pr_annual(s)
    q = 1 - (1 - s.pr_annual_prob) ** s.pr_m
    assert P[1].sum() == pytest.approx(q)
    assert P[:, 1].sum() == pytest.approx(q)


def test_independent_scenario_factorises():
    P = eq_annual("adb-independent")
    np.testing.assert_allclose(P, np.outer(P.sum(1), P.sum(0)), atol=1e-14)
    R = pr_annual("adb-independent")
    np.testing.assert_allclose(R, np.outer(R.sum(1), R.sum(0)), atol=1e-14)


def test_dependence_raises_joint_risk():
    jp = {sc: exact_targets(sc)["term_joint_attachment_probability"] for sc in ALL}
    assert jp["adb-independent"] < jp["adb-base"] < jp["adb-common-shock"]


def test_expected_loss_is_independent_of_dependence():
    el = [exact_targets(sc)["term_expected_principal_loss_usd"] for sc in
          ("adb-base", "adb-independent", "adb-common-shock")]
    assert el == pytest.approx([el[0]] * 3, rel=1e-12)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        get_scenario("nope")
    with pytest.raises(ValueError):
        compound_poisson_rates((0.01, 0.02, 0.0, 0.0))  # increasing tail
    with pytest.raises(ValueError):
        estimate("adb-base", "nope", 1024)


@pytest.mark.parametrize("method", list(METHODS))
def test_sampling_agrees_with_exact(method):
    ex = exact_targets("adb-base")
    res = estimate("adb-base", method, 2**14, seed=1)
    for t in TARGETS:
        if t == "term_both_bonds_exhausted_probability":
            continue  # ~3e-5: too rare for this budget
        e = res[t]
        assert abs(e.mean - ex[t]) <= 5 * e.stderr + 1e-12, (t, e, ex[t])
