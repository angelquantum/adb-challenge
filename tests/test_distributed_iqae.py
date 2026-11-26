import numpy as np
import pytest

from quantum.new.distributed_iqae import (
    Estimate,
    _aggregate,
    _group_law,
    _loss_masses,
    _tasks,
    run_distributed_iqae,
)
from quantum.new.portfolio import adb_portfolio, exact_targets, loss_unit, outcome_losses, shock_model


@pytest.mark.parametrize("scenario", ["adb-base", "adb-common-shock", "adb-rain-cover"])
@pytest.mark.parametrize("groups", [((0,), (1,)), ((0, 1),)])
def test_distributed_iqae_aggregation_matches_exact_conditional_reference(scenario, groups):
    portfolio = adb_portfolio(2, scenario)
    pz, cond = shock_model(portfolio)
    unit = loss_unit(portfolio)
    units = np.rint(outcome_losses(portfolio) / unit).astype(np.int64)
    occupied = np.flatnonzero(pz > 0)
    laws = [
        [_group_law(g, (10 ** len(g) - 1).bit_length() + 1, cond[:, z], units) for z in occupied]
        for g in groups
    ]
    estimates = []
    for group in laws:
        estimates.append(
            [
                [Estimate(*([float(values @ law.probabilities)] * 3)) for values in _tasks(law)]
                for law in group
            ]
        )
    result = _aggregate(portfolio, unit, pz[occupied], laws, estimates)
    for name, exact in exact_targets(portfolio).items():
        assert result[name].value == pytest.approx(exact, rel=1e-12, abs=1e-12)
        assert result[name].low == result[name].high


def test_cdf_bands_enclose_compatible_loss_distribution():
    support = np.array([0, 2, 4])
    masses = _loss_masses(support, [Estimate(0.3, 0.15, 0.45), Estimate(0.6, 0.55, 0.8)])
    assert masses.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.all(masses >= 0)
    threshold = 2
    assert masses[:, threshold + 1 :].sum(axis=1) == pytest.approx([0.4, 0.2, 0.45])


def test_capacity_is_checked_before_execution():
    portfolio = adb_portfolio(2)
    with pytest.raises(ValueError, match="QPU 2 requires 5 qubits"):
        run_distributed_iqae(portfolio, capacities=(6, 4))
    with pytest.raises(ValueError, match="exactly once"):
        run_distributed_iqae(portfolio, groups=((0,), (0, 1)))


def test_distributed_iqae_runs_independent_five_qubit_circuits():
    portfolio = adb_portfolio(2, "adb-independent")
    result = run_distributed_iqae(
        portfolio, capacities=(5, 6), epsilon=0.05, shots=128, max_rounds=3, seed=12
    )
    assert result.shared_strata == 1
    assert [qpu.qubits for qpu in result.qpus] == [5, 5]
    for qpu in result.qpus:
        assert qpu.shots > 0
        assert qpu.forward_preparations == qpu.inverse_preparations + qpu.shots
    for name, exact in exact_targets(portfolio).items():
        interval = result.targets[name]
        assert interval.low <= exact <= interval.high
