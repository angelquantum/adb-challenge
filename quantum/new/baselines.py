"""Classical sampling baselines on n countries (generalise classical/sampling.py).

A path is an exact draw of the term law: the shock state z = (K, J), then one outcome x_i per
country from cond[i, z]. Cost O(n) per path.

    monte_carlo  inverse CDF with i.i.d. uniforms
    sobol        same map with scrambled Sobol points in n + 1 dimensions, 8 scrambles,
                 t-interval on 7 degrees of freedom
    stratified   strata = the 10 shock states, equal allocation, exact weights P(z). This is
                 the n-country analogue of the brief's "first payout year" strata and the
                 strongest classical method on joint tails here, because it removes all of
                 the cross-country dependence
    importance   50/50 mixture of the true law and its loss-tilted version, applied to the
                 shock state and to every country (the brief's tilt, per component)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
from scipy.stats import t as student_t

from classical import Estimate

from .portfolio import TARGETS, Portfolio, PortfolioModel

SOBOL_SCRAMBLES = 8


@dataclass(frozen=True)
class Batch:
    """Estimates of all five targets from one run: mean, standard error, t dof (inf = normal)."""
    mean: np.ndarray
    stderr: np.ndarray
    dof: float = np.inf


def _cdf(p: np.ndarray) -> np.ndarray:
    c = np.cumsum(p, axis=-1)
    c[..., -1] = 1.0
    return c


def _draw_z(zcdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.minimum(np.searchsorted(zcdf, u, side="right"), len(zcdf) - 1)


def _draw_x(ccdf: np.ndarray, z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse CDF of cond[i, z] per country. ccdf: (n, 10, 10); z: (N,); u: (N, n)."""
    N, n = u.shape
    out = np.empty((N, n), dtype=np.int8)
    step = max(1, 4_000_000 // (n * 10))
    idx = np.arange(n)[None, :]
    for s in range(0, N, step):
        cdf = ccdf[idx, z[s:s + step, None], :]
        out[s:s + step] = np.minimum((u[s:s + step, :, None] >= cdf).sum(-1), 9)
    return out


def _mean_se(F: np.ndarray, w: np.ndarray | None = None) -> Batch:
    G = F if w is None else F * w[:, None]
    return Batch(G.mean(0), G.std(0, ddof=1) / np.sqrt(len(G)))


def monte_carlo(m: PortfolioModel, N: int, rng: np.random.Generator) -> Batch:
    u = rng.random((N, m.n + 1))
    z = _draw_z(_cdf(m.pz), u[:, 0])
    return _mean_se(m.values(_draw_x(_cdf(m.cond), z, u[:, 1:])))


def sobol(m: PortfolioModel, N: int, rng: np.random.Generator) -> Batch:
    per = N // SOBOL_SCRAMBLES
    lg = int(np.log2(per))
    zcdf, ccdf = _cdf(m.pz), _cdf(m.cond)
    reps = []
    for _ in range(SOBOL_SCRAMBLES):
        eng = qmc.Sobol(d=m.n + 1, scramble=True, seed=np.random.default_rng(rng.integers(2**63)))
        u = eng.random_base2(lg) if 2**lg == per else eng.random(per)
        reps.append(m.values(_draw_x(ccdf, _draw_z(zcdf, u[:, 0]), u[:, 1:])).mean(0))
    reps = np.array(reps)
    return Batch(reps.mean(0), reps.std(0, ddof=1) / np.sqrt(SOBOL_SCRAMBLES), SOBOL_SCRAMBLES - 1)


def stratified(m: PortfolioModel, N: int, rng: np.random.Generator) -> Batch:
    zs = np.flatnonzero(m.pz > 0)
    per = np.full(len(zs), N // len(zs))
    per[: N - per.sum()] += 1
    ccdf = _cdf(m.cond)
    mean, var = np.zeros(5), np.zeros(5)
    for z0, nz in zip(zs, per, strict=True):
        F = m.values(_draw_x(ccdf, np.full(nz, z0), rng.random((nz, m.n))))
        mean += m.pz[z0] * F.mean(0)
        var += m.pz[z0] ** 2 * F.var(0, ddof=1) / nz
    return Batch(mean, np.sqrt(var))


def _tilted(p: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """0.5 p + 0.5 p * loss / E_p[loss] (falls back to p where E_p[loss] = 0)."""
    el = (p * loss).sum(-1, keepdims=True)
    tilt = np.divide(p * loss, el, out=p.copy(), where=el > 0)
    return 0.5 * p + 0.5 * tilt


def importance(m: PortfolioModel, N: int, rng: np.random.Generator) -> Batch:
    el_z = (m.cond * m.loss[:, None, :]).sum(-1).sum(0)          # E[L | z]
    qz = _tilted(m.pz, el_z)
    qx = _tilted(m.cond, m.loss[:, None, :])
    rz = np.divide(m.pz, qz, out=np.zeros_like(qz), where=qz > 0)
    rx = np.divide(m.cond, qx, out=np.zeros_like(qx), where=qx > 0)
    u = rng.random((N, m.n + 1))
    z = _draw_z(_cdf(qz), u[:, 0])
    x = _draw_x(_cdf(qx), z, u[:, 1:])
    w = rz[z] * np.prod(rx[np.arange(m.n)[None, :], z[:, None], x], axis=1)
    return _mean_se(m.values(x), w)


METHODS: dict[str, Callable[[PortfolioModel, int, np.random.Generator], Batch]] = {
    "monte_carlo": monte_carlo,
    "sobol": sobol,
    "stratified": stratified,
    "importance": importance,
}


def estimate(portfolio: Portfolio | PortfolioModel, method: str, n_paths: int,
             seed: int | None = None) -> dict[str, Estimate]:
    """All five targets with `method` and a budget of `n_paths` term paths (cf. classical.estimate)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    m = portfolio if isinstance(portfolio, PortfolioModel) else PortfolioModel(portfolio)
    b = METHODS[method](m, n_paths, np.random.default_rng(seed))
    q = student_t.ppf(0.975, b.dof) if np.isfinite(b.dof) else 1.96
    return {t: Estimate(float(b.mean[k]), float(b.stderr[k]), float(q * b.stderr[k]))
            for k, t in enumerate(TARGETS)}
