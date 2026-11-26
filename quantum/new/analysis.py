"""Read the benchmark CSVs (quantum/new/results) and derive the summary numbers.

Shared by benchmark.py (BENCHMARK.md) and report.py (figures and PDF).
"""
from __future__ import annotations

import csv
from functools import cache
from pathlib import Path

import numpy as np

from .portfolio import PROBABILITY_TARGETS, TARGETS

RESULTS = Path(__file__).resolve().parent / "results"
CLASSICAL = ("monte_carlo", "sobol", "stratified", "importance")
METHOD_LABEL = {"monte_carlo": "Monte Carlo", "sobol": "Scrambled Sobol", "stratified": "Stratified",
                "importance": "Importance sampling", "iqae": "IQAE"}


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v != "" else float("nan")
            except (TypeError, ValueError):
                pass
    return rows


def _order(r: dict) -> tuple:
    """Stable row order: study, n, target (in TARGETS order), method, then the sweep variable."""
    t = r.get("target")
    return (str(r.get("study", "")), r.get("n", 0), TARGETS.index(t) if t in TARGETS else -1,
            str(r.get("method", "")), r.get("p2q", 0), r.get("paths", 0), -r.get("eps_rel", 0),
            r.get("shots", 0), -r.get("alpha", 0), r.get("k", 0))


def write(rows: list[dict], name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    rows = sorted(rows, key=_order)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


@cache
def table(name: str) -> tuple[dict, ...]:
    return tuple(read(name))


def clear_cache() -> None:
    table.cache_clear()


def exact(n: int, target: str) -> float:
    return next(r["exact"] for r in table("exact.csv") if r["n"] == n and r["target"] == target)


def curve(n: int, target: str, method: str, **filt) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(cost, relative RMSE, rows) sorted by cost; cost = paths (classical) or oracle calls."""
    name = "iqae.csv" if method == "iqae" else "classical.csv"
    rr = [r for r in table(name) if r["n"] == n and r["target"] == target
          and r.get("method", "iqae") == method and all(r.get(k) == v for k, v in filt.items())]
    key = "queries" if method == "iqae" else "paths"
    rr.sort(key=lambda r: r[key])
    return np.array([r[key] for r in rr], float), np.array([r["rel_rmse"] for r in rr], float), rr


def monotone(y: np.ndarray) -> np.ndarray:
    """Non-increasing least-squares fit of log(y) (pool adjacent violators), back in y units.

    Expected error falls with budget; the measured curve does not always, because a rare
    heavy-tailed run (an IQAE interval miss, a lucky tail hit) can dominate one replicate set.
    """
    blocks: list[list[float]] = []                      # [mean, weight]
    for v in np.log(np.maximum(y, 1e-300)):
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, w2 = blocks.pop()
            m1, w1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return np.exp(np.concatenate([[m] * int(w) for m, w in blocks]))


def crossing(q: np.ndarray, y: np.ndarray, level: float, extrapolate: bool = False):
    """Cost at which the (monotone-fitted) error curve reaches `level`: (value, flag, index).

    flag "" = interpolated, "<=" = already met at the smallest budget, "~" = extrapolated
    along 1/sqrt(N) from the largest budget, ">" = not reached.
    """
    if len(q) == 0:
        return np.nan, ">", None
    y = monotone(y)
    ok = y <= level
    if ok.all():
        return float(q[0]), "<=", 0
    i = int(np.flatnonzero(~ok)[-1]) + 1
    if i >= len(q):
        if extrapolate and np.isfinite(y[-1]) and y[-1] > 0:
            return float(q[-1] * (y[-1] / level) ** 2), "~", None
        return np.nan, ">", None
    x0, x1 = np.log(q[i - 1]), np.log(q[i])
    y0, y1 = np.log(max(y[i - 1], 1e-300)), np.log(max(y[i], 1e-300))
    if y1 == y0:
        return float(q[i]), "", i
    return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0))), "", i


def cost_to(level: float, n: int, target: str, method: str):
    """(cost, flag, k_max) to reach relative RMSE `level`.

    Classical curves are extrapolated along 1/sqrt(N) when they are reliable at the largest
    budget (fewer than 5% all-zero runs); Monte Carlo on a probability target that never saw
    a hit falls back to the exact binomial formula (1 - p) / (p level^2), flag "theory".
    """
    q, y, rr = curve(n, target, method)
    if method == "iqae":
        v, flag, i = crossing(q, y, level)
        return v, flag, (rr[i]["k_max_median"] if i is not None else np.nan)
    reliable = bool(rr) and rr[-1]["frac_zero"] < 0.05
    v, flag, _ = crossing(q, y, level, extrapolate=reliable)
    if not np.isfinite(v) and method == "monte_carlo" and target in PROBABILITY_TARGETS:
        p = exact(n, target)
        return (1 - p) / (p * level**2), "theory", 0.0
    return v, flag, 0.0


def sweep_point(study: str, key: str, value: float, n: int, target: str, level: float):
    """(oracle calls, flag, k_max) at relative RMSE `level` for one setting of a trade-off study."""
    rr = sorted((r for r in table("tradeoff.csv") if r["study"] == study and r[key] == value
                 and r["n"] == n and r["target"] == target), key=lambda r: r["queries"])
    q = np.array([r["queries"] for r in rr], float)
    y = np.array([r["rel_rmse"] for r in rr], float)
    v, flag, i = crossing(q, y, level)
    return v, flag, (rr[i]["k_max_median"] if i is not None else np.nan)


def best_classical(level: float, n: int, target: str):
    """(cost, method, flag) of the classical sampler needing the fewest paths."""
    best = (np.inf, "", "")
    for m in CLASSICAL:
        v, flag, _ = cost_to(level, n, target, m)
        if np.isfinite(v) and v < best[0]:
            best = (v, m, flag)
    return best


def advantage(level: float, n: int, target: str):
    """(best classical paths / IQAE queries, best method, bound marker, classical, iqae)."""
    cv, meth, cf = best_classical(level, n, target)
    qv, qf, _ = cost_to(level, n, target, "iqae")
    bound = ""
    if cf == "<=" and qf != "<=":
        bound = "≤"
    elif qf == "<=" and cf != "<=":
        bound = "≥"
    return cv / qv, meth, bound, cv, qv


def counts(n: int, target: str) -> dict:
    return next(r for r in table("counts.csv") if r["n"] == n and r["target"] == target)


def ns() -> list[int]:
    return sorted({int(r["n"]) for r in table("exact.csv")})


def fmt_sci(v: float, digits: int = 1) -> str:
    if not np.isfinite(v):
        return "n/a"
    s = f"{v:.{digits}e}"
    m, e = s.split("e")
    return f"{m}e{int(e)}"


__all__ = ["CLASSICAL", "METHOD_LABEL", "RESULTS", "TARGETS", "advantage", "best_classical",
           "clear_cache", "cost_to", "counts", "crossing", "curve", "exact", "fmt_sci", "ns",
           "read", "table", "write"]
