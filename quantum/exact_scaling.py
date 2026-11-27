"""Exact (shock-factorised) solver scaling and the cost of P(all exhausted) by method.

python -m quantum.exact_scaling   ->  results/exact_scaling.csv,
                                          results/exhausted_cost.csv,
                                          results/figures/f10_exact_scaling.png,
                                          results/figures/f11_exhausted_cost.png

Part 1 times exact_targets (all five targets) from 2 to 1000 countries and the brute-force
joint chain where it is feasible (n <= 5).
Part 2 compares, for P(all exhausted) at a 95% CI half-width of 1% of the value:
    exact        one evaluation of the factorised formula (measured wall-clock time)
    Monte Carlo  paths N = (1 - p) / (p (eps/1.96)^2); time = N x measured cost per path
    IQAE         oracle queries from the ideal-device IQAE run; time = sum over rounds of
                 shots x circuit depth (A then Q^k) x LAYER_S, i.e. an idealised
                 fault-tolerant device that executes one circuit layer per LAYER_S seconds.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
import numpy as np

from .baselines import monte_carlo
from .circuits import gate_counts, grover_operator, oracle
from .iqae import GroverModel, iqae
from .portfolio import PortfolioModel, adb_portfolio, brute_force, exact_targets

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"
EPS, ALPHA, SHOTS, LAYER_S, Z95 = 0.01, 0.05, 100, 1e-6, 1.959964
N_SCALING = (2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
N_COST = (2, 3, 5, 10, 15, 20)
BLUE, ORANGE, AQUA, INK, INK2 = "#257de0", "#f47230", "#00b894", "#191919", "#666666"


def _time(f, reps: int) -> float:
    f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def scaling() -> list[dict]:
    rows = []
    for n in N_SCALING:
        pf = adb_portfolio(n)
        t_ex = _time(lambda pf=pf: exact_targets(pf), 20 if n <= 100 else 3)
        t_bf = _time(lambda pf=pf: brute_force(pf), 1) if n <= 5 else np.nan
        rows.append(dict(n=n, log10_joint_states=n, t_exact_s=t_ex, t_brute_s=t_bf))
        print(f"  n={n:5d}  exact {t_ex * 1e3:9.2f} ms   brute {t_bf:9.3g} s", flush=True)
    return rows


def exhausted_cost() -> list[dict]:
    rows = []
    for n in N_COST:
        pf = adb_portfolio(n)
        m = PortfolioModel(pf)
        p = exact_targets(pf)["all_exhausted"]
        t_ex = _time(lambda pf=pf: exact_targets(pf)["all_exhausted"], 20)
        n_paths = 1_000_000
        per_path = _time(lambda m=m: monte_carlo(m, n_paths, np.random.default_rng(0)), 1) / n_paths
        mc_paths = (1 - p) / (p * (EPS / Z95) ** 2)
        r = iqae(GroverModel(p), EPS * p, alpha=ALPHA, shots=SHOTS, rng=0)
        A = oracle(m, "all_exhausted", "scalable")
        ga, gq = gate_counts(A), gate_counts(grover_operator(A))
        q_time = sum(SHOTS * (ga["depth"] + k * gq["depth"]) for k in r.powers) * LAYER_S
        rows.append(dict(n=n, p=p, t_exact_s=t_ex, mc_paths=mc_paths, mc_time_s=mc_paths * per_path,
                         iqae_queries=r.queries, iqae_k_max=r.k_max, iqae_time_s=q_time,
                         deepest_cx=ga["cx"] + r.k_max * gq["cx"],
                         iqae_const=r.queries * EPS * np.sqrt(p)))
        print(f"  n={n:3d}  p={p:.3g}  MC {mc_paths:.3g} paths  IQAE {r.queries:.3g} queries", flush=True)
    return rows


def _style(ax, title, ylabel, xlabel="countries"):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _line(ax, x, y, c, label, short, ls="-"):
    ax.plot(x, y, ls, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5, label=label)
    ax.annotate(short, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                color=INK, fontsize=9)


def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def _write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figures(sc: list[dict], ec: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    # f10: exact solver scaling
    fig, ax = plt.subplots(figsize=(7, 4.6))
    n, t_ex, t_bf = _col(sc, "n"), _col(sc, "t_exact_s"), _col(sc, "t_brute_s")
    _line(ax, n, t_ex, AQUA, "factorised exact solver (measured)", "factorised")
    ok = np.isfinite(t_bf)
    _line(ax, n[ok], t_bf[ok], ORANGE,
          "brute-force joint chain, 5^n 2^n states (measured)", "brute force")
    ref = t_ex[-1] * (n / n[-1]) ** 2
    ax.plot(n[n >= 50], ref[n >= 50], ":", color=INK2, lw=1.2, label="n^2 reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 2500)
    _style(ax, "Exact computation of all five targets: wall-clock time", "seconds")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "f10_exact_scaling.png", dpi=200)
    plt.close(fig)

    # f11: P(all exhausted), queries and time
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    n = _col(ec, "n")
    _line(a1, n, _col(ec, "mc_paths"), ORANGE, "Monte Carlo paths", "Monte Carlo")
    _line(a1, n, _col(ec, "iqae_queries"), BLUE, "IQAE oracle queries", "IQAE")
    _line(a1, n, np.ones_like(n), AQUA, "Exact: 1 formula evaluation", "Exact")
    _style(a1, "Samples / queries to reach ±1% (95% CI)", "samples / queries")
    _line(a2, n, _col(ec, "mc_time_s"), ORANGE, "Monte Carlo (measured rate, laptop)", "Monte Carlo")
    _line(a2, n, _col(ec, "iqae_time_s"), BLUE, "IQAE (ideal FT device, 1 µs/layer)", "IQAE", "--")
    _line(a2, n, _col(ec, "t_exact_s"), AQUA, "Exact (measured)", "Exact")
    _style(a2, "Wall-clock time to reach ±1% (95% CI)", "seconds")
    for yv, lab in ((1, "1 s"), (3600, "1 hour"), (86400 * 365, "1 year")):
        a2.axhline(yv, color=INK2, lw=0.6, ls=":")
        a2.text(26.5, yv * 1.4, lab, color=INK2, fontsize=8, ha="right")
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.set_xticks(N_COST)
        ax.set_xlim(1, 27)
    a1.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1, 0.3))
    a2.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1, 0.43))
    fig.tight_layout()
    fig.savefig(FIG / "f11_exhausted_cost.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print("exact solver scaling")
    sc = scaling()
    _write(sc, RES / "exact_scaling.csv")
    print("P(all exhausted): exact vs Monte Carlo vs IQAE")
    ec = exhausted_cost()
    _write(ec, RES / "exhausted_cost.csv")
    figures(sc, ec)
    print(f"wrote {RES / 'exact_scaling.csv'}, {RES / 'exhausted_cost.csv'}, f10 and f11 figures")


if __name__ == "__main__":
    main()
