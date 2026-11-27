"""Figures for the IQAE report (static PNGs for print, light theme).

python -m quantum.figures   ->  results/figures/*.png

Colour follows the entity: IQAE and each classical sampler keep one categorical slot (validated
palette, adjacent CVD dE >= 11.8), every series also has its own marker, and portfolio size uses
one ordinal blue ramp. Gridlines are solid hairlines.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import analysis as an
from .portfolio import LABELS, TARGETS, adb_portfolio, loss_distribution
from .resources import ft_estimate

FIG = an.RESULTS / "figures"
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
STYLE = {   # slot order of the validated categorical palette
    "iqae": dict(color="#257de0", marker="o", label="IQAE"),
    "monte_carlo": dict(color="#f47230", marker="s", label="Monte Carlo"),
    "sobol": dict(color="#00b894", marker="^", label="Scrambled Sobol"),
    "stratified": dict(color="#f2a600", marker="D", label="Stratified (shock state)"),
    "importance": dict(color="#ee7fac", marker="v", label="Importance sampling"),
}
N_COLOR = {2: "#84b9f3", 3: "#338ced", 5: "#1b64be", 10: "#0e3d7e"}   # ordinal blue ramp
N_MARKER = {2: "o", 3: "s", 5: "^", 10: "D"}
LEVEL_STYLE = {0.10: dict(color="#257de0", marker="o"), 0.01: dict(color="#f47230", marker="s")}
SHORT = {t: LABELS[t].replace("-", "−") for t in TARGETS}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 8.8, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.3,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 1.7,
        "lines.markersize": 4.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def _plot(ax, x, y, color, marker, label=None, ls="-", **kw):
    ax.plot(x, y, color=color, marker=marker, ls=ls, label=label, markeredgecolor=SURFACE,
            markeredgewidth=0.7, **kw)


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. the problem

def fig_exceedance(ns) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for n in ns:
        pf = adb_portfolio(n)
        L, pmf = loss_distribution(pf)
        surv = pmf[::-1].cumsum()[::-1]                     # P(L >= x_j)
        frac = L / pf.total_principal * 100
        ax.step(frac[:-1], surv[1:], where="post", color=N_COLOR[n], lw=1.7, label=f"{n} countries")
        ax.plot([100], [pmf[-1]], N_MARKER[n], color=N_COLOR[n], ms=6, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=5)
        ax.text(102, pmf[-1], f"{pmf[-1]:.1e}".replace("e-0", "e-"), fontsize=6.8, color=INK2, va="center")
    ax.axvline(50, color=INK2, lw=0.8)
    ax.text(51, 0.3, "T = half the\ntotal principal", fontsize=6.8, color=INK2, va="top")
    ax.text(100, 2e-3, "all bonds\nexhausted", fontsize=6.8, color=INK2, ha="center")
    ax.set(yscale="log", ylim=(4e-9, 1.2), xlim=(0, 116), xticks=[0, 25, 50, 75, 100],
           xticklabels=["0%", "25%", "50%", "75%", "100%"],
           xlabel="portfolio loss x, as a share of the total principal", ylabel="P(L > x)")
    ax.legend(loc="lower left", ncol=2)
    _save(fig, "f1_exceedance.png")


# ---------------------------------------------------------------- 2. benchmark

def fig_benchmark(ns) -> None:
    rows = ("expected_loss", "exceedance", "all_exhausted")
    fig, axes = plt.subplots(len(rows), len(ns), figsize=(7.2, 6.0), sharex=True, sharey="row")
    for i, t in enumerate(rows):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            for lvl in (0.1, 0.01):
                ax.axhline(lvl, color=AXIS, lw=0.8, zorder=1)
            for meth in (*an.CLASSICAL, "iqae"):
                q, y, _ = an.curve(n, t, meth)
                keep = (q >= 5e2) & (q <= 5e7)
                st = STYLE[meth]
                _plot(ax, q[keep], np.minimum(y[keep], 3.0), st["color"], st["marker"],
                      st["label"], lw=1.3, ms=3.6, zorder=4 if meth == "iqae" else 3)
            ax.set(xscale="log", yscale="log", xlim=(5e2, 5e7))
            if i == 0:
                ax.set_title(f"{n} countries")
            if j == 0:
                ax.set_ylabel(f"{SHORT[t]}\nrelative RMSE")
            if i == len(rows) - 1:
                ax.set_xlabel("paths / oracle calls")
            if j == len(ns) - 1:
                for lvl, txt in ((0.1, "10%"), (0.01, "1%")):
                    ax.text(1.03, lvl, txt, transform=ax.get_yaxis_transform(), fontsize=6.6,
                            color=INK2, va="center")
        axes[i, 0].set_ylim({"expected_loss": (3e-4, 0.5), "exceedance": (1e-3, 1.5),
                             "all_exhausted": (2e-3, 3.2)}[t])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.8, w_pad=0.5)
    _save(fig, "f2_benchmark.png")


def fig_advantage(ns) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#c22f30", "#ec4647", "#f5a3a3", "#efefef", "#9dc8f7", "#338ced", "#1757ac"])
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    short = {"monte_carlo": "MC", "sobol": "Sobol", "stratified": "Strat.", "importance": "IS"}
    for ax, level in zip(axes, (0.10, 0.01), strict=True):
        M = np.zeros((len(TARGETS), len(ns)))
        for i, t in enumerate(TARGETS):
            for j, n in enumerate(ns):
                r, meth, bound, _, _ = an.advantage(level, n, t)
                M[i, j] = np.log10(r)
                col = SURFACE if abs(M[i, j]) > 1.6 else INK
                ax.text(j, i - 0.13, f"{bound}{r:.0f}×" if r >= 9.5 else f"{bound}{r:.1f}×",
                        ha="center", va="center", fontsize=7.4, color=col, weight="bold")
                ax.text(j, i + 0.25, f"vs {short.get(meth, meth)}", ha="center", va="center",
                        fontsize=5.8, color=col)
        ax.imshow(np.clip(M, -2.5, 2.5), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(TARGETS)), [SHORT[t] for t in TARGETS] if level == 0.10 else [""] * 5)
        ax.set_xlabel("countries")
        ax.set_title(f"{level:.0%} relative RMSE")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(ns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(TARGETS)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", length=0)
    fig.text(0.5, -0.03, "Cell = paths the best classical sampler needs ÷ oracle calls IQAE needs.  "
             "Blue: IQAE needs fewer.  Red: a classical sampler needs fewer.", ha="center",
             fontsize=6.8, color=INK2)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "f3_advantage.png")


# ---------------------------------------------------------------- 3. limiting factors

def scaling_points(ns, level: float) -> np.ndarray:
    """Rows (amplitude, IQAE queries, IQAE k_max, best classical paths, n, target index)."""
    pts = []
    for n in ns:
        for k, t in enumerate(TARGETS):
            q, flag, km = an.cost_to(level, n, t, "iqae")
            if not np.isfinite(q) or flag == "<=":
                continue
            a = next(r["amplitude"] for r in an.table("exact.csv") if r["n"] == n and r["target"] == t)
            pts.append((a, q, km, an.best_classical(level, n, t)[0], n, k))
    return np.array(pts)


def fig_rarity(ns) -> dict:
    """Queries and deepest Grover power vs the amplitude being estimated; returns fitted slopes."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    slopes = {}
    aa = np.geomspace(5e-9, 0.2, 50)
    for level in (0.10, 0.01):
        P = scaling_points(ns, level)
        st = LEVEL_STYLE[level]
        s_q, _ = np.polyfit(np.log(P[:, 0]), np.log(P[:, 1]), 1)
        s_k, c_k = np.polyfit(np.log(P[:, 0]), np.log(P[:, 2]), 1)
        slopes[level] = (s_q, s_k)
        axes[0].plot(P[:, 0], P[:, 1], st["marker"], color=st["color"], markeredgecolor=SURFACE,
                     markeredgewidth=0.7, label=f"IQAE, {level:.0%} error (slope {s_q:.2f})", zorder=4)
        axes[0].plot(aa, (1 - aa) / (aa * level**2), color=st["color"], lw=1.0, ls="--",
                     label=f"Monte Carlo, {level:.0%} error (slope −1)")
        axes[1].plot(P[:, 0], P[:, 2], st["marker"], color=st["color"], markeredgecolor=SURFACE,
                     markeredgewidth=0.7, label=f"{level:.0%} error (slope {s_k:.2f})", zorder=4)
        axes[1].plot(aa, np.exp(c_k) * aa**s_k, color=st["color"], lw=1.0)
    axes[0].set(xscale="log", yscale="log", xlabel="amplitude a = target / f_max",
                ylabel="oracle calls (IQAE) or paths (MC)", title="Cost grows as a^(−1/2), not a^(−1)")
    axes[1].set(xscale="log", yscale="log", xlabel="amplitude a = target / f_max",
                ylabel="deepest Grover power k_max", title="…and so does circuit depth")
    axes[0].legend(fontsize=6.3, loc="upper right")
    axes[1].legend(fontsize=6.6, loc="upper right")
    fig.tight_layout(w_pad=1.6)
    _save(fig, "f4_rarity.png")
    return slopes


def depth_rows(ns) -> list[dict]:
    out = []
    for t, level in (("exceedance", 0.10), ("exceedance", 0.01), ("all_exhausted", 0.10),
                     ("all_exhausted", 0.01)):
        for n in ns:
            q, _, km = an.cost_to(level, n, t, "iqae")
            if not np.isfinite(q):
                continue
            c = an.counts(n, t)
            cxd = c["cx_A"] + km * c["cx_Q"]
            out.append(dict(n=n, target=t, level=level, queries=q, k_max=km, cx_A=c["cx_A"],
                            cx_Q=c["cx_Q"], qubits_A=c["qubits_A"], qubits_Q=c["qubits_Q"],
                            cx_deepest=cxd, p2q_needed=0.105 / cxd))
    return out


def fig_depth(ns) -> None:
    rows = depth_rows(ns)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    series = ((("exceedance", 0.10), "#257de0", "o", "-"), (("exceedance", 0.01), "#257de0", "o", "--"),
              (("all_exhausted", 0.10), "#f47230", "s", "-"), (("all_exhausted", 0.01), "#f47230", "s", "--"))
    for (t, lvl), col, mk, ls in series:
        rr = [r for r in rows if r["target"] == t and r["level"] == lvl]
        if not rr:
            continue
        lab = f"{SHORT[t]}, {lvl:.0%} error"
        face = SURFACE if lvl == 0.01 else col               # 1% error: hollow markers
        for ax, key in ((axes[0], "cx_deepest"), (axes[1], "p2q_needed")):
            ax.plot([r["n"] for r in rr], [r[key] for r in rr], color=col, marker=mk, ls=ls, label=lab,
                    markerfacecolor=face, markeredgecolor=col if lvl == 0.01 else SURFACE,
                    markeredgewidth=1.1 if lvl == 0.01 else 0.7)
    axes[0].set(yscale="log", ylabel="two-qubit gates", title="Deepest single circuit (k_max Grover steps)")
    axes[1].axhspan(1e-4, 1e-2, color="#f2f2f2", zorder=0, lw=0)
    axes[1].text(10.3, 1.5e-3, "typical two-qubit\nerror today", fontsize=6.4, color=INK2, va="center",
                 ha="right")
    axes[1].set(yscale="log", ylim=(1e-10, 1e-1), ylabel="two-qubit error rate p",
                title="Error rate for 90% fidelity of that circuit")
    for ax in axes:
        ax.set_xticks(ns)
        ax.set_xlabel("countries")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=1.6, rect=(0, 0.05, 1, 1))
    _save(fig, "f5_depth.png")


def fig_noise(ns) -> None:
    rows = an.table("noise.csv")
    cfg = (("exceedance", 0.01), ("all_exhausted", 0.10))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    xs = sorted({r["p2q"] for r in rows})
    xpos = {p: (1e-8 if p == 0 else p) for p in xs}
    for j, (t, lvl) in enumerate(cfg):
        for n in ns:
            rr = sorted((r for r in rows if r["n"] == n and r["target"] == t and r["eps_rel"] == lvl),
                        key=lambda r: r["p2q"])
            if not rr:
                continue
            x = [xpos[r["p2q"]] for r in rr]
            _plot(axes[0, j], x, [max(r["rel_rmse"], 1e-4) for r in rr], N_COLOR[n], N_MARKER[n],
                  f"{n} countries")
            _plot(axes[1, j], x, [r["ci_coverage"] for r in rr], N_COLOR[n], N_MARKER[n], f"{n} countries")
        axes[0, j].set(yscale="log", ylim=(5e-4, 2e3), title=f"{SHORT[t]}, {lvl:.0%} target error")
        axes[0, j].axhline(lvl, color=AXIS, lw=0.8)
        axes[1, j].set(ylim=(-0.03, 1.05), xlabel="two-qubit gate error p (λ = p × CX per oracle call)")
        axes[1, j].set_xscale("log")
        axes[1, j].minorticks_off()
        axes[1, j].set_xticks([xpos[p] for p in xs], ["0" if p == 0 else f"1e{int(np.log10(p))}" for p in xs])
    axes[0, 0].set_ylabel("relative RMSE")
    axes[1, 0].set_ylabel("CI coverage (nominal 99.99%)")
    axes[0, 0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(h_pad=0.8, w_pad=1.4)
    _save(fig, "f6_noise.png")


def fig_noisy_circuit() -> None:
    rows = an.table("noisy_circuit.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ks = sorted({r["k"] for r in rows})
    kk = np.linspace(0, max(ks), 200)
    a = next(r["amplitude"] for r in an.table("exact.csv") if r["n"] == 2 and r["target"] == "exceedance")
    ax.plot(kk, np.sin((2 * kk + 1) * np.arcsin(np.sqrt(a))) ** 2, color=INK2, lw=1.1, label="ideal circuit")
    for p, col, mk in ((1e-4, "#338ced", "o"), (1e-3, "#0e3d7e", "s")):
        rr = sorted((r for r in rows if r["p2q"] == p), key=lambda r: r["k"])
        _plot(ax, [r["k"] for r in rr], [r["measured"] for r in rr], col, mk, f"Aer, p = {p:g} (measured)",
              ls="none")
        ax.plot([r["k"] for r in rr], [r["depolarising_model"] for r in rr], color=col, lw=0.9, ls="--",
                label=f"global depolarising model, p = {p:g}")
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set(xlabel="Grover power k", ylabel="P(objective = 1)", xticks=ks, ylim=(0, 1.0))
    ax.legend(fontsize=6.2, loc="upper left")
    _save(fig, "f7_noisy_circuit.png")


def fig_tradeoff() -> None:
    pts = tradeoff_points()
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for (n, t, lvl), col, mk in (((5, "exceedance", 0.01), "#257de0", "o"),
                                 ((10, "all_exhausted", 0.10), "#f47230", "s")):
        rr = pts.get((n, t, lvl), [])
        if not rr:
            continue
        lab = f"{n} countries, {SHORT[t]}, {lvl:.0%} RMSE"
        _plot(ax, [r[1] for r in rr], [r[2] for r in rr], col, mk, lab, lw=0.8)
        for r in (rr[0], rr[-1]):
            ax.annotate(f"{r[0]:,.0f} shots/round", (r[1], r[2]), xytext=(5, 2), textcoords="offset points",
                        fontsize=6.2, color=INK2)
        c0 = np.exp(np.mean([np.log(r[1] * r[2]) for r in rr]))   # calls x k_max = const guide
        x = np.geomspace(min(r[1] for r in rr) / 1.5, max(r[1] for r in rr) * 1.5, 20)
        ax.plot(x, c0 / x, color=col, lw=0.8, ls="--", alpha=0.7)
    ax.plot([], [], color=INK2, lw=0.8, ls="--", label="k_max × calls = constant")
    ax.set(xscale="log", yscale="log", xlabel="oracle calls to reach the RMSE",
           ylabel="deepest Grover power k_max")
    ax.legend(fontsize=6.4, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    _save(fig, "f8_tradeoff.png")


def tradeoff_points() -> dict:
    """{(n, target, level): [(shots per round, calls, k_max), ...]} at equal achieved RMSE."""
    rows = [r for r in an.table("tradeoff.csv") if r["study"] == "shots"]
    out = {}
    for n, t, lvl in ((5, "exceedance", 0.01), (10, "all_exhausted", 0.10)):
        pts = []
        for s in sorted({r["shots"] for r in rows if r["n"] == n and r["target"] == t}):
            q, _, km = an.sweep_point("shots", "shots", s, n, t, lvl)
            if np.isfinite(q):
                pts.append((s, q, km))
        out[(n, t, lvl)] = pts
    return out


def resource_rows(ns, target: str = "exceedance", level: float = 0.01) -> list[dict]:
    out = []
    for n in ns:
        c = an.counts(n, target)
        q, _, km = an.cost_to(level, n, target, "iqae")
        if not np.isfinite(q):
            continue
        e = ft_estimate(int(c["qubits_Q"]), int(c["t_A"]), int(c["rot_A"]), int(c["t_Q"]), int(c["rot_Q"]),
                        q, km)
        cv, meth, cflag = an.best_classical(level, n, target)
        spp = float(np.median([r["seconds_per_path"] for r in an.table("classical.csv")
                               if r["n"] == n and r["method"] == meth]))
        t_ex = next(r["t_exact_s"] for r in an.table("exact.csv") if r["n"] == n)
        out.append(dict(n=n, target=target, level=level, queries=q, k_max=km, cx_A=c["cx_A"], cx_Q=c["cx_Q"],
                        qubits_A=c["qubits_A"], cx_deepest=c["cx_A"] + km * c["cx_Q"],
                        best_classical=meth, classical_paths=cv, classical_flag=cflag,
                        classical_seconds=cv * spp, exact_seconds=t_ex, **e))
    return out


def fig_time(ns) -> None:
    rr = resource_rows(ns)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    x = [r["n"] for r in rr]
    _plot(ax, x, [r["runtime_s_at_10000Hz"] for r in rr], "#257de0", "o",
          "IQAE, fault-tolerant, 10 kHz logical T rate", ls="--")
    _plot(ax, x, [r["runtime_s_at_1000000Hz"] for r in rr], "#257de0", "o",
          "IQAE, fault-tolerant, 1 MHz logical T rate")
    _plot(ax, x, [r["classical_seconds"] for r in rr], "#f2a600", "D", "best classical sampler, laptop")
    _plot(ax, x, [r["exact_seconds"] for r in rr], INK2, "s", "exact factorised solver, laptop")
    for s, lab in ((1, "1 s"), (3600, "1 h"), (86400, "1 day")):
        ax.axhline(s, color=GRID, lw=0.9, zorder=0)
        ax.text(max(x) + 0.3, s, lab, fontsize=6.3, color=INK2, va="center")
    ax.set(yscale="log", ylim=(1e-4, 1e7), xticks=ns, xlabel="countries",
           ylabel="seconds to P(L > T) at 1% error")
    ax.legend(fontsize=6.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    _save(fig, "f9_time.png")


def main() -> dict:
    setup()
    an.clear_cache()
    ns = an.ns()
    fig_exceedance(ns)
    fig_benchmark(ns)
    fig_advantage(ns)
    slopes = fig_rarity(ns)
    fig_depth(ns)
    fig_noise(ns)
    fig_noisy_circuit()
    fig_tradeoff()
    fig_time(ns)
    print("figures written to", FIG)
    return slopes


if __name__ == "__main__":
    main()
