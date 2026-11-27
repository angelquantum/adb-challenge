"""Plot the loss distribution and benchmark results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from quantum.portfolio import LABELS, TARGETS, adb_portfolio, loss_distribution
from quantum.resources import logical_estimate

from . import analysis as an

FIG = an.RESULTS / "figures"
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
STYLE = {
    "iqae": dict(color="#257de0", marker="o", label="IQAE"),
    "monte_carlo": dict(color="#f47230", marker="s", label="Monte Carlo"),
    "sobol": dict(color="#00b894", marker="^", label="Scrambled Sobol"),
    "stratified": dict(color="#f2a600", marker="D", label="Stratified (shock state)"),
    "importance": dict(color="#ee7fac", marker="v", label="Importance sampling"),
}
N_COLOR = {2: "#84b9f3", 3: "#338ced", 5: "#1b64be", 10: "#0e3d7e"}
N_MARKER = {2: "o", 3: "s", 5: "^", 10: "D"}
LEVEL_STYLE = {0.10: dict(color="#257de0", marker="o"), 0.01: dict(color="#f47230", marker="s")}
SHORT = {t: LABELS[t].replace("-", "−") for t in TARGETS}
BENCHMARK_YLIM = {
    "expected_loss": (3e-4, 0.5),
    "all_pay": (1e-7, 1.0),
    "exceedance": (1e-3, 1.5),
    "stop_loss": (1e-5, 1.0),
    "all_exhausted": (2e-3, 3.2),
}


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


def fig_exceedance(ns) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for n in ns:
        pf = adb_portfolio(n)
        L, pmf = loss_distribution(pf)
        survival = pmf[::-1].cumsum()[::-1]
        frac = L / pf.total_principal * 100
        ax.step(frac[:-1], survival[1:], where="post", color=N_COLOR[n], lw=1.7, label=f"{n} countries")
        ax.plot([100], [pmf[-1]], N_MARKER[n], color=N_COLOR[n], ms=6, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=5)
        ax.text(102, pmf[-1], f"{pmf[-1]:.1e}".replace("e-0", "e-"), fontsize=6.8, color=INK2, va="center")
    ax.axvline(50, color=INK2, lw=0.8)
    ax.text(51, 0.3, "T = half the\ntotal principal", fontsize=6.8, color=INK2, va="top")
    ax.text(100, 2e-3, "all bonds\nexhausted", fontsize=6.8, color=INK2, ha="center")
    ax.set(yscale="log", ylim=(4e-9, 1.2), xlim=(0, 116), xticks=[0, 25, 50, 75, 100],
           xticklabels=["0%", "25%", "50%", "75%", "100%"],
           xlabel="portfolio loss x, as a share of the total principal", ylabel="P(L > x)")
    ax.set_title("Probability that portfolio loss exceeds a given level")
    ax.legend(loc="lower left", ncol=2)
    _save(fig, "loss_exceedance.png")


def plot_accuracy_grid(ns, xlabel, xlim, series, name, title, *, note=None, all_pay_ylim=None):
    """Plot one accuracy curve per method, with panels by target and country count."""
    fig, axes = plt.subplots(len(TARGETS), len(ns), figsize=(7.2, 9.6),
                             sharex=True, sharey="row", squeeze=False)
    for i, t in enumerate(TARGETS):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            for lvl in (0.1, 0.01):
                ax.axhline(lvl, color=AXIS, lw=0.8, zorder=1)
            for x, y, st, z in series(n, t):
                keep = (x >= xlim[0]) & (x <= xlim[1])
                _plot(ax, x[keep], np.minimum(y[keep], 3.0), st["color"], st["marker"], st["label"],
                      lw=1.3, ms=3.6, zorder=z)
            ax.set(xscale="log", yscale="log", xlim=xlim)
            if i == 0:
                ax.set_title(f"{n} countries")
            if j == 0:
                ax.set_ylabel(f"{SHORT[t]}\nrelative RMSE")
            if i == len(TARGETS) - 1:
                ax.set_xlabel(xlabel)
            if j == len(ns) - 1:
                for lvl, txt in ((0.1, "10%"), (0.01, "1%")):
                    ax.text(1.03, lvl, txt, transform=ax.get_yaxis_transform(), fontsize=6.6,
                            color=INK2, va="center")
        limits = all_pay_ylim if t == "all_pay" and all_pay_ylim else BENCHMARK_YLIM[t]
        axes[i, 0].set_ylim(limits)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title)
    legend_columns = 3 if note else 5
    legend_top = 0.96 if note else 0.95
    fig.legend(h, lab, loc="upper center", ncol=legend_columns, bbox_to_anchor=(0.5, legend_top))
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=6.6, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.90 if note else 0.92), h_pad=0.8, w_pad=0.5)
    _save(fig, name)


def fig_benchmark(data, ns) -> None:
    def by_calls(n, target):
        for method in (*an.CLASSICAL, "iqae"):
            calls, error, _ = data.curve(n, target, method)
            yield calls, error, STYLE[method], 4 if method == "iqae" else 3

    plot_accuracy_grid(ns, "paths / oracle calls", (5e2, 5e7), by_calls, "iqae_accuracy.png",
                       "IQAE vs. classical sampling accuracy by computational work")


def fig_advantage(data, ns) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#c22f30", "#ec4647", "#f5a3a3", "#efefef", "#9dc8f7", "#338ced", "#1757ac"])
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    cmap.set_bad("#f2f2f2")
    fig = plt.figure(figsize=(8.3, 5.7))
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.06))
    axes = [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)]
    color_axis = fig.add_subplot(grid[:, 2])
    fig.suptitle("Monolithic and distributed IQAE cost vs. classical sampling")
    short = {"monte_carlo": "MC", "sobol": "Sobol", "stratified": "Strat.", "importance": "IS"}
    for row, (method, label) in enumerate((("iqae", "Monolithic"), ("distributed_iqae", "Distributed"))):
        for ax, level in zip(axes[row], (0.10, 0.01), strict=True):
            values = np.full((len(TARGETS), len(ns)), np.nan)
            for i, t in enumerate(TARGETS):
                for j, n in enumerate(ns):
                    r, meth, bound, _, _ = data.advantage(level, n, t, method)
                    if not np.isfinite(r) or r <= 0:
                        ax.text(j, i, "\N{EM DASH}", ha="center", va="center", color=INK2)
                        continue
                    values[i, j] = np.log10(r)
                    col = SURFACE if abs(values[i, j]) > 1.6 else INK
                    ratio = f"{r:.0f}" if r >= 9.5 else f"{r:.2g}"
                    ax.text(j, i - 0.13, f"{bound}{ratio}×", ha="center", va="center",
                            fontsize=7.4, color=col, weight="bold")
                    ax.text(j, i + 0.25, f"vs {short.get(meth, meth)}", ha="center", va="center",
                            fontsize=5.8, color=col)
            heatmap = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(len(ns)), [str(n) for n in ns])
            ax.set_yticks(range(len(TARGETS)), [SHORT[t] if level == 0.10 else "" for t in TARGETS])
            ax.set_xlabel("countries")
            ax.set_title(f"{label} IQAE at {level:.0%} relative RMSE")
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, len(ns)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(TARGETS)), minor=True)
            ax.grid(which="minor", color=SURFACE, lw=2)
            ax.tick_params(which="minor", length=0)
    colorbar = fig.colorbar(heatmap, cax=color_axis, extend="both", ticks=(-2, -1, 0, 1, 2))
    colorbar.ax.set_yticklabels(["0.01×", "0.1×", "1×", "10×", "100×"])
    colorbar.set_label("Classical paths per IQAE oracle call")
    fig.text(0.5, -0.01,
             "Best classical sampler paths ÷ IQAE oracle calls (summed across QPUs for distributed IQAE).\n"
             "Blue (>1×): fewer IQAE calls. White (1×): equal counts. Red (<1×): fewer classical paths.\n"
             "Each color step is a factor of 10. Gray dash: no ratio at the target accuracy.",
             ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), w_pad=1.0)
    _save(fig, "iqae_advantage.png")


def scaling_points(data, ns, level: float) -> np.ndarray:
    """Rows (amplitude, IQAE queries, IQAE k_max, best classical paths, n, target index)."""
    pts = []
    for n in ns:
        for k, t in enumerate(TARGETS):
            q, flag, km = data.cost_to(level, n, t, "iqae")
            if not np.isfinite(q) or flag == "<=":
                continue
            a = next(r["amplitude"] for r in data.exact_rows if r["n"] == n and r["target"] == t)
            pts.append((a, q, km, data.best_classical(level, n, t)[0], n, k))
    return np.array(pts)


def fig_rarity(data, ns) -> dict:
    """Queries and deepest Grover power vs the amplitude being estimated; returns fitted slopes."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    slopes = {}
    aa = np.geomspace(5e-9, 0.2, 50)
    for level in (0.10, 0.01):
        P = scaling_points(data, ns, level)
        if len(P) < 2:
            continue
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
    fig.suptitle("IQAE query cost and circuit depth as target amplitude falls")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.6)
    _save(fig, "iqae_rarity.png")
    return slopes


def depth_rows(data, ns) -> list[dict]:
    out = []
    for t, level in (("exceedance", 0.10), ("exceedance", 0.01), ("all_exhausted", 0.10),
                     ("all_exhausted", 0.01)):
        for n in ns:
            q, _, km = data.cost_to(level, n, t, "iqae")
            if not np.isfinite(q):
                continue
            c = data.counts(n, t)
            cxd = c["cx_A"] + km * c["cx_Q"]
            out.append(dict(n=n, target=t, level=level, queries=q, k_max=km, cx_A=c["cx_A"],
                            cx_Q=c["cx_Q"], qubits_A=c["qubits_A"], qubits_Q=c["qubits_Q"],
                            cx_deepest=cxd))
    return out


def fig_depth(data, ns) -> None:
    rows = depth_rows(data, ns)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    series = ((("exceedance", 0.10), "#257de0", "o", "-"), (("exceedance", 0.01), "#257de0", "o", "--"),
              (("all_exhausted", 0.10), "#f47230", "s", "-"), (("all_exhausted", 0.01), "#f47230", "s", "--"))
    for (t, lvl), col, mk, ls in series:
        rr = [r for r in rows if r["target"] == t and r["level"] == lvl]
        if not rr:
            continue
        lab = f"{SHORT[t]}, {lvl:.0%} error"
        face = SURFACE if lvl == 0.01 else col
        ax.plot([r["n"] for r in rr], [r["cx_deepest"] for r in rr], color=col, marker=mk, ls=ls, label=lab,
                markerfacecolor=face, markeredgecolor=col if lvl == 0.01 else SURFACE,
                markeredgewidth=1.1 if lvl == 0.01 else 0.7)
    ax.set(yscale="log", ylabel="two-qubit gates", title="Two-qubit gate count in the deepest IQAE circuit",
           xticks=ns, xlabel="countries")
    ax.legend(loc="upper center", ncol=2, fontsize=6.6, bbox_to_anchor=(0.5, -0.2))
    _save(fig, "iqae_gate_count.png")


def fig_tradeoff(data) -> None:
    pts = tradeoff_points(data)
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
        c0 = np.exp(np.mean([np.log(r[1] * r[2]) for r in rr]))  # calls x k_max = const guide
        x = np.geomspace(min(r[1] for r in rr) / 1.5, max(r[1] for r in rr) * 1.5, 20)
        ax.plot(x, c0 / x, color=col, lw=0.8, ls="--", alpha=0.7)
    ax.plot([], [], color=INK2, lw=0.8, ls="--", label="k_max × calls = constant")
    ax.set(xscale="log", yscale="log", xlabel="oracle calls to reach the RMSE",
           ylabel="deepest Grover power k_max", title="IQAE query and depth trade-off by shots per round")
    ax.legend(fontsize=6.4, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    _save(fig, "iqae_shot_tradeoff.png")


def tradeoff_points(data) -> dict:
    """{(n, target, level): [(shots per round, calls, k_max), ...]} at equal achieved RMSE."""
    rows = [r for r in data.tradeoff if r["study"] == "shots"]
    out = {}
    for n, t, lvl in ((5, "exceedance", 0.01), (10, "all_exhausted", 0.10)):
        pts = []
        for s in sorted({r["shots"] for r in rows if r["n"] == n and r["target"] == t}):
            q, _, km = data.sweep_point("shots", "shots", s, n, t, lvl)
            if np.isfinite(q):
                pts.append((s, q, km))
        out[(n, t, lvl)] = pts
    return out


def resource_rows(data, ns, target: str = "exceedance", level: float = 0.01) -> list[dict]:
    out = []
    for n in ns:
        c = data.counts(n, target)
        q, _, km = data.cost_to(level, n, target, "iqae")
        if not np.isfinite(q):
            continue
        e = logical_estimate(int(c["qubits_Q"]), int(c["t_A"]), int(c["rot_A"]), int(c["t_Q"]),
                             int(c["rot_Q"]), q, km)
        cv, meth, cflag = data.best_classical(level, n, target)
        spp = float(np.median([r["seconds_per_path"] for r in data.classical
                               if r["n"] == n and r["method"] == meth]))
        t_ex = next(r["t_exact_s"] for r in data.exact_rows if r["n"] == n)
        out.append(dict(n=n, target=target, level=level, queries=q, k_max=km, cx_A=c["cx_A"], cx_Q=c["cx_Q"],
                        qubits_A=c["qubits_A"], cx_deepest=c["cx_A"] + km * c["cx_Q"],
                        best_classical=meth, classical_paths=cv, classical_flag=cflag,
                        classical_seconds=cv * spp, exact_seconds=t_ex, **e))
    return out


def fig_time(data, ns) -> None:
    rr = resource_rows(data, ns)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    x = [r["n"] for r in rr]
    _plot(ax, x, [r["runtime_s_at_10000Hz"] for r in rr], "#257de0", "o",
          "IQAE, 10 kHz logical T rate", ls="--")
    _plot(ax, x, [r["runtime_s_at_1000000Hz"] for r in rr], "#257de0", "o",
          "IQAE, 1 MHz logical T rate")
    _plot(ax, x, [r["classical_seconds"] for r in rr], "#f2a600", "D", "best classical sampler, laptop")
    _plot(ax, x, [r["exact_seconds"] for r in rr], INK2, "s", "exact factorised solver, laptop")
    for s, lab in ((1, "1 s"), (3600, "1 h"), (86400, "1 day")):
        ax.axhline(s, color=GRID, lw=0.9, zorder=0)
        ax.text(max(ns) + 0.3, s, lab, fontsize=6.3, color=INK2, va="center")
    ax.set(yscale="log", ylim=(1e-4, 1e7), xticks=ns, xlabel="countries",
           ylabel="seconds", title="Time to estimate loss exceedance at 1% relative RMSE")
    ax.legend(fontsize=6.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    _save(fig, "exceedance_runtime.png")


def generate(data: an.BenchmarkData) -> None:
    setup()
    ns = data.ns()
    fig_exceedance(ns)
    fig_benchmark(data, ns)
    fig_advantage(data, ns)
    fig_rarity(data, ns)
    fig_depth(data, ns)
    fig_tradeoff(data)
    fig_time(data, ns)
    print("figures written to", FIG)
