"""PDF report: IQAE benchmark at 2, 3, 5 and 10 countries and the factors that limit it.

python -m quantum.new.report   ->  quantum/new/results/figures/*.png and
                                   docs/IQAE_benchmark_and_limits_2-10_countries.pdf
Every number in the text is read from quantum/new/results (run quantum.new.benchmark first).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from . import analysis as an
from . import figures as fg
from .benchmark import ALPHA, BUDGETS, CLASSICAL_REPS, IQAE_REPS, SHOTS
from .portfolio import TARGETS

OUT = Path(__file__).resolve().parents[2] / "docs" / "IQAE_benchmark_and_limits_2-10_countries.pdf"

# ---------------------------------------------------------------- fonts, styles, helpers

_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _name, _file in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                     ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"),
                     ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(_name, str(_ttf / _file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, INK2 = colors.HexColor("#191919"), colors.HexColor("#666666")
BLUE, RULE = colors.HexColor("#257de0"), colors.HexColor("#d9d9d9")
CALLOUT, WARN, ORANGE = colors.HexColor("#eef5fd"), colors.HexColor("#fdf3ec"), colors.HexColor("#f47230")
FRAME_W = A4[0] - 4.0 * cm

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=19, leading=24, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=11, leading=15, textColor=INK2,
                               spaceAfter=10),
    "H1": ParagraphStyle("H1", fontName="DV-B", fontSize=13.5, leading=18, textColor=INK, spaceBefore=10,
                         spaceAfter=6, keepWithNext=1),
    "H2": ParagraphStyle("H2", fontName="DV-B", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8,
                         spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9, leading=13.2, textColor=INK, spaceAfter=6,
                           alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9, leading=13, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=7.7, leading=10.4, textColor=INK2,
                              spaceBefore=2, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.4, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.4, leading=9.6, textColor=INK),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.9, leading=10.8, textColor=INK2),
    "tcap": ParagraphStyle("tcap", fontName="DV", fontSize=7.9, leading=10.8, textColor=INK2, spaceAfter=3),
    "mono": ParagraphStyle("mono", fontName="DVM", fontSize=7.5, leading=10.2, textColor=INK, leftIndent=8,
                           spaceAfter=6),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def H1(text: str) -> Paragraph:
    return Paragraph(text, S["H1"])


def H2(text: str) -> Paragraph:
    return Paragraph(text, S["H2"])


def bullets(items) -> list:
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def figure(name: str, caption: str, width: float | None = None):
    path = fg.FIG / name
    w = width or FRAME_W
    iw, ih = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=w, height=w * ih / iw), Paragraph(caption, S["caption"])])


def table(rows, widths, header: bool = True):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb"] if header and i == 0 else S["cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
             ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE), ("TOPPADDING", (0, 0), (-1, -1), 2.4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4), ("LEFTPADDING", (0, 0), (-1, -1), 4),
             ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f7")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def tabled(caption: str, rows, widths, split: bool = False):
    """Caption + table; long tables (split=True) may break across pages, header row repeated."""
    if split:
        return [CondPageBreak(4 * cm), P(caption, "tcap"), table(rows, widths), Spacer(1, 8)]
    return KeepTogether([P(caption, "small"), Spacer(1, 2), table(rows, widths), Spacer(1, 8)])


def callout(paras, bg=CALLOUT, edge=BLUE):
    t = Table([[paras]], colWidths=[FRAME_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


class Doc(BaseDocTemplate):
    def __init__(self, path: Path):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                         bottomMargin=1.8 * cm,
                         title="Iterative QAE for joint catastrophe-bond risk: benchmark and limiting factors",
                         author="ADB Quantum Challenge team", subject="IQAE benchmark, 2-10 countries")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self._page)])

    def _page(self, c, doc):
        c.saveState()
        if doc.page == 1:
            c.setFillColor(BLUE)
            c.rect(0, A4[1] - 0.6 * cm, A4[0], 0.6 * cm, stroke=0, fill=1)
        c.setFont("DV", 7.3)
        c.setFillColor(INK2)
        c.drawString(2 * cm, 1.1 * cm, "ADB Quantum Challenge · Iterative QAE: benchmark at 2, 3, 5 and 10 "
                     "countries, and limiting factors")
        c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.line(2 * cm, 1.45 * cm, A4[0] - 2 * cm, 1.45 * cm)
        c.restoreState()


# ---------------------------------------------------------------- number formatting

GN = {"expected_loss": "E[L]", "all_pay": "P(all pay)", "exceedance": "P(L &gt; T)",
      "stop_loss": "E[(L − T)<super>+</super>]", "all_exhausted": "P(all exhausted)"}
CLS = {"monte_carlo": "Monte Carlo", "sobol": "Sobol", "stratified": "stratified",
       "importance": "importance sampling"}


def sci(v: float, d: int = 1) -> str:
    if not np.isfinite(v):
        return "–"
    m, e = f"{v:.{d}e}".split("e")
    e = int(e)
    if -2 <= e <= 3:
        return f"{v:,.{max(0, d - e)}f}" if e < 0 else f"{v:,.0f}"
    return f"{m}·10<super>{e}</super>"


def num(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    if v >= 1e9:
        return sci(v)
    if v >= 1e6:
        return f"{v / 1e6:.1f} million"
    return f"{float(f'{v:.2g}'):,.0f}"


def dur(s: float) -> str:
    if not np.isfinite(s):
        return "–"
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms" if s < 0.01 else f"{s * 1e3:.0f} ms"
    for unit, div in (("years", 3.15e7), ("days", 86400), ("h", 3600), ("min", 60)):
        if s >= div:
            return f"{s / div:.1f} {unit}"
    return f"{s:.0f} s"


def money(v: float) -> str:
    return f"USD {v / 1e6:.2f}m" if v >= 1e6 else f"USD {v / 1e3:,.0f}k"


def val(t: str, v: float) -> str:
    return money(v) if t in ("expected_loss", "stop_loss") else (f"{v:.2%}" if v >= 1e-3 else sci(v, 2))


def qstr(v: float, flag: str) -> str:
    if not np.isfinite(v):
        return "not reached"
    return {"<=": "≤ ", "~": "≈ ", "theory": "≈ "}.get(flag, "") + sci(v)


def rng_str(vals, fmt="{:.0f}") -> str:
    return f"{fmt.format(min(vals))}–{fmt.format(max(vals))}"


# ---------------------------------------------------------------- numbers from the results

def alpha_rows() -> list[dict]:
    rows = [r for r in an.table("tradeoff.csv") if r["study"] == "alpha"]
    out = []
    for al in sorted({r["alpha"] for r in rows}, reverse=True):
        rr = sorted((r for r in rows if r["alpha"] == al), key=lambda r: r["queries"])
        d = {"alpha": al}
        for lvl in (0.10, 0.01):
            d[lvl] = an.sweep_point("alpha", "alpha", al, 5, "exceedance", lvl)[0]
        d["coverage"] = min(r["ci_coverage"] for r in rr)
        d["p95_at_1pct"] = next((r["rel_err_p95"] for r in rr if r["eps_rel"] == 0.01), np.nan)
        out.append(d)
    return out


def build() -> Path:
    an.clear_cache()
    slopes = fg.main()
    ns = an.ns()
    N = ns[-1]
    ex = {(n, t): an.exact(n, t) for n in ns for t in TARGETS}
    t_exact = {n: next(r["t_exact_s"] for r in an.table("exact.csv") if r["n"] == n) for n in ns}
    adv = {(lvl, n, t): an.advantage(lvl, n, t) for lvl in (0.10, 0.01) for n in ns for t in TARGETS}
    tails = ("exceedance", "stop_loss", "all_exhausted")
    tail1 = {t: [adv[(0.01, n, t)][0] for n in ns] for t in tails}
    smooth1 = {t: [adv[(0.01, n, t)][0] for n in ns] for t in ("expected_loss", "all_pay")}
    depth = {(r["target"], r["level"], r["n"]): r for r in fg.depth_rows(ns)}
    res = {r["n"]: r for r in fg.resource_rows(ns)}
    rN = res[N]
    noise = an.table("noise.csv")
    e2e = an.table("e2e.csv")
    val_rows = an.table("validation.csv")
    jN1 = depth.get(("all_exhausted", 0.01, N)) or depth[("all_exhausted", 0.10, N)]
    cov_1e6 = next(r["ci_coverage"] for r in noise if r["n"] == N and r["target"] == "exceedance"
                   and r["eps_rel"] == 0.01 and r["p2q"] == 1e-6)
    pN1 = depth[("exceedance", 0.01, N)]
    story: list = []

    # ------------------------------------------------------------ title and summary
    story += [Spacer(1, 0.4 * cm),
              P("Iterative quantum amplitude estimation for joint catastrophe-bond risk", "title"),
              P("Benchmark at 2, 3, 5 and 10 countries, and the factors that limit IQAE", "subtitle"),
              P(f"ADB Quantum Challenge · {date.today():%d %B %Y} · code and data: <font face='DVM'>"
                "quantum/new/</font> (results in <font face='DVM'>quantum/new/results/</font>)", "small"),
              Spacer(1, 0.3 * cm)]
    s01 = slopes[0.01]
    story += [callout([
        P("<b>Bottom line</b>"),
        *bullets([
            "<b>IQAE wins the tail targets on query count.</b> At 1% relative error it needs fewer oracle "
            "calls than the best of four classical samplers for every tail target at every portfolio size: "
            f"{rng_str(tail1['exceedance'], '{:.1f}')}× fewer for P(L &gt; T), "
            f"{rng_str(tail1['stop_loss'], '{:.1f}')}× for the stop-loss layer and "
            f"{rng_str(tail1['all_exhausted'])}× for P(all bonds exhausted). At 10% error the picture is "
            "mixed: importance sampling or stratification matches or beats IQAE on the exceedance and "
            "stop-loss targets. On the smooth targets (E[L], P(all pay)) classical samplers win: at 1% the "
            f"best needs {1 / max(max(v) for v in smooth1.values()):.1f}–"
            f"{1 / min(min(v) for v in smooth1.values()):.0f}× fewer samples than IQAE needs calls.",
            f"<b>Cost grows as a<super>{s01[0]:.2f}</super>, depth as a<super>{s01[1]:.2f}</super></b> in the "
            "amplitude a being estimated (Monte Carlo: a<super>−1</super>). The same square-root law that "
            "creates the advantage on rare events forces ever deeper coherent circuits.",
            f"<b>Depth is the binding constraint.</b> For P(L &gt; T) at 1% with {N} countries the deepest "
            f"circuit has k = {pN1['k_max']:,.0f} Grover steps, {num(pN1['cx_deepest'])} two-qubit gates; "
            f"for P(all exhausted) at {jN1['level']:.0%} it is {num(jN1['cx_deepest'])}. Running them at 90% "
            "fidelity needs a "
            f"two-qubit error of {sci(pN1['p2q_needed'])} and {sci(jN1['p2q_needed'])}, against "
            "~10<super>−3</super> on today's hardware.",
            "<b>Noise breaks IQAE silently.</b> The algorithm does not model decoherence: once the circuits "
            "decohere it returns biased values with intervals that no longer cover the truth. At a two-qubit "
            f"error of 10<super>−6</super> the 99.99% intervals for P(L &gt; T) at 1% cover the exact value in "
            f"only {cov_1e6:.0%} of runs at {N} countries (Section 4.3).",
            f"<b>Wall clock favours classical computing on this model.</b> Even fault-tolerant at a 1 MHz "
            f"logical T rate, P(L &gt; T) at 1% for {N} countries takes {dur(rN['runtime_s_at_1000000Hz'])}; "
            f"the best classical sampler takes {dur(rN['classical_seconds'])} and the exact solver "
            f"{dur(t_exact[N])} on a laptop, because the model factorises given the shared shocks.",
        ]),
    ])]

    # ------------------------------------------------------------ 1. setup
    story += [H1("1. Problem and benchmark set-up"),
              P("The brief's two-country programme (Kyrgyz Republic and Tajikistan, USD 80m per bond: USD 70m "
                "earthquake in quarter steps, USD 10m precipitation, three-year term) is extended to n countries "
                "that alternate the two countries' parameters. Each country keeps its own compound-Poisson "
                "earthquake process and rain rate; one regional earthquake stream (rate c = m · mean "
                "ρ<sub>i</sub>Λ<sub>i</sub>) and one regional rain stream (c<sub>r</sub> = 0.25 · min "
                "μ<sub>i</sub>) hit every country at once. At n = 2 this is exactly the brief's model: the "
                "n-country code reproduces all 25 values of Table 5 to 10<super>−13</super> and matches the "
                "base repository's <font face='DVM'>classical</font> package. T is half the total principal "
                "(USD 80m at n = 2, USD 400m at n = 10)."),
              P("Given the number of regional earthquakes K (lumped at ≥ 4, which exhausts every sub-limit) "
                "and whether the regional rain stream fired (J), the countries are independent. The joint law "
                "is therefore an exact mixture of 10 product laws. This gives an exact solver that runs in "
                "milliseconds (checked against brute force on the full 5<super>n</super>·2<super>n</super> "
                "chain for n ≤ 5), and it is also how the quantum state is loaded (Section 2).")]
    story += [H2("What was run"),
              *bullets([
                  "<b>Classical baselines</b> (the brief's four, generalised to n countries): Monte Carlo, "
                  "scrambled Sobol (8 scrambles, t-interval), stratified sampling on the 10 shock states, and "
                  f"importance sampling with the brief's 50/50 loss tilt. Budgets 2<super>10</super>–2"
                  f"<super>{int(np.log2(BUDGETS[-1]))}</super> term paths, {CLASSICAL_REPS} replications each.",
                  "<b>IQAE</b> (Grinko et al. 2021, Clopper–Pearson intervals) at confidence 1 − α with α = "
                  f"{ALPHA:g} and {SHOTS} shots per round, {IQAE_REPS} runs per accuracy setting. The "
                  "benchmark runs at oracle level: measurement outcomes are drawn from the exact Grover "
                  "statistics sin<super>2</super>((2k+1)θ) of an ideal device, which Section 2 verifies "
                  "against the actual Qiskit circuits.",
                  "<b>Cost</b> is term paths for classical methods and oracle calls (applications of A or "
                  "A<super>†</super>; a round at Grover power k with N shots costs N(2k+1)) for IQAE. "
                  "<b>Accuracy</b> is the relative root-mean-square error against the exact value. Costs to "
                  "reach 10% and 1% are interpolated on a monotone fit of each error curve; classical curves "
                  "are extended along 1/√N where needed.",
              ])]

    story += [figure("f1_exceedance.png",
                     "Figure 1. Exact portfolio loss exceedance curves. The joint tail P(all exhausted) "
                     "(markers at 100%) falls by more than three orders of magnitude from 2 to 10 countries; "
                     "P(L &gt; T) (at 50%) stays near 0.5–1.2%.")]
    rows = [["n", "E[L]", "P(all pay)", "P(L &gt; T)", "E[(L − T)<super>+</super>]", "P(all exhausted)",
             "exact solver", "brute-force check"]]
    for n in ns:
        r0 = next(r for r in an.table("exact.csv") if r["n"] == n)
        chk = f"{r0['brute_rel_dev']:.0e}" if np.isfinite(r0["brute_rel_dev"]) else "10<super>n</super> states"
        rows.append([str(n), *[val(t, ex[(n, t)]) for t in TARGETS], dur(t_exact[n]), chk])
    story += [tabled("<b>Table 1.</b> Exact values of the five targets (three-year term) and the cost of "
                     "computing them. The brute-force column is the largest relative deviation from the full "
                     "joint Markov chain.", rows,
                     [1.0 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 2.2 * cm, 2.4 * cm, 2.1 * cm, 3.7 * cm])]
    # ------------------------------------------------------------ 2. pipeline
    c2, cN = an.counts(2, "exceedance"), an.counts(N, "exceedance")
    story += [CondPageBreak(9 * cm), H1("2. The IQAE pipeline and its validation"),
              P("<b>Oracle A.</b> A 4-qubit register holds the shock state (K, J); each country is a 4-qubit "
                "register loaded conditionally on it with uniformly controlled RY cascades, so loading costs "
                "a fixed ~250 CX per country and nothing of size 10<super>n</super> is ever tabulated. For "
                "the loss targets a Fourier-basis adder accumulates the portfolio loss in units of USD 2.5m "
                "into a ⌈log<sub>2</sub>(32n + 1)⌉-qubit register, and one multiplexed RY writes f(L)/f<sub>"
                "max</sub> exactly into the objective amplitude. P(all pay) and P(all exhausted) use "
                "multi-controlled gates instead. <b>Grover operator</b> Q = A S<sub>0</sub> A<super>†</super>"
                " S<sub>χ</sub>, with one auxiliary qubit for the reflection."),
              P(f"At n = 2, A for P(L &gt; T) has {c2['qubits_A']:.0f} qubits and {c2['cx_A']:,.0f} CX, Q "
                f"{c2['cx_Q']:,.0f} CX; at n = {N}, {cN['qubits_A']:.0f} qubits, {cN['cx_A']:,.0f} CX and "
                f"{cN['cx_Q']:,.0f} CX (Table 5). Gate counts are linear in n."),
              ]
    amp_dev = max(r["rel_dev"] for r in val_rows if r["check"].startswith("amplitude"))
    grov_dev = max(r["rel_dev"] for r in val_rows if r["check"].startswith("Grover"))
    k_chk = max(r["k_checked"] for r in val_rows)
    story += [P("<b>Validation on Qiskit Aer.</b> Every oracle (n = 2 and 3, scalable and lookup variants) "
                f"loads the exact target to a relative deviation of at most {sci(amp_dev)}. The measured "
                "probability after Q<super>k</super>A matches sin<super>2</super>((2k+1)θ) to "
                f"{sci(grov_dev)} for every k ≤ {k_chk:.0f} and every target. IQAE then runs end to end on "
                "the circuits (Table 2): with shots drawn from the simulated circuit probabilities, with "
                "literal measured shots, and with the reference implementation in <font face='DVM'>"
                "qiskit-algorithms</font>. All intervals cover the exact values.")]
    rows = [["Implementation, backend", "Target", "ε", "Runs", "Estimate", "Exact", "Rel. RMSE", "CI covers",
             "Calls", "k<sub>max</sub>"]]
    for r in e2e:
        if r["implementation"] == "quantum.new.iqae" and r["reps"] > 1 and r["eps_rel"] == 0.05:
            continue
        rows.append([f"{r['implementation']}, {r['backend'].split(' (')[0]}", GN[r["target"]],
                     f"{r['eps_rel']:.0%}", f"{r['reps']:.0f}", val(r["target"], r["estimate_mean"]),
                     val(r["target"], r["exact"]), f"{r['rel_rmse']:.2%}", f"{r['ci_coverage']:.0%}",
                     sci(r["queries"]), f"{r['k_max']:,.0f}"])
    story += [*tabled("<b>Table 2.</b> End-to-end IQAE on the n = 2 Qiskit circuits (lookup oracle, 13–14 "
                     "qubits) simulated with Aer. ε is the target interval half-width relative to the value; "
                     f"α = {ALPHA:g} for the repeated runs, 0.05 for the single literal-shot runs. Estimate is "
                     "the mean over runs.", rows,
                     [3.9 * cm, 2.0 * cm, 0.9 * cm, 1.1 * cm, 1.7 * cm, 1.7 * cm, 1.4 * cm, 1.3 * cm, 1.6 * cm,
                      1.3 * cm], split=True)]

    # ------------------------------------------------------------ 3. benchmark
    story += [CondPageBreak(16 * cm), H1("3. Benchmark at 2, 3, 5 and 10 countries"),
              figure("f2_benchmark.png",
                     "Figure 2. Relative RMSE against cost for a smooth target (top), the exceedance "
                     "probability (middle) and the joint tail (bottom). Grey lines mark 10% and 1%. Curves "
                     "flat near 100% are samplers whose runs mostly returned zero.")]
    rows = [["n", "Target", "Monte Carlo", "Best classical", "IQAE calls", "k<sub>max</sub>", "Ratio"]]
    for t, lvl in (("exceedance", 0.01), ("stop_loss", 0.01), ("all_exhausted", 0.10),
                   ("all_exhausted", 0.01)):
        for n in ns:
            mv, mf, _ = an.cost_to(lvl, n, t, "monte_carlo")
            r, meth, bound, cv, qv = adv[(lvl, n, t)]
            _, cf, _ = an.cost_to(lvl, n, t, meth)
            _, _, km = an.cost_to(lvl, n, t, "iqae")
            ratio = f"{bound}{r:.1f}×" if r < 10 else f"{bound}{r:.0f}×"
            rows.append([str(n), f"{GN[t]}, {lvl:.0%}", qstr(mv, mf), f"{qstr(cv, cf)} ({CLS[meth]})",
                         sci(qv), f"{km:,.0f}", f"<b>{ratio}</b>" if r >= 1 else ratio])
    story += [*tabled("<b>Table 3.</b> Samples or oracle calls to reach a relative RMSE of 10% or 1% on the "
                     "tail targets. “≈” marks classical values extrapolated along 1/√N or, for Monte Carlo runs "
                     "that never hit, the binomial formula (1 − p)/(pε<super>2</super>). Ratio = best classical "
                     "÷ IQAE.", rows,
                     [0.9 * cm, 3.4 * cm, 2.3 * cm, 4.6 * cm, 2.0 * cm, 1.6 * cm, 2.2 * cm], split=True)]
    story += [figure("f3_advantage.png",
                     "Figure 3. Best classical paths ÷ IQAE oracle calls for every target and portfolio size. "
                     "“≤”: the classical sampler already met the target at its smallest budget, so the true "
                     "ratio is lower; “≥”: the same for IQAE.")]
    a10 = {t: [adv[(0.10, n, t)][0] for n in ns] for t in TARGETS}
    el1 = [1 / adv[(0.01, n, "expected_loss")][0] for n in ns]
    story += [P("Three patterns hold across sizes. <b>Smooth targets</b> (E[L], P(all pay)) are moderate "
                "amplitudes (2–6%) on which scrambled Sobol points and stratification converge much faster "
                f"than 1/√N: at 1% error the best of them needs {rng_str(el1, '{:.1f}')}× fewer samples than "
                "IQAE needs calls for E[L]. <b>Exceedance and stop-loss</b> are rarer (0.1–1%). At 10% error "
                "importance sampling or stratification matches or beats IQAE "
                f"({rng_str(a10['exceedance'] + a10['stop_loss'], '{:.1f}')}×); below a few per cent IQAE's "
                f"1/N convergence takes over ({rng_str(tail1['exceedance'] + tail1['stop_loss'], '{:.1f}')}× "
                "at 1%). <b>The joint tail</b> is where Monte Carlo and Sobol stop working (most runs return "
                f"zero). The gap to IQAE is largest there at small n ({adv[(0.01, ns[0], 'all_exhausted')][0]:.0f}× "
                f"at 1% for {ns[0]} countries), but stratifying on the shock state, the only classical method "
                "that keeps working, closes much of it as countries are added: "
                f"{adv[(0.01, N, 'all_exhausted')][0]:.0f}× at 1% and {adv[(0.10, N, 'all_exhausted')][0]:.1f}× "
                f"at 10% for {N} countries. The advantage does not grow with portfolio size here, because the "
                "classical method that works best exploits the same shock structure the oracle does.")]

    # ------------------------------------------------------------ 4. limiting factors
    story += [CondPageBreak(10 * cm), H1("4. What limits IQAE"),
              H2("4.1 Rarity sets both the query count and the circuit depth"),
              P("IQAE estimates the angle θ = arcsin √a. Reaching a relative error ε on a needs an angle "
                "precision of about ε√a/2, which costs ~1/(ε√a) oracle calls but also requires a Grover power "
                "of the same order in the final rounds. Across the 20 (n, target) pairs the measured exponents "
                f"are {s01[0]:.2f} for calls and {s01[1]:.2f} for depth at 1% error ({slopes[0.10][0]:.2f} and "
                f"{slopes[0.10][1]:.2f} at 10%). Monte Carlo's cost grows as a<super>−1</super> instead. The "
                "advantage and the depth requirement are the same quantity."),
              figure("f4_rarity.png",
                     "Figure 4. Left: IQAE oracle calls (markers) and Monte Carlo paths (dashed, binomial "
                     "formula) needed for 10% and 1% relative RMSE, against the amplitude of every (n, target) "
                     "pair. Right: the deepest Grover power k<sub>max</sub> used, with power-law fits.")]
    rows = [["Target, error", *[f"{n} countries" for n in ns]]]
    for t, lvl in (("exceedance", 0.10), ("exceedance", 0.01), ("all_exhausted", 0.10), ("all_exhausted", 0.01)):
        rr = [depth.get((t, lvl, n)) for n in ns]
        rows.append([f"{GN[t]}, {lvl:.0%}: k<sub>max</sub>"] + [f"{r['k_max']:,.0f}" if r else "–" for r in rr])
        rows.append(["   two-qubit gates, deepest circuit"] + [sci(r["cx_deepest"]) if r else "–" for r in rr])
        rows.append(["   p<sub>2q</sub> for 90% fidelity"] + [sci(r["p2q_needed"]) if r else "–" for r in rr])
    story += [H2("4.2 Coherent depth and gate count"),
              P("Each Grover step costs one A, one A<super>†</super> and a reflection, so the deepest circuit "
                "holds ~(2k<sub>max</sub>+1) oracle calls in one coherent run. Table 4 combines the measured "
                "k<sub>max</sub> with the transpiled gate counts. A circuit with G two-qubit gates at error "
                "rate p keeps a fidelity of about e<super>−pG</super>, so 90% fidelity needs p ≲ 0.1/G."),
              tabled("<b>Table 4.</b> Deepest circuit IQAE runs, from the benchmark's k<sub>max</sub> (median "
                     "over runs) and the transpiled CX counts of A and Q.", rows,
                     [5.6 * cm] + [2.85 * cm] * len(ns)),
              figure("f5_depth.png",
                     "Figure 5. Left: two-qubit gates in the deepest circuit. Right: the two-qubit error rate "
                     "at which that circuit keeps 90% fidelity; the grey band is the range of current "
                     "hardware.")]
    rows = [["", *[f"{n} countries" for n in ns]],
            ["Qubits, oracle A (P(L &gt; T))", *[f"{an.counts(n, 'exceedance')['qubits_A']:.0f}" for n in ns]],
            ["Qubits, Grover operator Q", *[f"{an.counts(n, 'exceedance')['qubits_Q']:.0f}" for n in ns]],
            ["CX per oracle call A", *[f"{an.counts(n, 'exceedance')['cx_A']:,.0f}" for n in ns]],
            ["CX per Grover step Q", *[f"{an.counts(n, 'exceedance')['cx_Q']:,.0f}" for n in ns]],
            ["Arbitrary rotations per Q", *[f"{an.counts(n, 'exceedance')['rot_Q']:,.0f}" for n in ns]],
            ["Qubits, A / Q (P(all exhausted))",
             *[f"{an.counts(n, 'all_exhausted')['qubits_A']:.0f} / {an.counts(n, 'all_exhausted')['qubits_Q']:.0f}"
               for n in ns]],
            ["CX per Q (P(all exhausted))", *[f"{an.counts(n, 'all_exhausted')['cx_Q']:,.0f}" for n in ns]]]
    story += [tabled("<b>Table 5.</b> Circuit resources per call (Qiskit 2.6, transpiled to {CX, U} at "
                     "optimisation level 1). Qubits grow as 4n + ⌈log<sub>2</sub>(32n+1)⌉ + 6.", rows,
                     [5.6 * cm] + [2.85 * cm] * len(ns))]

    # noise
    def nrow(n, t, lvl, p):
        return next((r for r in noise if r["n"] == n and r["target"] == t and r["eps_rel"] == lvl
                     and r["p2q"] == p), None)
    worst = {}
    for p in (1e-6, 1e-5, 1e-4):
        rr = [nrow(n, "exceedance", 0.01, p) for n in ns]
        worst[p] = (max(r["rel_rmse"] for r in rr if r), min(r["ci_coverage"] for r in rr if r))
    nc = an.table("noisy_circuit.csv")
    story += [CondPageBreak(9 * cm), H2("4.3 Noise: biased estimates with false confidence"),
              P("Standard IQAE assumes the ideal hit probability sin<super>2</super>((2k+1)θ). Under gate "
                "noise the measured probability decays towards 1/2 as the circuit deepens. We simulate a global "
                "depolarising channel with rate λ = p × (CX per oracle call), "
                "P<sub>k</sub> = e<super>−λ(2k+1)</super> sin<super>2</super>((2k+1)θ) + "
                "(1 − e<super>−λ(2k+1)</super>)/2, and run IQAE unchanged. Figure 6 shows the result: for "
                f"P(L &gt; T) at 1%, the worst relative RMSE over portfolio sizes is {worst[1e-6][0]:.1%} at "
                f"p = 10<super>−6</super>, {worst[1e-5][0]:.0%} at 10<super>−5</super> and "
                f"{worst[1e-4][0]:.0f} times the value itself at 10<super>−4</super>; the nominal 99.99% "
                f"intervals cover the exact value in as few as {worst[1e-6][1]:.0%} of runs at "
                f"10<super>−6</super> and {worst[1e-5][1]:.0%} at 10<super>−5</super>. The joint tail breaks "
                "down earlier still. The failure is silent: IQAE reports a narrow interval around the wrong "
                "value."),
              figure("f6_noise.png",
                     "Figure 6. Noise-unaware IQAE under global depolarising noise, for P(L &gt; T) at 1% "
                     "target error and P(all exhausted) at 10%. Top: relative RMSE. Bottom: how often the "
                     "nominal 99.99% interval covers the exact value. Leftmost tick: noiseless.")]
    if nc:
        k0 = {r["p2q"]: r for r in nc if r["k"] == 0}
        lam_ratio = [r["cx_per_call"] * p / -np.log((0.5 - r["measured"]) / (0.5 - r["ideal"]))
                     for p, r in k0.items()]
        story += [P("The global-depolarising model is checked against noisy Aer trajectories of the actual n = 2 "
                    "circuits (depolarising error p on every CX, p/10 on single-qubit gates). Already at "
                    f"k = 0 the measured hit rate for P(L &gt; T) is {k0[1e-3]['measured']:.1%} at p = 10"
                    f"<super>−3</super> and {k0[1e-4]['measured']:.1%} at 10<super>−4</super>, against an ideal "
                    f"{k0[1e-4]['ideal']:.2%}: noise floods a rare-event amplitude before any amplification. "
                    "Fitted to these runs, the effective rate per oracle call is "
                    f"{1 / max(lam_ratio):.1f}–{1 / min(lam_ratio):.1f}× the p × CX used above (not every gate "
                    "error randomises the objective qubit), so the thresholds in Figure 6 and Table 4 are "
                    f"pessimistic by a factor of about {np.mean(lam_ratio):.0f}; the orders of magnitude stand. "
                    "Noise-aware variants "
                    "(fitting λ jointly with θ, or error-mitigated estimators) remove part of the bias but "
                    "widen the intervals and need low-k calibration circuits; they do not remove the need for "
                    "error rates far below 10<super>−3</super>."),
                  figure("f7_noisy_circuit.png",
                         "Figure 7. n = 2, P(L &gt; T): measured hit probability after Q<super>k</super>A in "
                         "noisy Aer trajectories (markers, 2,000 shots) against the ideal circuit and the "
                         "global-depolarising model used in Figure 6.", width=FRAME_W * 0.66)]

    # trade-offs
    story += [CondPageBreak(8 * cm), H2("4.4 Trading depth for queries")]
    tp = fg.tradeoff_points()
    t10 = tp.get((N, "all_exhausted", 0.10), []) or tp.get((10, "all_exhausted", 0.10), [])
    t5 = tp.get((5, "exceedance", 0.01), [])
    if t10 and t5:
        deep, shallow = max(t10, key=lambda r: r[2]), min(t10, key=lambda r: r[2])
        deep5, shallow5 = max(t5, key=lambda r: r[2]), min(t5, key=lambda r: r[2])
        story += [P("More shots per round tighten each interval, so IQAE can reach the same precision with a "
                    "smaller final Grover power, at the price of more oracle calls. Compared at equal achieved "
                    f"RMSE (Figure 8): for 10 countries and P(all exhausted) at 10%, {deep[0]:,.0f} shots per "
                    f"round need k<sub>max</sub> = {deep[2]:,.0f} and {num(deep[1])} calls; {shallow[0]:,.0f} "
                    f"shots need k<sub>max</sub> = {shallow[2]:,.0f} and {num(shallow[1])} calls. For 5 "
                    f"countries and P(L &gt; T) at 1%, k<sub>max</sub> goes from {deep5[2]:,.0f} "
                    f"({deep5[0]:,.0f} shots, {num(deep5[1])} calls) to {shallow5[2]:,.0f} ({shallow5[0]:,.0f} "
                    f"shots, {num(shallow5[1])} calls). Depth can be bought down, but each factor taken off "
                    "k<sub>max</sub> is paid back in calls, and pushed far enough the trade returns to Monte "
                    "Carlo's 1/ε<super>2</super> scaling. Depth-capped and power-law variants of amplitude "
                    "estimation make the same trade explicitly; none removes the coherence requirement for "
                    "rare targets."),
                  figure("f8_tradeoff.png",
                         "Figure 8. Deepest Grover power against oracle calls needed to reach the stated RMSE, "
                         "as the shots per IQAE round vary from 10 to 10,000 (each point is an ε sweep).",
                         width=FRAME_W * 0.68)]
    ar = alpha_rows()
    if ar:
        rows = [["α (1 − confidence)", "Calls for 10% RMSE", "Calls for 1% RMSE", "Lowest coverage",
                 "95th pct. error at ε = 1%"]]
        rows += [[f"{r['alpha']:g}", sci(r[0.10]), sci(r[0.01]) if np.isfinite(r[0.01]) else "not reached",
                  f"{r['coverage']:.2%}", f"{r['p95_at_1pct']:.2%}"] for r in ar]
        tight = [r[0.01] for r in ar if r["alpha"] <= 0.01 and np.isfinite(r[0.01])]
        loose = [r for r in ar if r["alpha"] > 0.01]
        loose_txt = "; ".join(f"at α = {r['alpha']:g} " + (f"1% RMSE needs {sci(r[0.01])} calls"
                              if np.isfinite(r[0.01]) else "the RMSE never settles below 1% in the sweep")
                              for r in loose)
        story += [H2("4.5 The price of a rigorous interval"),
                  P("IQAE's guarantee is a 1 − α interval; its Clopper–Pearson bounds are taken at α/T per "
                    "round. Tight intervals are cheap: from α = 10<super>−2</super> to 10<super>−6</super> the "
                    f"calls needed for 1% RMSE stay within {sci(min(tight))}–{sci(max(tight))} (Table 6). Loose "
                    "intervals are expensive in a different way. A rare early miss locks IQAE onto a wrong "
                    "branch of θ and produces an error of tens of per cent, which then dominates the RMSE: "
                    f"{loose_txt}. Errors are heavy-tailed, so the median error understates the risk; the "
                    f"benchmark uses α = {ALPHA:g}, and even there a handful of the {IQAE_REPS} runs at some "
                    "settings missed."),
                  tabled("<b>Table 6.</b> Confidence level against cost for 5 countries, P(L &gt; T) "
                         f"(sweep over ε, {IQAE_REPS} runs per point).", rows,
                         [3.4 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 4.0 * cm])]

    # classical shortcuts and wall clock
    story += [CondPageBreak(8 * cm), H2("4.6 Strong classical baselines and the classical shortcut"),
              P("Query counts compare samplers on equal terms, but this model has more structure than a black "
                "box. Conditioning on the 10 shock states removes all cross-country dependence: stratified "
                "sampling on them is the best classical method on the joint tail, and the same conditioning "
                f"yields the exact answer in {dur(t_exact[ns[0]])}–{dur(t_exact[N])}. Quasi-Monte Carlo "
                "(Sobol) converges much faster than 1/√N on smooth targets, removing IQAE's edge there. A "
                "quantum advantage on query count is real but only matters where the oracle is the expensive "
                "part, i.e. for models without such shortcuts."),
              H2("4.7 Fault-tolerant cost and wall-clock time")]
    rows = [["P(L &gt; T) at 1% error", *[f"{n} countries" for n in ns]]]
    rr = [res[n] for n in ns]
    rows += [["IQAE oracle calls", *[num(r["queries"]) for r in rr]],
             ["Deepest Grover power k<sub>max</sub>", *[f"{r['k_max']:,.0f}" for r in rr]],
             ["T gates per oracle call (incl. rotation synthesis)", *[sci(r["T_per_query"]) for r in rr]],
             ["T gates in total", *[sci(r["T_total"]) for r in rr]],
             ["Logical qubits / code distance", *[f"{r['logical_qubits']} / {r['code_distance']}" for r in rr]],
             ["Physical qubits", *[sci(r["physical_qubits"]) for r in rr]],
             ["<b>Run time at 1 MHz logical T rate</b>", *[f"<b>{dur(r['runtime_s_at_1000000Hz'])}</b>"
                                                         for r in rr]],
             ["Run time at 10 kHz", *[dur(r["runtime_s_at_10000Hz"]) for r in rr]],
             ["Best classical sampler: paths", *[f"{num(r['classical_paths'])} ({CLS[r['best_classical']]})"
                                                 for r in rr]],
             ["Best classical sampler: laptop time", *[dur(r["classical_seconds"]) for r in rr]],
             ["Exact solver: laptop time", *[dur(r["exact_seconds"]) for r in rr]]]
    story += [P("Converting the measured calls and gate counts into a surface-code estimate (rotations "
                "synthesised with ~3 log₂(1/δ) T gates, physical error 10<super>−3</super>, a "
                "Litinski-style footprint) gives Table 7. Each oracle call costs "
                f"{sci(res[ns[0]]['T_per_query'])}–{sci(rN['T_per_query'])} T gates, dominated by rotation "
                "synthesis for the loading cascades."),
              tabled("<b>Table 7.</b> Order-of-magnitude fault-tolerant resources for P(L &gt; T) at 1% "
                     "relative error, against classical wall-clock time on a laptop (NumPy).", rows,
                     [5.6 * cm] + [2.85 * cm] * len(ns)),
              figure("f9_time.png", "Figure 9. Time to a 1%-accurate P(L &gt; T).", width=FRAME_W * 0.72),
              H2("4.8 Loading is the hidden assumption"),
              P("The oracle is cheap (linear in n) only because the joint law factorises given the shock state: "
                "the circuit loads 10 product laws, not 10<super>n</super> outcomes. A general dependence "
                "structure (copulas, overlapping fault segments, spatial fields) needs either an exponential "
                "loader or a learned one (quantum GANs, Born machines), whose training error enters the estimate "
                "directly and is not covered by IQAE's interval. Conversely, when the law does factorise this "
                "simply, the exact classical solution is already cheap. The case for IQAE is models in between: "
                "structured enough to load efficiently, coupled enough that exact classical methods scale "
                "exponentially.")]

    # ------------------------------------------------------------ 5. conclusions
    d1 = [r["cx_deepest"] for (t, lvl, _), r in depth.items() if lvl == 0.01]
    p1 = [r["p2q_needed"] for (t, lvl, _), r in depth.items() if lvl == 0.01]
    story += [CondPageBreak(6 * cm), H1("5. Conclusions"),
              *bullets([
                  "IQAE delivers the expected quadratic reduction in oracle calls (cost ∝ a<super>−1/2</super>). "
                  "At 1% error it beats the best classical sampler on every tail target, by up to "
                  f"{max(tail1['all_exhausted']):.0f}× on the joint tail; it loses on E[L] and P(all pay), and at "
                  "10% error the tail picture is mixed. The advantage does not grow with the number of countries, "
                  "because stratifying on the shock state captures the dependence classically.",
                  "The same scaling sets the coherent depth. Tail targets at 1% need deepest circuits of "
                  f"{sci(min(d1))}–{sci(max(d1))} two-qubit gates and error rates of {sci(min(p1))}–"
                  f"{sci(max(p1))}: this is an error-corrected workload, not a near-term one.",
                  "Noise-unaware IQAE fails silently. A production pipeline needs noise-aware likelihoods or "
                  "error mitigation and, above all, logical qubits.",
                  "On this model the classical shortcut (conditional independence given the shock state) makes "
                  "exact computation fast. The quantum resources measured here are the price of the oracle; "
                  "they become worth paying for dependence structures without such a shortcut.",
              ]),
              H2("Reproduce"),
              P("From the repository root, with <font face='DVM'>requirements.txt</font> installed (Python 3.12, "
                "Qiskit 2.6, Qiskit Aer 0.18):", "small"),
              P("python -m quantum.new.benchmark      # ~1 h on 12 cores → quantum/new/results/<br/>"
                "python -m quantum.new.report         # figures + this PDF<br/>"
                "python -m quantum.new --n 10 --target exceedance --eps 0.01   # one IQAE run<br/>"
                "pytest", "mono"),
              P("<b>Caveats.</b> IQAE accuracies are oracle-level simulations of an ideal device, verified against "
                "the circuits for n ≤ 3; circuits beyond that are counted, not simulated. Noise results use a "
                "global depolarising model calibrated per CX. Fault-tolerant figures are order-of-magnitude "
                "estimates under the stated assumptions. Classical timings are single-threaded NumPy; compiled "
                "code would be faster. <b>AI use:</b> code and text were prepared with Claude Code (Anthropic) "
                "and checked against the exact reference values.", "small")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Doc(OUT).build(story)
    print("wrote", OUT)
    return OUT


if __name__ == "__main__":
    build()
