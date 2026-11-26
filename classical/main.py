"""python -m classical [--scenario adb-base] [--n 65536] [--seed 0]

Prints the exact reference values (Table 5) for all scenarios, then the
sampling baselines for one scenario against the exact values.
"""
import argparse

from . import METHODS, SCENARIOS, TARGETS, estimate, exact_targets

SHORT = {
    "term_expected_principal_loss_usd": "EL (USD)",
    "term_joint_attachment_probability": "Both pay",
    "term_portfolio_loss_exceedance_probability": "L > T",
    "term_portfolio_stop_loss_usd": "Stop-loss (USD)",
    "term_both_bonds_exhausted_probability": "Both exhausted",
}


def fmt(name: str, x: float) -> str:
    return f"{x:,.0f}" if name.endswith("_usd") else f"{100 * x:.4f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="adb-base", choices=list(SCENARIOS))
    ap.add_argument("--n", type=int, default=2**16, help="term paths per method")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Exact reference values (three-year term)\n")
    print(f"{'scenario':18s}" + "".join(f"{SHORT[t]:>17s}" for t in TARGETS))
    for sc in SCENARIOS:
        ex = exact_targets(sc)
        print(f"{sc:18s}" + "".join(f"{fmt(t, ex[t]):>17s}" for t in TARGETS))

    ex = exact_targets(args.scenario)
    res = {m: estimate(args.scenario, m, args.n, args.seed) for m in METHODS}
    print(f"\nSampling baselines: {args.scenario}, n = {args.n:,}, seed = {args.seed}\n")
    for t in TARGETS:
        print(f"{SHORT[t]}  (exact {fmt(t, ex[t])})")
        for m in METHODS:
            e = res[m][t]
            rel = f"{100 * abs(e.mean - ex[t]) / ex[t]:6.2f}%" if ex[t] else "   n/a"
            print(f"  {m:12s} {fmt(t, e.mean):>15s} ± {fmt(t, e.ci95):<13s} rel.err {rel}")
        print()


if __name__ == "__main__":
    main()
