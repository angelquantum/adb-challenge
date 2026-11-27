"""python -m quantum [--n 2] [--scenario adb-base] [--target all] [--eps 0.01] [--backend oracle]

Runs IQAE on the n-country portfolio and prints, per target, the estimate against the exact
value and the resources: oracle calls, shots, deepest Grover power, qubits and two-qubit gates
of the oracle A and of the Grover operator Q, and two-qubit gates in the deepest circuit.

Backends
    oracle  exact Grover statistics of the ideal circuit (any n; fast)
    aer     the actual Qiskit circuits Q^k A on Qiskit Aer (n <= 3; lookup oracle, exact for
            every target). --shots-mode runs literal measured shots instead of drawing them
            from the simulated circuit probabilities.
"""
import argparse
import time

import numpy as np

from classical import SCENARIOS

from .circuits import CircuitSampler, gate_counts, grover_operator, oracle
from .iqae import GroverModel, iqae
from .portfolio import LABELS, TARGETS, PortfolioModel, adb_portfolio


def fmt(target: str, v: float) -> str:
    return f"USD {v:,.0f}" if target in ("expected_loss", "stop_loss") else f"{v:.6g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2, help="number of countries")
    ap.add_argument("--scenario", default="adb-base", choices=list(SCENARIOS))
    ap.add_argument("--target", default="all", choices=["all", *TARGETS])
    ap.add_argument("--eps", type=float, default=0.01, help="target CI half-width, relative to the value")
    ap.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level")
    ap.add_argument("--shots", type=int, default=100, help="shots per IQAE round")
    ap.add_argument("--backend", default="oracle", choices=["oracle", "aer"])
    ap.add_argument("--shots-mode", action="store_true", help="aer: run literal measured shots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.backend == "aer" and args.n > 3:
        ap.error("the Aer backend simulates the full statevector; use --n <= 3")

    m = PortfolioModel(adb_portfolio(args.n, args.scenario))
    targets = TARGETS if args.target == "all" else (args.target,)
    print(f"{args.n} countries, scenario {args.scenario}, T = USD {m.T / 1e6:.0f}m, "
          f"eps = {args.eps:.1%} of the value, alpha = {args.alpha:g}, {args.shots} shots per round, "
          f"backend = {args.backend}\n")
    for t in targets:
        k = TARGETS.index(t)
        a = m.amplitude(k)
        A = oracle(m, t, "lookup" if args.backend == "aer" else "scalable")
        Q = grover_operator(A)
        if args.backend == "aer":
            sampler = CircuitSampler(A, Q, mode="shots" if args.shots_mode else "statevector")
        else:
            sampler = GroverModel(a)
        t0 = time.perf_counter()
        r = iqae(sampler, min(args.eps * a, 0.49), alpha=args.alpha, shots=args.shots, rng=args.seed)
        dt = time.perf_counter() - t0
        ga, gq = gate_counts(A), gate_counts(Q)
        s = m.scale(k)
        mc = (1 - a) / (a * (args.eps / 2) ** 2) if t not in ("expected_loss", "stop_loss") else np.nan
        print(f"{LABELS[t]}  ({t})")
        print(f"  exact      {fmt(t, m.exact[k])}")
        print(f"  IQAE       {fmt(t, r.estimate * s)}   CI [{fmt(t, r.ci[0] * s)}, {fmt(t, r.ci[1] * s)}]   "
              f"error {r.estimate / a - 1:+.3%}   ({dt:.1f} s)")
        print(f"  cost       {r.queries:,} oracle calls, {r.shots:,} shots, {r.rounds} rounds, "
              f"deepest circuit k = {r.k_max}"
              + (f"   (Monte Carlo, same 95% half-width: ~{mc:.2g} paths)" if np.isfinite(mc) else ""))
        print(f"  circuit    A: {ga['qubits']} qubits, {ga['cx']:,} CX, depth {ga['depth']:,}   "
              f"Q: {gq['qubits']} qubits, {gq['cx']:,} CX   deepest: {ga['cx'] + r.k_max * gq['cx']:,} CX\n")


if __name__ == "__main__":
    main()
