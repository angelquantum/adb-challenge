"""Order-of-magnitude fault-tolerant resource estimate for an IQAE run.

Measured inputs: Clifford+T counts of the oracle A and of the Grover operator Q (circuits.ft_counts)
and, from the IQAE benchmark, the oracle calls and the deepest Grover power k_max for a target
accuracy. Everything else is an explicit assumption:

* arbitrary rotations are synthesised to precision delta with ~3 log2(1/delta) T gates
  (Ross & Selinger 2016); delta is set so the deepest circuit's total synthesis error is <= 1e-3;
* surface code at physical error rate p = 1e-3, threshold 1e-2: logical error per qubit per
  T-step p_L(d) = 0.1 (p / 0.01)^((d + 1) / 2); d is the smallest odd distance giving < 1%
  logical failure over the deepest circuit (logical qubits x T count);
* physical qubits = 2 d^2 per logical qubit x 1.5 routing overhead, plus 4 magic-state
  factories of ~ 15 x 12 d^2 qubits (a rough Litinski-style footprint);
* run time = total T count / logical T rate, at an optimistic 1 MHz and a conservative 10 kHz.
  The shots of one round can run in parallel on several machines; the estimate is serial.
"""
from __future__ import annotations

import numpy as np

T_RATES = (1e6, 1e4)


def rotation_t_cost(rotations: float, budget: float = 1e-3) -> int:
    delta = budget / max(rotations, 1.0)
    return int(np.ceil(3 * np.log2(1 / delta)))


def ft_estimate(qubits: int, t_A: int, rot_A: int, t_Q: int, rot_Q: int, queries: float,
                k_max: float, p_phys: float = 1e-3) -> dict:
    rot_deepest = rot_A + k_max * rot_Q
    t_rot = rotation_t_cost(rot_deepest)
    T_A = t_A + rot_A * t_rot
    T_Q = t_Q + rot_Q * t_rot
    deepest = T_A + k_max * T_Q
    total = queries * T_A                     # one oracle call ~ one A or A^dagger
    d = 3
    while qubits * deepest * 0.1 * (p_phys / 0.01) ** ((d + 1) / 2) > 0.01 and d < 99:
        d += 2
    out = dict(logical_qubits=qubits, t_per_rotation=t_rot, T_per_query=T_A, T_per_grover=T_Q,
               T_deepest_circuit=deepest, T_total=total, code_distance=d,
               physical_qubits=int(qubits * 2 * d**2 * 1.5 + 4 * 15 * 12 * d**2))
    for r in T_RATES:
        out[f"runtime_s_at_{int(r)}Hz"] = total / r
    return out
