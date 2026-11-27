"""Logical resource estimate for an IQAE run.

Measured inputs: Clifford+T counts of the oracle A and of the Grover operator Q (circuits.ft_counts)
and, from the IQAE benchmark, the oracle calls and the deepest Grover power k_max for a target
accuracy. Everything else is an explicit assumption:

* arbitrary rotations are synthesised to precision delta with ~3 log2(1/delta) T gates
  (Ross & Selinger 2016); delta is set so the deepest circuit's total synthesis error is <= 1e-3;
* run time = total T count / logical T rate, at an optimistic 1 MHz and a conservative 10 kHz.
  The shots of one round can run in parallel on several machines; the estimate is serial.
"""
from __future__ import annotations

import numpy as np

T_RATES = (1e6, 1e4)


def rotation_t_cost(rotations: float, budget: float = 1e-3) -> int:
    delta = budget / max(rotations, 1.0)
    return int(np.ceil(3 * np.log2(1 / delta)))


def logical_estimate(qubits: int, t_A: int, rot_A: int, t_Q: int, rot_Q: int, queries: float,
                k_max: float) -> dict:
    rot_deepest = rot_A + k_max * rot_Q
    t_rot = rotation_t_cost(rot_deepest)
    T_A = t_A + rot_A * t_rot
    T_Q = t_Q + rot_Q * t_rot
    deepest = T_A + k_max * T_Q
    total = queries * T_A  # one oracle call ~ one A or A^dagger
    out = dict(logical_qubits=qubits, t_per_rotation=t_rot, T_per_query=T_A, T_per_grover=T_Q,
               T_deepest_circuit=deepest, T_total=total)
    for r in T_RATES:
        out[f"runtime_s_at_{int(r)}Hz"] = total / r
    return out
