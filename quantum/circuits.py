"""Build and simulate PennyLane IQAE circuits.

The state preparation circuit loads the shared shock on four wires, followed
by four wires per country for the conditional earthquake and rain outcomes.
All registers put the least significant bit first.

Loss targets use a Fourier-basis adder to accumulate total loss before
encoding the objective. Joint payout and exhaustion use controlled X gates.
The lookup circuit encodes the objective directly and is limited to three
countries. Work wires start and end in zero and are excluded from the Grover
reflection.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from .portfolio import TARGETS, PortfolioModel

DEVICES = ("lightning.qubit", "default.qubit")
DEFAULT_DEVICE = DEVICES[0]
COUNT_BASIS = {"CNOT", "Hadamard", "PauliX", "PauliY", "PauliZ", "S", "T",
               "RX", "RY", "RZ", "PhaseShift", "GlobalPhase"}


@dataclass(frozen=True)
class Circuit:
    operations: tuple
    num_qubits: int
    objective: int
    work_wires: tuple[int, ...] = ()


def multiplexed_ry(angles, controls, target: int) -> list:
    """Decompose a uniformly controlled RY using a Gray-code CNOT sequence."""
    coeffs = np.array(angles, dtype=float, copy=True)
    # Walsh-Hadamard transform: rotation at Gray-code index g has signs (-1)^(g · c).
    for width in (1 << i for i in range(len(controls))):
        blocks = coeffs.reshape(-1, 2, width)
        first, second = blocks[:, 0].copy(), blocks[:, 1].copy()
        blocks[:, 0], blocks[:, 1] = first + second, first - second
    coeffs /= len(coeffs)
    ops = []
    for i in range(len(coeffs)):
        gray = i ^ (i >> 1)
        if abs(coeffs[gray]) > 1e-15:
            ops.append(qml.RY(float(coeffs[gray]), wires=target))
        if controls:
            j = (i + 1) % len(coeffs)
            changed = gray ^ j ^ (j >> 1)
            ops.append(qml.CNOT(wires=(controls[changed.bit_length() - 1], target)))
    return ops


def _cascade(ops: list, controls, targets, cond: np.ndarray) -> None:
    """Prepare sqrt(cond[c, x]) on `targets` for every basis state c of `controls`.

    cond: (2^len(controls), 2^len(targets)), rows sum to 1 (or 0 if never reached). Bits are
    loaded from the most significant down, each as a UCRY controlled by the controls plus the
    already-loaded higher bits.
    """
    nc, nt = len(controls), len(targets)
    xs = np.arange(2**nt)
    for j in reversed(range(nt)):
        higher = list(targets[j + 1:])
        ctrl = list(controls) + higher
        angles = np.zeros(2 ** len(ctrl))
        for c in range(2**nc):
            for h in range(2 ** len(higher)):
                sel = (xs >> (j + 1)) == h
                p_h = cond[c, sel].sum()
                if p_h <= 0:
                    continue
                p0 = cond[c, sel & (((xs >> j) & 1) == 0)].sum()
                angles[c + (h << nc)] = 2 * np.arccos(np.sqrt(np.clip(p0 / p_h, 0, 1)))
        ops.extend(multiplexed_ry(angles, ctrl, targets[j]))


def state_preparation(m: PortfolioModel) -> tuple[list, range, list[range]]:
    """Build the shared-shock preparation and return its wire registers."""
    lat = range(4)
    regs = [range(4 + 4 * i, 8 + 4 * i) for i in range(m.n)]
    ops = []
    latent = np.zeros((1, 16))
    table = np.zeros((m.n, 16, 16))
    for K, J in itertools.product(range(5), range(2)):
        latent[0, K | (J << 3)] = m.pz[2 * K + J]
        for q, r in itertools.product(range(5), range(2)):
            table[:, K | (J << 3), q | (r << 3)] = m.cond[:, 2 * K + J, 2 * q + r]
    _cascade(ops, [], lat, latent)
    for i, reg in enumerate(regs):
        _cascade(ops, lat, reg, table[i])
    return ops, lat, regs


def loss_weights(m: PortfolioModel) -> tuple[float, list[tuple[int, int]], int]:
    """(unit u, per-country (quarter-step weight, rain weight) in units of u, max total S)."""
    from .portfolio import loss_unit
    u = loss_unit(m.pf)
    w = [(round(c.eq_limit / 4 / u), round(c.rain_limit / u)) for c in m.pf.countries]
    return u, w, sum(4 * wq + wr for wq, wr in w)


def _add_constant(ops: list, S, c: int, ctrl=None) -> None:
    """Fourier-basis addition of the constant c to register S (in the Fourier basis)."""
    N = 2 ** len(S)
    for j, s in enumerate(S):
        angle = 2 * np.pi * ((c * 2**j) % N) / N
        if angle < 1e-12:
            continue
        if ctrl is None:
            ops.append(qml.PhaseShift(angle, wires=s))
        else:
            ops.append(qml.ControlledPhaseShift(angle, wires=(ctrl, s)))


def loss_register_oracle(m: PortfolioModel, target: str) -> Circuit:
    """A = A_prep + loss register S = L / u + exact RY(f(S)) on the objective."""
    u, w, s_max = loss_weights(m)
    b = int(np.ceil(np.log2(s_max + 1)))
    start = 4 + 4 * m.n
    S, obj = range(start, start + b), start + b
    ops, _, regs = state_preparation(m)
    ops.extend(qml.Hadamard(wires=s) for s in S)
    for reg, (wq, wr) in zip(regs, w, strict=True):
        for bit, weight in ((0, wq), (1, 2 * wq), (2, 4 * wq), (3, wr)):
            _add_constant(ops, S, weight, ctrl=reg[bit])
    ops.append(qml.adjoint(qml.QFT(wires=tuple(reversed(S)))))
    L = np.arange(2**b) * u
    if target == "expected_loss":
        v = L / m.total
    elif target == "exceedance":
        v = (L > m.T + 0.5).astype(float)
    elif target == "stop_loss":
        v = np.maximum(L - m.T, 0.0) / (m.total - m.T)
    else:
        raise ValueError(f"no loss-register objective for {target!r}")
    v[np.arange(2**b) > s_max] = 0.0
    ops.extend(multiplexed_ry(2 * np.arcsin(np.sqrt(np.clip(v, 0, 1))), S, obj))
    return Circuit(tuple(ops), obj + 1, obj)


def all_pay_oracle(m: PortfolioModel) -> Circuit:
    """Objective = AND_i OR(country i qubits): every country has some payout."""
    start = 4 + 4 * m.n
    flags, obj = range(start, start + m.n), start + m.n
    anc = tuple(range(obj + 1, obj + 1 + max(2, m.n - 2)))
    ops, _, regs = state_preparation(m)
    for reg, flag in zip(regs, flags, strict=True):
        ops.append(qml.MultiControlledX(wires=(*reg, flag), control_values=[0] * len(reg), work_wires=anc))
        ops.append(qml.X(wires=flag))
    ops.append(_mcx(flags, obj, anc))
    return Circuit(tuple(ops), obj + 1 + len(anc), obj, anc)


def all_exhausted_oracle(m: PortfolioModel) -> Circuit:
    """Objective = AND_i (q_i = 4 and r_i = 1). q <= 4 by construction, so q = 4 <=> bit 2."""
    obj = 4 + 4 * m.n
    anc = tuple(range(obj + 1, obj + 1 + max(1, 2 * m.n - 2)))
    ops, _, regs = state_preparation(m)
    ops.append(_mcx([q for reg in regs for q in (reg[2], reg[3])], obj, anc))
    return Circuit(tuple(ops), obj + 1 + len(anc), obj, anc)


def lookup_oracle(m: PortfolioModel, target: str) -> Circuit:
    """A = A_prep + RY(2 asin sqrt(f(x) / f_max)) multiplexed on all 4n country qubits (n <= 3)."""
    if m.n > 3:
        raise ValueError("the lookup oracle needs 2^(4n) rotations; use n <= 3")
    k = TARGETS.index(target)
    obj = 4 + 4 * m.n
    ops, _, regs = state_preparation(m)
    xs = np.array(list(itertools.product(range(10), repeat=m.n)))[:, ::-1]
    vals = m.values(xs)[:, k] / m.scale(k)
    idx = sum(((xs[:, i] // 2) | ((xs[:, i] % 2) << 3)) << (4 * i) for i in range(m.n))
    angles = np.zeros(2 ** (4 * m.n))
    angles[idx] = 2 * np.arcsin(np.sqrt(np.clip(vals, 0, 1)))
    ops.extend(multiplexed_ry(angles, [q for reg in regs for q in reg], obj))
    return Circuit(tuple(ops), obj + 1, obj)


def oracle(m: PortfolioModel, target: str, kind: str = "scalable") -> Circuit:
    """State preparation + objective for one of the five targets ("scalable" or "lookup")."""
    if kind == "lookup":
        return lookup_oracle(m, target)
    if target == "all_pay":
        return all_pay_oracle(m)
    if target == "all_exhausted":
        return all_exhausted_oracle(m)
    return loss_register_oracle(m, target)


def _mcx(controls, target, work_wires):
    if len(controls) == 1:
        return qml.CNOT(wires=(*controls, target))
    return qml.MultiControlledX(wires=(*controls, target), work_wires=work_wires)


def grover_operator(A: Circuit, aux: int | None = None) -> Circuit:
    """Q = A S_0 A† S_chi, up to a global phase."""
    aux = int(not A.work_wires) if aux is None else aux
    work = (*A.work_wires, *range(A.num_qubits, A.num_qubits + aux))
    reflect = [wire for wire in range(A.num_qubits) if wire not in A.work_wires]
    controls, target = reflect[:-1], reflect[-1]
    flips = [qml.X(wires=q) for q in reflect]
    ops = [qml.Z(wires=A.objective)]
    ops.extend(qml.adjoint(op, lazy=False) for op in reversed(A.operations))
    ops.extend(flips)
    ops.extend((qml.Hadamard(wires=target), _mcx(controls, target, work), qml.Hadamard(wires=target)))
    ops.extend(flips)
    ops.extend(A.operations)
    return Circuit(tuple(ops), A.num_qubits + aux, A.objective, work)


def _decompose(circuit: Circuit) -> tuple:
    tape = qml.tape.QuantumScript(circuit.operations)
    (decomposed,), _ = qml.transforms.decompose(tape, gate_set=COUNT_BASIS | {"Adjoint(S)", "Adjoint(T)"})
    return tuple(op for op in decomposed.operations if not isinstance(op, qml.GlobalPhase))


def _depth(ops) -> int:
    levels = {}
    for op in ops:
        level = 1 + max((levels.get(w, 0) for w in op.wires), default=0)
        levels.update(dict.fromkeys(op.wires, level))
    return max(levels.values(), default=0)


def gate_counts(circuit: Circuit) -> dict:
    """Count gates after PennyLane decomposition to CNOT and one-qubit operations."""
    ops = _decompose(circuit)
    cnots = [op for op in ops if isinstance(op, qml.CNOT)]
    return dict(qubits=circuit.num_qubits, cx=len(cnots), one_qubit=len(ops) - len(cnots),
                depth=_depth(ops), cx_depth=_depth(cnots))


def ft_counts(circuit: Circuit) -> dict:
    """Count explicit T gates and non-Clifford rotations in the decomposed circuit."""
    ops = _decompose(circuit)
    n_t = n_rot = 0
    for op in ops:
        if op.name in ("T", "Adjoint(T)"):
            n_t += 1
        elif op.name in ("RX", "RY", "RZ", "PhaseShift"):
            quarter_turns = float(op.parameters[0]) / (np.pi / 4)
            if abs(quarter_turns - round(quarter_turns)) < 1e-9:
                n_t += round(quarter_turns) % 2
            else:
                n_rot += 1
    return dict(qubits=circuit.num_qubits, cx=sum(isinstance(op, qml.CNOT) for op in ops),
                t_gates=n_t, rotations=n_rot, depth=_depth(ops))


def _execute(device, operations, measurement, shots=None):
    script = qml.tape.QuantumScript(operations, [measurement], shots=shots)
    return qml.execute((script,), device, diff_method=None, interface=None)[0]


def objective_probability(A: Circuit, device: str = DEFAULT_DEVICE) -> float:
    """P(objective = 1) after A on a noiseless PennyLane simulator."""
    simulator = qml.device(device, wires=range(A.num_qubits))
    return float(_execute(simulator, A.operations, qml.probs(wires=A.objective))[1])


class CircuitSampler:
    """Sample an IQAE circuit using analytic probabilities or finite shots."""

    def __init__(self, A: Circuit, Q: Circuit | None = None, mode: str = "statevector",
                 chunk: int = 16, device: str = DEFAULT_DEVICE):
        if mode not in ("statevector", "shots"):
            raise ValueError(mode)
        if chunk < 1:
            raise ValueError("chunk must be positive")
        self.A = A
        self.Q = grover_operator(A) if Q is None else Q
        self.wires = range(self.Q.num_qubits)
        self.sim = qml.device(device, wires=self.wires)
        self.device, self.mode, self.chunk = device, mode, chunk
        self.calls: list[tuple[int, int]] = []
        self._p: list[float] = []
        self._psi = None

    def _advance(self) -> None:
        for _ in range(self.chunk):
            ops = (self.A.operations if self._psi is None
                   else (qml.StatePrep(self._psi, wires=self.wires), *self.Q.operations))
            self._psi = _execute(self.sim, ops, qml.state())
            probabilities = np.abs(self._psi.reshape((2,) * len(self.wires))) ** 2
            p = np.take(probabilities, 1, axis=self.A.objective).sum()
            self._p.append(float(np.clip(p, 0.0, 1.0)))

    def probability(self, k: int) -> float:
        """P(objective = 1) after Q^k A, from the circuit."""
        while len(self._p) <= k:
            self._advance()
        return self._p[k]

    def __call__(self, k: int, shots: int, rng: np.random.Generator) -> int:
        self.calls.append((k, shots))
        if self.mode == "statevector":
            return int(rng.binomial(shots, self.probability(k)))
        device = qml.device(self.device, wires=self.wires, seed=int(rng.integers(2**31)))
        samples = _execute(device, self.A.operations + self.Q.operations * k,
                           qml.sample(wires=self.A.objective), shots=shots)
        return int(np.sum(samples))
