"""Qiskit circuits for IQAE on the n-country model.

State preparation (shared-shock loading). Registers, Qiskit little-endian:
    z    4 qubits: K (3 bits, 0..4 regional quakes) | J (1 bit, regional rain fired)
    c_i  4 qubits per country: q_i (3 bits, 0..4 quarter steps) | r_i (1 bit, rain trigger)
A_prep|0> = sum_z sqrt(P(z)) |z> (x)_i sum_x sqrt(P_i(x | z)) |x>_i, from uniformly controlled
RY cascades. Each country is a 4-qubit load multiplexed on the 4 shock qubits, so the cost is
linear in n; nothing of size 10^n is ever tabulated.

Objectives (the objective qubit is the register "obj"; registers named "anc" or "aux" are
auxiliary qubits returned to |0> and are left out of the Grover reflection):
    loss register  S = sum_i L_i / u accumulated with Fourier-basis constant adders
                   (Draper 2000), then one uniformly controlled RY encodes f(S) exactly:
                   E[L], P(L > T), E[(L - T)+]. O(n log n) two-qubit gates.
    all pay        per-country OR flags, then a multi-controlled X
    all exhausted  multi-controlled X on (q_i = 4, r_i = 1) for every country
    lookup         RY multiplexed on every country qubit: any target, exact, 2^(4n) CX.
                   Small n only; used for the end-to-end runs at n = 2.

grover_operator builds Q = A S_0 A^dagger S_chi; CircuitSampler executes Q^k A on Qiskit Aer
and plugs into iqae.iqae as a backend.
"""
from __future__ import annotations

import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UCRYGate

from .portfolio import TARGETS, PortfolioModel

AUX_NAMES = ("anc", "aux")
BASIS = ["cx", "u"]
FT_BASIS = ["cx", "rz", "sx", "x"]


# ---------------------------------------------------------------- state preparation

def _cascade(qc: QuantumCircuit, controls, targets, cond: np.ndarray) -> None:
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
        if ctrl:
            qc.append(UCRYGate(list(angles)), [targets[j], *ctrl])
        else:
            qc.ry(angles[0], targets[j])


def state_preparation(m: PortfolioModel) -> tuple[QuantumCircuit, QuantumRegister, list[QuantumRegister]]:
    """A_prep for the shared-shock factorisation: (circuit, shock register, country registers)."""
    lat = QuantumRegister(4, "z")
    regs = [QuantumRegister(4, f"c{i}") for i in range(m.n)]
    qc = QuantumCircuit(lat, *regs, name="A_prep")
    latent = np.zeros((1, 16))
    table = np.zeros((m.n, 16, 16))
    for K, J in itertools.product(range(5), range(2)):
        latent[0, K | (J << 3)] = m.pz[2 * K + J]
        for q, r in itertools.product(range(5), range(2)):
            table[:, K | (J << 3), q | (r << 3)] = m.cond[:, 2 * K + J, 2 * q + r]
    _cascade(qc, [], list(lat), latent)
    for i, reg in enumerate(regs):
        _cascade(qc, list(lat), list(reg), table[i])
    return qc, lat, regs


def _with_prep(m: PortfolioModel, *extra: QuantumRegister) -> tuple[QuantumCircuit, list[QuantumRegister]]:
    prep, lat, regs = state_preparation(m)
    qc = QuantumCircuit(lat, *regs, *extra, name="A")
    qc.compose(prep, qubits=list(lat) + [q for r in regs for q in r], inplace=True)
    return qc, regs


# ---------------------------------------------------------------- objectives

def loss_weights(m: PortfolioModel) -> tuple[float, list[tuple[int, int]], int]:
    """(unit u, per-country (quarter-step weight, rain weight) in units of u, max total S)."""
    from .portfolio import loss_unit
    u = loss_unit(m.pf)
    w = [(round(c.eq_limit / 4 / u), round(c.rain_limit / u)) for c in m.pf.countries]
    return u, w, sum(4 * wq + wr for wq, wr in w)


def _add_constant(qc: QuantumCircuit, S, c: int, ctrl=None) -> None:
    """Fourier-basis addition of the constant c to register S (in the Fourier basis)."""
    N = 2 ** len(S)
    for j, s in enumerate(S):
        angle = 2 * np.pi * ((c * 2**j) % N) / N
        if angle < 1e-12:
            continue
        if ctrl is None:
            qc.p(angle, s)
        else:
            qc.cp(angle, ctrl, s)


def loss_register_oracle(m: PortfolioModel, target: str) -> QuantumCircuit:
    """A = A_prep + loss register S = L / u + exact RY(f(S)) on the objective."""
    u, w, s_max = loss_weights(m)
    b = int(np.ceil(np.log2(s_max + 1)))
    S, obj = QuantumRegister(b, "S"), QuantumRegister(1, "obj")
    qc, regs = _with_prep(m, S, obj)
    qc.h(S)                                       # |0> -> QFT|0>
    for reg, (wq, wr) in zip(regs, w, strict=True):
        for bit, weight in ((0, wq), (1, 2 * wq), (2, 4 * wq), (3, wr)):
            _add_constant(qc, S, weight, ctrl=reg[bit])
    qc.append(QFTGate(b).inverse(), list(S))
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
    qc.append(UCRYGate(list(2 * np.arcsin(np.sqrt(np.clip(v, 0, 1))))), [obj[0], *S])
    return qc


def all_pay_oracle(m: PortfolioModel) -> QuantumCircuit:
    """Objective = AND_i OR(country i qubits): every country has some payout."""
    flags, obj = QuantumRegister(m.n, "pay"), QuantumRegister(1, "obj")
    anc = QuantumRegister(max(2, m.n - 2), "anc")
    qc, regs = _with_prep(m, flags, obj, anc)
    for reg, f in zip(regs, flags, strict=True):
        qc.x(reg)
        qc.mcx(list(reg), f)                     # f = NOT(all bits 0)
        qc.x(reg)
        qc.x(f)
    qc.mcx(list(flags), obj[0])
    return qc


def all_exhausted_oracle(m: PortfolioModel) -> QuantumCircuit:
    """Objective = AND_i (q_i = 4 and r_i = 1). q <= 4 by construction, so q = 4 <=> bit 2."""
    obj = QuantumRegister(1, "obj")
    anc = QuantumRegister(max(1, 2 * m.n - 2), "anc")
    qc, regs = _with_prep(m, obj, anc)
    qc.mcx([q for r in regs for q in (r[2], r[3])], obj[0])
    return qc


def lookup_oracle(m: PortfolioModel, target: str) -> QuantumCircuit:
    """A = A_prep + RY(2 asin sqrt(f(x) / f_max)) multiplexed on all 4n country qubits (n <= 3)."""
    if m.n > 3:
        raise ValueError("the lookup oracle needs 2^(4n) rotations; use n <= 3")
    k = TARGETS.index(target)
    obj = QuantumRegister(1, "obj")
    qc, regs = _with_prep(m, obj)
    xs = np.array(list(itertools.product(range(10), repeat=m.n)))[:, ::-1]   # country 0 fastest
    vals = m.values(xs)[:, k] / m.scale(k)
    idx = sum(((xs[:, i] // 2) | ((xs[:, i] % 2) << 3)) << (4 * i) for i in range(m.n))
    angles = np.zeros(2 ** (4 * m.n))
    angles[idx] = 2 * np.arcsin(np.sqrt(np.clip(vals, 0, 1)))
    qc.append(UCRYGate(list(angles)), [obj[0], *(q for r in regs for q in r)])
    return qc


def oracle(m: PortfolioModel, target: str, kind: str = "scalable") -> QuantumCircuit:
    """State preparation + objective for one of the five targets ("scalable" or "lookup")."""
    if kind == "lookup":
        return lookup_oracle(m, target)
    if target == "all_pay":
        return all_pay_oracle(m)
    if target == "all_exhausted":
        return all_exhausted_oracle(m)
    return loss_register_oracle(m, target)


# ---------------------------------------------------------------- Grover operator

def objective_index(qc: QuantumCircuit) -> int:
    reg = next(r for r in qc.qregs if r.name == "obj")
    return qc.find_bit(reg[0]).index


def reflection_qubits(qc: QuantumCircuit) -> list[int]:
    aux = {q for r in qc.qregs if r.name in AUX_NAMES for q in r}
    return [i for i, q in enumerate(qc.qubits) if q not in aux]


def grover_operator(A: QuantumCircuit, aux: int | None = None) -> QuantumCircuit:
    """Q = A S_0 A^dagger S_chi, up to a global phase (irrelevant: Q is never controlled).

    S_chi = Z on the objective; S_0 = I - 2|0><0| on the reflection qubits, as X . MCZ . X.
    `aux` clean qubits are added for the MCX synthesis when A has no auxiliary register
    (default 1: linear gate count at the cost of one qubit).
    """
    has_aux = any(r.name in AUX_NAMES for r in A.qregs)
    aux = (0 if has_aux else 1) if aux is None else aux
    Q = QuantumCircuit(*A.qregs, *([QuantumRegister(aux, "aux")] if aux else []), name="Q")
    nA = A.num_qubits
    refl = reflection_qubits(A)
    tgt, ctrl = refl[-1], refl[:-1]
    Q.z(objective_index(A))
    Q.compose(A.inverse(), qubits=range(nA), inplace=True)
    Q.x(refl)
    Q.h(tgt)
    Q.mcx(ctrl, tgt)
    Q.h(tgt)
    Q.x(refl)
    Q.compose(A, qubits=range(nA), inplace=True)
    return Q


# ---------------------------------------------------------------- resource counts

def gate_counts(qc: QuantumCircuit, opt: int = 1) -> dict:
    """Qubits, two-qubit gates, one-qubit gates and depth after transpiling to {CX, U}."""
    t = transpile(qc, basis_gates=BASIS, optimization_level=opt, seed_transpiler=0)
    ops = t.count_ops()
    return dict(qubits=t.num_qubits, cx=int(ops.get("cx", 0)), one_qubit=int(ops.get("u", 0)),
                depth=int(t.depth()), cx_depth=int(t.depth(lambda ins: ins.operation.num_qubits == 2)))


def ft_counts(qc: QuantumCircuit, opt: int = 1) -> dict:
    """Clifford+T bookkeeping from a {CX, RZ, SX, X} transpilation.

    RZ angles that are multiples of pi/2 are Clifford, odd multiples of pi/4 are one T gate,
    anything else is an arbitrary rotation that must be synthesised (resources.py).
    """
    t = transpile(qc, basis_gates=FT_BASIS, optimization_level=opt, seed_transpiler=0)
    n_t = n_rot = 0
    for ins in t.data:
        if ins.operation.name != "rz":
            continue
        m4 = (float(ins.operation.params[0]) % (2 * np.pi)) / (np.pi / 4)
        if abs(m4 - round(m4)) < 1e-9:
            n_t += round(m4) % 2
        else:
            n_rot += 1
    return dict(qubits=t.num_qubits, cx=int(t.count_ops().get("cx", 0)), t_gates=n_t,
                rotations=n_rot, depth=int(t.depth()))


# ---------------------------------------------------------------- Aer backend for IQAE

def objective_probability(A: QuantumCircuit, precision: str = "double") -> float:
    """P(objective = 1) after A, from an Aer statevector simulation of the transpiled circuit."""
    from qiskit_aer import AerSimulator

    qc = transpile(A, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
    qc.save_probabilities([objective_index(A)])
    sim = AerSimulator(method="statevector", precision=precision)
    return float(sim.run(qc).result().data(0)["probabilities"][1])


class CircuitSampler:
    """IQAE backend that executes the actual circuits Q^k A with Qiskit Aer.

    mode="statevector": P(objective = 1) after Q^k A is read off the simulated statevector,
        which is advanced one Grover step at a time and cached, and the shots are drawn from
        it. Statistically identical to measuring the circuit `shots` times on a noiseless
        device, and fast for deep circuits.
    mode="shots": builds Q^k A with a measurement and runs it with `shots` shots, optionally
        under a noise model (trajectory simulation). Literal, but the circuit grows with k.
    """

    def __init__(self, A: QuantumCircuit, Q: QuantumCircuit | None = None, mode: str = "statevector",
                 noise_model=None, chunk: int = 16):
        from qiskit_aer import AerSimulator

        if mode not in ("statevector", "shots"):
            raise ValueError(mode)
        Q = grover_operator(A) if Q is None else Q
        self.nA, self.nq = A.num_qubits, Q.num_qubits
        self.obj = objective_index(A)
        self.At = transpile(A, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
        self.Qt = transpile(Q, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
        self.mode, self.chunk = mode, chunk
        self.sim = AerSimulator(method="statevector", noise_model=noise_model)
        self.calls: list[tuple[int, int]] = []
        self._p: list[float] = []
        self._psi = None

    def _advance(self) -> None:
        """Simulate the next `chunk` Grover steps from the cached statevector."""
        qc = QuantumCircuit(self.nq)
        labels = []
        if self._psi is None:
            qc.compose(self.At, qubits=range(self.nA), inplace=True)
            qc.save_probabilities([self.obj], label="s0")
            labels.append("s0")
        else:
            qc.set_statevector(self._psi)
        for _ in range(self.chunk):
            qc.compose(self.Qt, inplace=True)
            labels.append(f"s{len(labels)}")
            qc.save_probabilities([self.obj], label=labels[-1])
        qc.save_statevector()
        data = self.sim.run(qc).result().data(0)
        self._p += [float(data[lab][1]) for lab in labels]
        self._psi = data["statevector"]

    def probability(self, k: int) -> float:
        """P(objective = 1) after Q^k A, from the circuit (ideal simulation)."""
        while len(self._p) <= k:
            self._advance()
        return self._p[k]

    def __call__(self, k: int, shots: int, rng: np.random.Generator) -> int:
        self.calls.append((k, shots))
        if self.mode == "statevector":
            return int(rng.binomial(shots, self.probability(k)))
        qc = QuantumCircuit(self.nq, 1)
        qc.compose(self.At, qubits=range(self.nA), inplace=True)
        for _ in range(k):
            qc.compose(self.Qt, inplace=True)
        qc.measure(self.obj, 0)
        res = self.sim.run(qc, shots=shots, seed_simulator=int(rng.integers(2**31))).result()
        return int(res.get_counts().get("1", 0))
