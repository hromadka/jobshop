"""
qiskit_oracle.py
================
Builds Qiskit 2.6 QuantumCircuit objects for the JSSP BBHT oracle.

For the five toy benchmarks (N_valid <= 4096) we use the explicit
phase-marking oracle: enumerate all marked JNS indices classically,
then mark each one with an X-MCZ-X pattern on the search register.
This is equivalent to the CPM oracle U_ΦT described in the paper
but requires no ancilla qubits — it is correct for small instances
and practical for simulation with AerSimulator.

The companion paper will give the full reversible arithmetic oracle
(JNS decode + CPM + threshold comparison) for large instances.

Requires: qiskit >= 2.6, qiskit-aer >= 0.15
Install:  pip install qiskit qiskit-aer

Circuit structure for each marked state x:
    for each bit position q:
        if bit q of x is 0: apply X(q)
    apply MCPhaseGate(pi) on all n qubits  (= multi-controlled-Z)
    for each bit position q:
        if bit q of x is 0: apply X(q)

Net effect: adds phase -1 to |x> and +1 to all others.
"""

from __future__ import annotations
import math
from typing import List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PhaseGate

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jssp_utils import JSSPInstance, phi_T


def build_oracle_circuit(
    inst: JSSPInstance,
    T: int,
    name: str = "U_PhiT",
) -> QuantumCircuit:
    """
    Build the explicit phase-marking oracle as a QuantumCircuit.

    Parameters
    ----------
    inst : JSSPInstance
    T    : makespan deadline
    name : circuit name (shown in circuit diagrams)

    Returns
    -------
    QuantumCircuit on n qubits that applies (-1)^{Phi_T(x)} to |x>.

    The circuit is unitary and self-inverse (applying it twice = identity).
    It contains no ancilla qubits: all marking is done in-place via
    phase kickback on the search register.
    """
    n = inst.n_perm
    N_valid = inst.N_valid

    # Identify all marked (feasible) states classically
    marked = [x for x in range(N_valid) if phi_T(x, inst, T)]

    qr = QuantumRegister(n, name="q")
    qc = QuantumCircuit(qr, name=name)

    if not marked:
        # No feasible states: oracle is identity
        return qc

    for x in marked:
        # Flip qubits where x has bit=0 so that |x> maps to |11...1>
        for q in range(n):
            if not ((x >> q) & 1):
                qc.x(qr[q])

        # Multi-controlled-Z: add phase -1 to |11...1>
        # Qiskit 2.x: use mcx + phase, or the built-in mcphase
        if n == 1:
            qc.z(qr[0])
        elif n == 2:
            qc.cz(qr[0], qr[1])
        else:
            # MCPhaseGate(pi) on n qubits = MCZ
            qc.append(
                PhaseGate(math.pi).control(n - 1),
                qr[:]
            )

        # Un-flip the same qubits
        for q in range(n):
            if not ((x >> q) & 1):
                qc.x(qr[q])

    qc.name = name
    return qc


def build_diffusion_circuit(n: int, name: str = "G") -> QuantumCircuit:
    """
    Grover diffusion operator G = 2|psi_0><psi_0| - I
    where |psi_0> = H^n |0>.

    Circuit: H^n, X^n, MCZ, X^n, H^n
    (standard textbook construction)

    Parameters
    ----------
    n    : number of qubits
    name : circuit name

    Returns
    -------
    QuantumCircuit on n qubits.
    """
    qr = QuantumRegister(n, name="q")
    qc = QuantumCircuit(qr, name=name)

    qc.h(qr[:])
    qc.x(qr[:])

    if n == 1:
        qc.z(qr[0])
    elif n == 2:
        qc.cz(qr[0], qr[1])
    else:
        qc.append(
            PhaseGate(math.pi).control(n - 1),
            qr[:]
        )

    qc.x(qr[:])
    qc.h(qr[:])

    qc.name = name
    return qc


def build_grover_iteration(
    inst: JSSPInstance,
    T: int,
) -> QuantumCircuit:
    """
    One full Grover iteration = oracle followed by diffusion.

    Returns a QuantumCircuit on n qubits.
    Useful for composing j iterations by circuit repetition.
    """
    n = inst.n_perm
    qr = QuantumRegister(n, name="q")
    qc = QuantumCircuit(qr, name="Grover_iter")

    oracle = build_oracle_circuit(inst, T, name="U_PhiT")
    diffusion = build_diffusion_circuit(n, name="G")

    qc.compose(oracle, qubits=qr[:], inplace=True)
    qc.compose(diffusion, qubits=qr[:], inplace=True)

    return qc


def build_bbht_round_circuit(
    inst: JSSPInstance,
    T: int,
    j: int,
) -> QuantumCircuit:
    """
    Build the full quantum circuit for one BBHT round with j Grover
    iterations:

        H^n |0>  ->  (oracle . diffusion)^j  ->  measure

    Parameters
    ----------
    inst : JSSPInstance
    T    : makespan deadline
    j    : number of Grover iterations (0 = uniform sampling only)

    Returns
    -------
    QuantumCircuit on n qubits with n classical bits for measurement.
    """
    n = inst.n_perm
    qr = QuantumRegister(n, name="q")
    from qiskit import ClassicalRegister
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr, name=f"BBHT_j{j}")

    # State preparation: H^n |0>
    qc.h(qr[:])

    if j > 0:
        grover_iter = build_grover_iteration(inst, T)
        for _ in range(j):
            qc.compose(grover_iter, qubits=qr[:], inplace=True)

    # Measure all qubits
    qc.measure(qr, cr)

    return qc


def oracle_gate_count(inst: JSSPInstance, T: int) -> dict:
    """
    Count the gates in the oracle circuit for diagnostic purposes.

    Returns a dict with gate type counts and total.
    """
    qc = build_oracle_circuit(inst, T)
    counts = qc.count_ops()
    total  = sum(counts.values())
    marked = sum(1 for x in range(inst.N_valid) if phi_T(x, inst, T))
    return {
        "n_qubits":   inst.n_perm,
        "N_valid":    inst.N_valid,
        "S":          marked,
        "gate_ops":   dict(counts),
        "total_gates": total,
    }
