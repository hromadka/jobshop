"""
qiskit_bbht.py
==============
Qiskit 2.6 implementation of Algorithm 1 (BBHT feasibility search)
and Algorithm 2 (binary search for optimal makespan).

Uses AerSimulator in statevector mode for exact simulation.
The BBHT logic is identical to quantum_sim.py; only the circuit
execution is replaced with Qiskit circuits + AerSimulator.

Requires: qiskit >= 2.6, qiskit-aer >= 0.15
Install:  pip install qiskit qiskit-aer

Usage
-----
    from jssp_utils import load_benchmark_file
    from qiskit_bbht import bbht_search_qiskit, find_optimal_makespan_qiskit

    inst = load_benchmark_file("benchmarks-toys/th02.txt")[0]
    x_hat, calls, rounds, stats = bbht_search_qiskit(inst, T=4, delta=0.1)
    print(f"Found JNS index {x_hat} in {calls} oracle calls")
"""

from __future__ import annotations
import math
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jssp_utils import JSSPInstance, phi_T, jns_decode, cpm_schedule
from qiskit_oracle import build_bbht_round_circuit


# ---------------------------------------------------------------------------
# Shared AerSimulator instance (statevector mode)
# ---------------------------------------------------------------------------

def _get_simulator() -> AerSimulator:
    """Return a statevector AerSimulator instance."""
    return AerSimulator(method="statevector")


def _run_circuit(
    qc: QuantumCircuit,
    simulator: AerSimulator,
    shots: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Transpile and run a circuit, returning measurement counts.

    Parameters
    ----------
    qc        : circuit with classical measurement(s)
    simulator : AerSimulator instance
    shots     : number of shots (1 for single-shot BBHT)
    seed      : simulator seed for reproducibility

    Returns
    -------
    dict mapping bitstring -> count, e.g. {"011": 1}
    """
    options = {}
    if seed is not None:
        options["seed_simulator"] = seed

    tqc = transpile(qc, simulator, optimization_level=0)
    job = simulator.run(tqc, shots=shots, **options)
    result = job.result()
    return result.get_counts()


def _counts_to_int(counts: Dict[str, int]) -> int:
    """
    Extract the single measured bitstring as an integer.

    Qiskit displays the most significant bit on the left (qubit 0 is rightmost).
    We convert to an integer consistently: bit 0 of the integer = qubit 0.
    """
    # With shots=1 there is exactly one entry
    bitstr = next(iter(counts))
    # Qiskit convention: bitstr[0] is the most significant qubit (qubit n-1)
    # The displayed binary string already has the correct integer significance.
    return int(bitstr.replace(" ", ""), 2)


# ---------------------------------------------------------------------------
# Algorithm 1: BBHT feasibility search (Qiskit version)
# ---------------------------------------------------------------------------

def bbht_search_qiskit(
    inst: JSSPInstance,
    T: int,
    delta: float = 0.1,
    lambda_: float = 6 / 5,
    seed: Optional[int] = None,
    verbose: bool = False,
    simulator: Optional[AerSimulator] = None,
) -> Tuple[Optional[int], int, int, Dict[str, Any]]:
    """
    Algorithm 1: Quantum JSSP Feasibility Search using Qiskit + AerSimulator.

    Exactly matches the pseudocode of Algorithm 1 in the paper:
      - Ceiling-based sampling: j ~ Uniform{0,...,ceil(m)-1}
      - Budget B = ceil(10 * sqrt(N) / delta)
      - Lambda = 6/5 (fixed)
      - Classical verification after each measurement
      - One-sided error <= delta on the Infeasible return

    Parameters
    ----------
    inst      : JSSPInstance
    T         : makespan deadline
    delta     : failure probability bound
    lambda_   : BBHT growth factor (default 6/5 per paper)
    seed      : base RNG seed for reproducibility
    verbose   : print round-by-round trace
    simulator : AerSimulator instance (created if None)

    Returns
    -------
    (x_hat, total_oracle_calls, total_rounds, stats)
      x_hat : satisfying JNS index, or None if INFEASIBLE declared
    """
    n       = inst.n_perm
    N       = 2 ** n
    N_valid = inst.N_valid

    # ── Preprocessing (paper Algorithm 1, lines 1-8) ──────────────────────
    T_lb = max(inst.lower_bound(), inst.job_lower_bound())
    if T < T_lb:
        return None, 0, 0, {"reason": "preprocessing: T < T_lb"}

    if T >= inst.P_max:
        return 0, 0, 0, {"reason": "trivial: T >= P_max"}

    # ── Budget (paper line 10) ─────────────────────────────────────────────
    D = 10
    B = math.ceil(D * math.sqrt(N) / delta)

    # ── Setup ─────────────────────────────────────────────────────────────
    if simulator is None:
        simulator = _get_simulator()

    rng = np.random.default_rng(seed)
    marked_set = set(x for x in range(N_valid) if phi_T(x, inst, T))
    S = len(marked_set)

    if verbose:
        print(f"  n={n} qubits, N={N}, N_valid={N_valid}, S={S}")
        print(f"  B={B} rounds, T={T}, T_lb={T_lb}, P_max={inst.P_max}")

    m               = 1.0
    total_oracle_calls = 0
    total_rounds    = 0
    round_log       = []

    # ── BBHT loop (paper lines 11-21) ─────────────────────────────────────
    for round_num in range(B):
        # Sample j (paper line 14)
        M = math.ceil(m)
        j = int(rng.integers(0, M))

        # Seed for this circuit run: deterministic from base seed + round
        circ_seed = None if seed is None else (seed * 10000 + round_num) % (2**31)

        # Build and run the BBHT round circuit
        qc = build_bbht_round_circuit(inst, T, j)
        counts = _run_circuit(qc, simulator, shots=1, seed=circ_seed)
        x_hat = _counts_to_int(counts)

        total_oracle_calls += j
        total_rounds       += 1

        if verbose:
            print(f"  Round {total_rounds}: m={m:.3f}, ceil(m)={M}, "
                  f"j={j}, measured x={x_hat} "
                  f"({'MARKED' if x_hat in marked_set else 'not marked'})")

        round_log.append({"round": total_rounds, "m": m, "j": j,
                          "x_hat": x_hat, "hit": x_hat in marked_set})

        # Classical verification (paper line 19)
        if phi_T(x_hat, inst, T):
            stats = {
                "S": S, "B": B, "N": N, "n": n,
                "success_round": total_rounds,
                "oracle_calls":  total_oracle_calls,
                "round_log":     round_log,
                "reason":        "success",
            }
            if verbose:
                print(f"  SUCCESS at round {total_rounds}, "
                      f"x_hat={x_hat}, oracle_calls={total_oracle_calls}")
            return x_hat, total_oracle_calls, total_rounds, stats

        # Grow m (paper line 20)
        m = min(lambda_ * m, math.sqrt(N))

    # Budget exhausted (paper line 22)
    stats = {
        "S": S, "B": B, "N": N, "n": n,
        "success_round": None,
        "oracle_calls":  total_oracle_calls,
        "round_log":     round_log,
        "reason":        "budget_exhausted",
    }
    if verbose:
        print(f"  INFEASIBLE declared after {B} rounds")
    return None, total_oracle_calls, total_rounds, stats


# ---------------------------------------------------------------------------
# Algorithm 2: Binary search for optimal makespan (Qiskit version)
# ---------------------------------------------------------------------------

def find_optimal_makespan_qiskit(
    inst: JSSPInstance,
    delta: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = False,
    simulator: Optional[AerSimulator] = None,
) -> Tuple[Optional[int], Optional[Dict], int, int, float]:
    """
    Algorithm 2: Binary search for optimal makespan using Qiskit BBHT.

    Parameters
    ----------
    inst      : JSSPInstance
    delta     : total failure probability bound (split across binary search)
    seed      : RNG seed
    verbose   : print trace
    simulator : AerSimulator instance (created if None)

    Returns
    -------
    (C_opt, schedule_info, total_oracle_calls, total_rounds, elapsed_s)
      C_opt : optimal makespan, or None if search failed
      schedule_info : dict with keys C_max, x, sequencing, schedule
    """
    if simulator is None:
        simulator = _get_simulator()

    T_lb = max(inst.lower_bound(), inst.job_lower_bound())
    T_ub = inst.P_max

    W    = T_ub - T_lb + 1
    Rbs  = math.floor(math.log2(W)) + 1
    dp   = delta / Rbs

    T_lo, T_hi = T_lb, T_ub
    best_T = None
    best_x = None
    total_calls  = 0
    total_rounds = 0

    if verbose:
        print(f"Algorithm 2: T in [{T_lb},{T_ub}], W={W}, "
              f"Rbs={Rbs}, delta'={dp:.4f}")

    t0 = time.perf_counter()

    while T_lo <= T_hi:
        T_mid = (T_lo + T_hi) // 2
        iter_seed = None if seed is None else seed + T_mid

        if verbose:
            print(f"  Testing T={T_mid}...")

        x_hat, calls, rounds, stats = bbht_search_qiskit(
            inst, T_mid, delta=dp,
            seed=iter_seed, verbose=False,
            simulator=simulator,
        )
        total_calls  += calls
        total_rounds += rounds

        if x_hat is not None:
            best_T = T_mid
            best_x = x_hat
            T_hi   = T_mid - 1
            if verbose:
                print(f"    FEASIBLE at T={T_mid} (x={x_hat})")
        else:
            T_lo = T_mid + 1
            if verbose:
                print(f"    INFEASIBLE at T={T_mid}")

    elapsed = time.perf_counter() - t0

    if best_x is None:
        return None, None, total_calls, total_rounds, elapsed

    job_sizes = [len(j) for j in inst.jobs]
    seq       = jns_decode(best_x, job_sizes)
    sched     = cpm_schedule(seq, inst)

    schedule_info = {
        "C_max":      best_T,
        "x":          best_x,
        "sequencing": seq,
        "schedule":   sched,
    }
    return best_T, schedule_info, total_calls, total_rounds, elapsed


# ---------------------------------------------------------------------------
# Amplitude distribution (for bar chart figures)
# ---------------------------------------------------------------------------

def compute_amplitude_distribution_qiskit(
    inst: JSSPInstance,
    T: int,
    j_iters: int,
    simulator: Optional[AerSimulator] = None,
    shots: int = 4096,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the measurement probability distribution after j Grover
    iterations at deadline T, using AerSimulator in shot mode.

    Used to generate the bar chart figures for Section 9.

    Parameters
    ----------
    inst    : JSSPInstance
    T       : makespan deadline
    j_iters : number of Grover iterations
    shots   : number of measurement shots for probability estimation
    seed    : simulator seed

    Returns
    -------
    (indices, probabilities)
      indices       : np.ndarray of shape (N_valid,), integer JNS indices
      probabilities : np.ndarray of shape (N_valid,), estimated probabilities
    """
    if simulator is None:
        simulator = _get_simulator()

    n = inst.n_perm
    N = 2 ** n

    qc = build_bbht_round_circuit(inst, T, j_iters)
    counts = _run_circuit(qc, simulator, shots=shots, seed=seed)

    # Convert counts to probability array over all 2^n states
    probs = np.zeros(N)
    for bitstr, count in counts.items():
        idx = int(bitstr.replace(" ", ""), 2)   # little-endian
        if idx < N:
            probs[idx] = count / shots

    # Return only the N_valid valid indices (rest are padding)
    indices = np.arange(inst.N_valid)
    return indices, probs[:inst.N_valid]
