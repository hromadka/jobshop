"""
Constraint-Decomposed JSSP Quantum Algorithm
Based on Schmid et al. (arXiv:2401.16381)

Target instance:
  2 jobs, 2 machines, 2 operations per job
  Job 0: [op(machine=0, time=1), op(machine=1, time=2)]
  Job 1: [op(machine=1, time=1), op(machine=0, time=2)]
  Optimal makespan: 4
"""

import math
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    job_id: int
    op_index: int       # index within job
    machine: int
    proc_time: int

    def __repr__(self):
        return f"O({self.job_id},{self.op_index},m{self.machine},t{self.proc_time})"


def build_operations(jobs: List[List[Tuple[int, int]]]) -> List[Operation]:
    """Flatten job descriptions into a list of Operation objects."""
    ops = []
    for job_id, job in enumerate(jobs):
        for op_idx, (machine, proc_time) in enumerate(job):
            ops.append(Operation(job_id, op_idx, machine, proc_time))
    return ops


# ---------------------------------------------------------------------------
# JNS (Job Number System) — classical encode / decode
# ---------------------------------------------------------------------------

def _count_valid_sequences(job_sizes: List[int]) -> int:
    """Multinomial coefficient: N_op! / prod(|J_i|!)"""
    n_op = sum(job_sizes)
    result = math.factorial(n_op)
    for s in job_sizes:
        result //= math.factorial(s)
    return result


def _jns_bases(job_sizes: List[int]) -> List[int]:
    """
    Mixed-radix bases for the JNS inversion vector.
    For each operation position j (0-indexed across flattened ops),
    the maximum value b[j] can take is the number of higher-indexed ops
    that could appear before it in a valid sequence.

    Following Schmid et al.: the digit for the k-th operation of job i
    (0-indexed within job) ranges over [b_prev, N_remaining - (remaining_in_job - 1)]
    but we store the *relative* count, so the base (number of choices) for
    each operation = (total remaining positions) - (remaining ops in same job - 1)
    minus the value already consumed by the previous intra-job op.

    Simpler implementation: enumerate all valid inversion vectors and map
    index -> vector directly for small instances, but we implement the
    general mixed-radix scheme here.

    The inversion vector b[j] for operation j counts how many of the
    higher-global-index operations appear *before* j in the permutation.
    Job precedence constrains: for consecutive ops in same job (j, j+1),
    b[j+1] >= b[j] is NOT the right constraint per Schmid — the actual
    constraint is that the absolute position of j is less than j+1.

    We use a direct enumeration approach for correctness, mapping index to
    valid permutation via factorial number system filtered by job precedence.
    """
    pass  # implemented below via enumeration


def _enumerate_valid_permutations(ops: List[Operation]) -> List[List[int]]:
    """
    Enumerate all permutations of operation indices that satisfy
    job precedence (within each job, earlier ops must come first).
    Returns list of permutations (each a list of op indices).
    """
    n = len(ops)
    # Group op indices by job, in order
    jobs_ops: Dict[int, List[int]] = {}
    for i, op in enumerate(ops):
        jobs_ops.setdefault(op.job_id, []).append(i)

    valid = []
    for perm in itertools.permutations(range(n)):
        # Check job precedence: for each job, ops must appear in original order
        ok = True
        for job_op_list in jobs_ops.values():
            # Find positions of each op in this permutation
            positions = [perm.index(idx) for idx in job_op_list]
            if positions != sorted(positions):
                ok = False
                break
        if ok:
            valid.append(list(perm))
    return valid


def jns_decode(index: int, ops: List[Operation]) -> List[int]:
    """
    Map integer index in [0, N_valid) to a valid permutation of op indices.
    Uses the enumeration-based approach (correct for small instances).
    """
    valid_perms = _enumerate_valid_permutations(ops)
    if index < 0 or index >= len(valid_perms):
        raise ValueError(f"Index {index} out of range [0, {len(valid_perms)})")
    return valid_perms[index]


def jns_encode(permutation: List[int], ops: List[Operation]) -> int:
    """
    Map a valid permutation back to its integer index.
    """
    valid_perms = _enumerate_valid_permutations(ops)
    for i, p in enumerate(valid_perms):
        if p == permutation:
            return i
    raise ValueError(f"Permutation {permutation} not found in valid permutations")


def validate_jns(ops: List[Operation]):
    """Verify jns_encode and jns_decode are inverses on all valid permutations."""
    valid_perms = _enumerate_valid_permutations(ops)
    n = len(valid_perms)
    print(f"  Total valid permutations: {n}")
    for i, perm in enumerate(valid_perms):
        decoded = jns_decode(i, ops)
        assert decoded == perm, f"Decode mismatch at index {i}"
        encoded = jns_encode(perm, ops)
        assert encoded == i, f"Encode mismatch for perm {perm}"
    print("  JNS encode/decode validation PASSED")


# ---------------------------------------------------------------------------
# Schedule evaluation (classical cost function)
# ---------------------------------------------------------------------------

def evaluate_schedule(permutation: List[int], ops: List[Operation]) -> Tuple[int, Dict[int, int]]:
    """
    Given a permutation of operation indices, compute earliest valid start
    times respecting job precedence and machine non-overlap.
    Returns (makespan, {op_index: start_time}).
    """
    n = len(ops)
    start_time = {}
    finish_time = {}

    # Track: last finish time on each machine, last finish time per job
    machine_free = {}   # machine -> time when it becomes free
    job_free = {}       # job_id -> time when predecessor finishes

    for op_idx in permutation:
        op = ops[op_idx]
        m = op.machine
        j = op.job_id

        earliest = max(
            machine_free.get(m, 0),
            job_free.get(j, 0)
        )
        start_time[op_idx] = earliest
        ft = earliest + op.proc_time
        finish_time[op_idx] = ft
        machine_free[m] = ft
        job_free[j] = ft

    makespan = max(finish_time.values())
    return makespan, start_time


# ---------------------------------------------------------------------------
# Stage 1 — Classical constraint propagation
# ---------------------------------------------------------------------------

def compute_feasible_windows(
    ops: List[Operation],
    T_max: int
) -> Dict[int, Tuple[int, int]]:
    """
    For each operation op_idx, compute [earliest_start, latest_finish].
    earliest_start[i,j] = sum of proc times of predecessors in job j.
    latest_finish[i,j]  = T_max - sum of proc times of successors in job j.
    Returns {op_idx: (earliest_start, latest_finish)}.
    """
    # Group ops by job in order
    jobs_ops: Dict[int, List[int]] = {}
    for i, op in enumerate(ops):
        jobs_ops.setdefault(op.job_id, []).append(i)

    feasible = {}
    for job_id, op_list in jobs_ops.items():
        running = 0
        for idx in op_list:
            feasible[idx] = [running, None]
            running += ops[idx].proc_time
        # latest_finish: work backwards
        running = 0
        for idx in reversed(op_list):
            running += ops[idx].proc_time
            feasible[idx][1] = T_max - (running - ops[idx].proc_time)

    return {k: tuple(v) for k, v in feasible.items()}


def group_operations_by_machine(ops: List[Operation]) -> Dict[int, List[int]]:
    """Returns {machine_id: [op_indices...]}"""
    result: Dict[int, List[int]] = {}
    for i, op in enumerate(ops):
        result.setdefault(op.machine, []).append(i)
    return result


# ---------------------------------------------------------------------------
# Per-machine ordering enumeration
# ---------------------------------------------------------------------------

def enumerate_machine_orderings(machine_ops: List[int]) -> List[List[int]]:
    """All permutations of op indices on a given machine."""
    return [list(p) for p in itertools.permutations(machine_ops)]


def check_machine_ordering_feasible(
    ordering: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
) -> Tuple[bool, Optional[Dict[int, int]]]:
    """
    Given an ordering of ops on one machine, compute a valid assignment of
    start times (each op starts as early as possible given the ordering and
    feasible windows).

    Returns (feasible, {op_idx: start_time}) or (False, None).
    The ordering defines: op[i] must finish before op[i+1] starts.
    We also need to respect feasible_windows[op_idx] = (ES, LF).
    """
    start_times = {}
    current_time = 0

    for op_idx in ordering:
        op = ops[op_idx]
        es, lf = feasible_windows[op_idx]
        start = max(current_time, es)
        finish = start + op.proc_time
        if start < es or finish > lf:
            return False, None
        start_times[op_idx] = start
        current_time = finish

    return True, start_times


def count_valid_machine_orderings(
    machine_ops: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
) -> Tuple[int, List[List[int]]]:
    """Returns (count, list_of_valid_orderings)."""
    valid = []
    for ordering in enumerate_machine_orderings(machine_ops):
        feasible, _ = check_machine_ordering_feasible(ordering, ops, feasible_windows)
        if feasible:
            valid.append(ordering)
    return len(valid), valid


# ---------------------------------------------------------------------------
# Stage 2 — Grover circuit per machine
# ---------------------------------------------------------------------------

def build_grover_circuit_2ops(
    machine_id: int,
    machine_ops: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
    n_iterations: int,
) -> QuantumCircuit:
    """
    Build a full Grover circuit for a machine with exactly 2 operations.
    With 2 ops there are 2! = 2 orderings, requiring 1 qubit.
    Ordering 0: machine_ops[0] then machine_ops[1]
    Ordering 1: machine_ops[1] then machine_ops[0]

    Oracle marks orderings that are feasible (non-overlapping within windows).
    """
    assert len(machine_ops) == 2, "This builder is for 2-op machines only"

    n_input = 1  # ceil(log2(2)) = 1

    # Determine which orderings are marked
    orderings = enumerate_machine_orderings(machine_ops)  # [[a,b], [b,a]]
    marked = []
    for i, ordering in enumerate(orderings):
        feasible, _ = check_machine_ordering_feasible(ordering, ops, feasible_windows)
        if feasible:
            marked.append(i)

    print(f"  Machine {machine_id}: orderings={orderings}, marked indices={marked}")

    # Build circuit: 1 input qubit + 1 phase qubit
    qr_input = QuantumRegister(n_input, name='inp')
    qr_phase = QuantumRegister(1, name='phase')
    cr = ClassicalRegister(n_input, name='c')
    qc = QuantumCircuit(qr_input, qr_phase, cr)

    # Initialize phase qubit to |->
    qc.x(qr_phase[0])
    qc.h(qr_phase[0])

    # Initialize input register to uniform superposition
    qc.h(qr_input[0])

    # Apply Grover iterations
    for iteration in range(n_iterations):
        # Oracle
        _apply_oracle_2ops(qc, qr_input, qr_phase, marked)
        # Diffusion
        _apply_diffusion(qc, qr_input)

    # Measure input register
    qc.measure(qr_input, cr)
    return qc


def _apply_oracle_2ops(
    qc: QuantumCircuit,
    qr_input: QuantumRegister,
    qr_phase: QuantumRegister,
    marked: List[int],
):
    """
    Oracle for 1-qubit input: apply X to phase qubit for each marked index.
    Index 0 = |0>, index 1 = |1>.
    Phase kickback: phase qubit is in |-> so X flips the phase.
    """
    for m in marked:
        if m == 0:
            # Mark |0>: apply controlled-X with control=|0> (add X before/after ctrl)
            qc.x(qr_input[0])
            qc.cx(qr_input[0], qr_phase[0])
            qc.x(qr_input[0])
        else:
            # Mark |1>: standard controlled-X
            qc.cx(qr_input[0], qr_phase[0])


def _apply_diffusion(qc: QuantumCircuit, qr_input: QuantumRegister):
    """
    Grover diffusion operator: H X (multi-ctrl Z) X H on input register.
    For 1 qubit: H X Z X H = H (2|0><0|-I) H
    """
    n = len(qr_input)
    qc.h(qr_input)
    qc.x(qr_input)
    if n == 1:
        qc.z(qr_input[0])
    else:
        # Multi-controlled Z: use H on last qubit, MCX, H on last qubit
        qc.h(qr_input[n - 1])
        qc.mcx(list(qr_input[:n - 1]), qr_input[n - 1])
        qc.h(qr_input[n - 1])
    qc.x(qr_input)
    qc.h(qr_input)


def build_grover_circuit_general(
    machine_id: int,
    machine_ops: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
    n_iterations: int,
) -> QuantumCircuit:
    """
    General Grover circuit for machines with k operations.
    k! orderings, ceil(log2(k!)) input qubits.
    Uses lookup-table oracle: for each marked ordering index, apply
    a phase flip using multi-controlled gates.
    """
    k = len(machine_ops)
    n_orderings = math.factorial(k)
    n_input = math.ceil(math.log2(n_orderings)) if n_orderings > 1 else 1

    orderings = enumerate_machine_orderings(machine_ops)
    marked = []
    for i, ordering in enumerate(orderings):
        feasible, _ = check_machine_ordering_feasible(ordering, ops, feasible_windows)
        if feasible:
            marked.append(i)

    print(f"  Machine {machine_id}: {k} ops, {n_orderings} orderings, "
          f"{n_input} qubits, marked={marked}")

    qr_input = QuantumRegister(n_input, name='inp')
    qr_phase = QuantumRegister(1, name='phase')
    cr = ClassicalRegister(n_input, name='c')
    qc = QuantumCircuit(qr_input, qr_phase, cr)

    # |->
    qc.x(qr_phase[0])
    qc.h(qr_phase[0])

    # Uniform superposition over input
    qc.h(qr_input)

    for _ in range(n_iterations):
        _apply_oracle_general(qc, qr_input, qr_phase, marked, n_input, n_orderings)
        _apply_diffusion(qc, qr_input)

    qc.measure(qr_input, cr)
    return qc


def _apply_oracle_general(
    qc: QuantumCircuit,
    qr_input: QuantumRegister,
    qr_phase: QuantumRegister,
    marked: List[int],
    n_input: int,
    n_orderings: int,
):
    """
    For each marked index m, apply a phase flip when input == m.
    Implemented as: flip qubits where bit=0, MCX to phase, flip back.
    """
    for m in marked:
        if m >= n_orderings:
            continue
        # Binary representation of m (LSB first for Qiskit convention)
        bits = [(m >> i) & 1 for i in range(n_input)]
        # X on qubits where bit is 0 (to convert 0-bits to 1 for ctrl)
        zero_bits = [i for i, b in enumerate(bits) if b == 0]
        if zero_bits:
            qc.x([qr_input[i] for i in zero_bits])
        # Multi-controlled X on phase qubit
        if n_input == 1:
            qc.cx(qr_input[0], qr_phase[0])
        else:
            qc.mcx(list(qr_input), qr_phase[0])
        # Uncompute X
        if zero_bits:
            qc.x([qr_input[i] for i in zero_bits])


def compute_grover_iterations(n_states: int, n_marked: int) -> int:
    """Optimal number of Grover iterations."""
    if n_marked == 0:
        return 0
    if n_marked >= n_states:
        return 0
    return max(1, round((math.pi / 4) * math.sqrt(n_states / n_marked)))


# ---------------------------------------------------------------------------
# Run Grover and decode result
# ---------------------------------------------------------------------------

def run_and_decode_machine(
    machine_id: int,
    machine_ops: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
    shots: int = 1024,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Build and run Grover circuit for one machine.
    Returns (best_ordering, counts).
    """
    k = len(machine_ops)
    n_orderings = math.factorial(k)

    # Count valid orderings to determine iterations
    n_valid, valid_orderings = count_valid_machine_orderings(
        machine_ops, ops, feasible_windows
    )

    if n_valid == 0:
        print(f"  Machine {machine_id}: no valid orderings found!")
        return machine_ops, {}  # fallback: return original order

    n_iter = compute_grover_iterations(n_orderings, n_valid)
    print(f"  Machine {machine_id}: n_orderings={n_orderings}, "
          f"n_valid={n_valid}, n_iter={n_iter}")

    if k == 2:
        qc = build_grover_circuit_2ops(
            machine_id, machine_ops, ops, feasible_windows, n_iter
        )
    else:
        qc = build_grover_circuit_general(
            machine_id, machine_ops, ops, feasible_windows, n_iter
        )

    # Run on AerSimulator
    sim = AerSimulator()
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Pick most frequent measurement
    best_bits = max(counts, key=counts.get)
    # Qiskit returns bits as string MSB-first; convert to integer
    best_idx = int(best_bits, 2)

    # Map index to ordering
    all_orderings = enumerate_machine_orderings(machine_ops)
    if best_idx < len(all_orderings):
        best_ordering = all_orderings[best_idx]
    else:
        best_ordering = all_orderings[0]

    # Verify feasibility; fall back to classical result if Grover picked wrong one.
    # For N=2, M=1 the Grover distribution is always 50/50 (sin²(3π/4)=1/2),
    # so the measurement is essentially a coin flip. Classical verification is
    # required for small instances.
    feasible, _ = check_machine_ordering_feasible(best_ordering, ops, feasible_windows)
    if not feasible:
        if valid_orderings:
            print(f"  [fallback] Grover returned infeasible ordering {best_ordering}; "
                  f"using classical result {valid_orderings[0]}")
            best_ordering = valid_orderings[0]
        else:
            print(f"  [warning] No valid orderings found for machine {machine_id}")

    return best_ordering, counts


# ---------------------------------------------------------------------------
# Schedule reconstruction from per-machine orderings
# ---------------------------------------------------------------------------

def reconstruct_schedule(
    machine_orderings: Dict[int, List[int]],
    ops: List[Operation],
) -> Dict[int, int]:
    """
    Given per-machine orderings, build a global permutation and compute
    start times.

    Strategy: merge machine orderings into a topological order respecting:
    1. Job precedence (from ops structure)
    2. Machine ordering constraints (from machine_orderings)

    For each machine, the ordering defines precedence edges:
      ordering[i] must finish before ordering[i+1] starts.

    We perform a priority-queue simulation:
    - At time t, schedule whichever op is ready (all predecessors done).
    """
    n = len(ops)

    # Build predecessor sets
    # Job precedence
    job_pred: Dict[int, Optional[int]] = {}  # op_idx -> predecessor op_idx in job
    jobs_ops: Dict[int, List[int]] = {}
    for i, op in enumerate(ops):
        jobs_ops.setdefault(op.job_id, []).append(i)
    for job_id, op_list in jobs_ops.items():
        job_pred[op_list[0]] = None
        for k in range(1, len(op_list)):
            job_pred[op_list[k]] = op_list[k - 1]

    # Machine ordering precedence
    machine_pred: Dict[int, Optional[int]] = {i: None for i in range(n)}
    for m, ordering in machine_orderings.items():
        for k in range(1, len(ordering)):
            machine_pred[ordering[k]] = ordering[k - 1]

    # Simulate: compute start times
    finish_time: Dict[int, int] = {}
    start_time: Dict[int, int] = {}

    def get_finish(op_idx: Optional[int]) -> int:
        if op_idx is None:
            return 0
        return finish_time.get(op_idx, -1)

    # Topological sort via iterative resolution
    scheduled = set()
    for _ in range(n):
        for i in range(n):
            if i in scheduled:
                continue
            jp = job_pred.get(i)
            mp = machine_pred.get(i)
            jp_done = (jp is None) or (jp in scheduled)
            mp_done = (mp is None) or (mp in scheduled)
            if jp_done and mp_done:
                jp_ft = get_finish(jp)
                mp_ft = get_finish(mp)
                if jp_ft >= 0 or jp is None:
                    if mp_ft >= 0 or mp is None:
                        st = max(
                            0 if jp is None else jp_ft,
                            0 if mp is None else mp_ft
                        )
                        start_time[i] = st
                        finish_time[i] = st + ops[i].proc_time
                        scheduled.add(i)
                        break

    return start_time


def compute_makespan(start_time: Dict[int, int], ops: List[Operation]) -> int:
    return max(start_time[i] + ops[i].proc_time for i in start_time)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_jssp(
    jobs: List[List[Tuple[int, int]]],
    T_max: int,
    shots: int = 1024,
    verbose: bool = True,
) -> Tuple[Optional[Dict[int, int]], int]:
    """
    Solve JSSP using Constraint-Decomposed Grover Search.

    Args:
        jobs: list of jobs, each a list of (machine_id, proc_time) tuples
        T_max: upper bound on makespan
        shots: simulator shots per machine
        verbose: print progress

    Returns:
        (start_time dict, makespan) or (None, T_max) if no solution found
    """
    ops = build_operations(jobs)
    n_ops = len(ops)

    if verbose:
        print(f"\n=== JSSP Solver ===")
        print(f"Jobs: {len(jobs)}, Ops: {n_ops}, T_max: {T_max}")
        for op in ops:
            print(f"  {op}")

    # Stage 1: Classical constraint propagation
    if verbose:
        print("\n--- Stage 1: Classical Constraint Propagation ---")
    feasible_windows = compute_feasible_windows(ops, T_max)
    ops_per_machine = group_operations_by_machine(ops)

    if verbose:
        for op_idx, (es, lf) in feasible_windows.items():
            print(f"  Op {op_idx} ({ops[op_idx]}): ES={es}, LF={lf}")

    # Stage 2: Per-machine Grover search
    if verbose:
        print("\n--- Stage 2: Per-Machine Quantum Search ---")

    machine_orderings: Dict[int, List[int]] = {}
    all_counts: Dict[int, Dict[str, int]] = {}

    for m, machine_ops in sorted(ops_per_machine.items()):
        if verbose:
            print(f"\nMachine {m}: ops={machine_ops}")
        ordering, counts = run_and_decode_machine(
            m, machine_ops, ops, feasible_windows, shots=shots
        )
        machine_orderings[m] = ordering
        all_counts[m] = counts
        if verbose:
            print(f"  Selected ordering: {ordering}")
            print(f"  Counts: {counts}")

    # Bridge 2: Schedule reconstruction
    if verbose:
        print("\n--- Bridge 2: Schedule Reconstruction ---")

    start_time = reconstruct_schedule(machine_orderings, ops)
    makespan = compute_makespan(start_time, ops)

    if verbose:
        print(f"Start times: {start_time}")
        print(f"Makespan: {makespan}")
        for i, op in enumerate(ops):
            st = start_time.get(i, -1)
            print(f"  Op {i} ({op}): start={st}, finish={st + op.proc_time}")

    if makespan <= T_max:
        if verbose:
            print(f"\nSolution found! Makespan={makespan} <= T_max={T_max}")
        return start_time, makespan
    else:
        if verbose:
            print(f"\nNo valid solution found (makespan={makespan} > T_max={T_max})")
        return None, makespan


# ---------------------------------------------------------------------------
# Validation routines
# ---------------------------------------------------------------------------

def validate_jns_encoding(ops: List[Operation]):
    print("\n[Validation] JNS Encode/Decode")
    validate_jns(ops)


def validate_schedule_evaluation(ops: List[Operation]):
    print("\n[Validation] Schedule Evaluation")
    valid_perms = _enumerate_valid_permutations(ops)
    print(f"  Evaluating {len(valid_perms)} valid permutations:")
    for perm in valid_perms:
        ms, st = evaluate_schedule(perm, ops)
        print(f"    perm={perm} -> makespan={ms}, start_times={st}")


def validate_oracle(
    machine_id: int,
    machine_ops: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
):
    """
    Verify oracle using statevector simulation.

    For each basis state |x> on the input register, with the phase qubit
    initialized to |-> = (|0>-|1>)/sqrt(2), the oracle should produce:
      - if x is marked:   |x> ⊗ (-|->)  [phase kickback gives global -1 on |x>]
      - if x is unmarked: |x> ⊗ |->

    We detect the phase flip by preparing the input in |x>, phase in |+>,
    applying the oracle, then checking whether the phase qubit's relative
    phase has flipped. Concretely:

      Prepare: |x> ⊗ |+>
      H on phase qubit -> effectively Oracle acts on |-> part
      Oracle: |x>|-> -> (-1)^f(x)|x>|->
      H on phase qubit to convert relative phase to amplitude:
        H|-> = |1>, H(-|->) = -|1> (same amplitude, not useful)

    The correct statevector approach: prepare |x>|-> and check that after
    the oracle the amplitude on |x,0> is +1/sqrt(2) * (-1)^f(x) and on
    |x,1> is -1/sqrt(2) * (-1)^f(x). A -1 global sign is unobservable,
    but we verify the RELATIVE sign between |x,0> and |x,1> is always
    negative (i.e., the phase qubit is still |->), and that the overall
    sign of the |x> component matches (-1)^f(x) by comparing with the
    amplitude before the oracle.
    """
    print(f"\n[Validation] Oracle for Machine {machine_id}")
    k = len(machine_ops)
    n_orderings = math.factorial(k)
    n_input = math.ceil(math.log2(n_orderings)) if n_orderings > 1 else 1

    orderings = enumerate_machine_orderings(machine_ops)
    marked = []
    for i, ordering in enumerate(orderings):
        feasible, _ = check_machine_ordering_feasible(ordering, ops, feasible_windows)
        if feasible:
            marked.append(i)

    print(f"  Orderings: {orderings}")
    print(f"  Expected marked: {marked}")

    all_pass = True
    for idx in range(min(n_orderings, 2 ** n_input)):
        # Build circuit: input |idx>, phase |->
        qr_input = QuantumRegister(n_input, name='inp')
        qr_phase = QuantumRegister(1, name='phase')
        qc = QuantumCircuit(qr_input, qr_phase)

        # Phase |->
        qc.x(qr_phase[0])
        qc.h(qr_phase[0])

        # Input |idx>
        bits = [(idx >> b) & 1 for b in range(n_input)]
        for b, bit in enumerate(bits):
            if bit:
                qc.x(qr_input[b])

        # Statevector BEFORE oracle
        sv_before = Statevector(qc)

        # Apply oracle
        if k == 2:
            _apply_oracle_2ops(qc, qr_input, qr_phase, marked)
        else:
            _apply_oracle_general(qc, qr_input, qr_phase, marked, n_input, n_orderings)

        # Statevector AFTER oracle
        sv_after = Statevector(qc)

        # The oracle should multiply the amplitude of basis state |idx, phase>
        # by (-1)^f(idx). We check by computing the ratio of a non-zero
        # amplitude component corresponding to |idx> on input.
        # Input qubit ordering in Qiskit statevector: qr_input[0] is bit 0 (LSB).
        # Phase qubit is the last register.
        # State index for |input=idx, phase=0>:
        state_idx_phase0 = idx  # input bits + phase=0 (LSB of phase reg is 0)
        state_idx_phase1 = idx + (1 << n_input)  # phase=1

        amp_before_p0 = sv_before.data[state_idx_phase0]
        amp_after_p0 = sv_after.data[state_idx_phase0]

        # Expected: amp_after = (-1)^f(idx) * amp_before (up to floating point)
        expected_sign = -1 if idx in marked else +1
        if abs(amp_before_p0) > 1e-9:
            ratio = amp_after_p0 / amp_before_p0
            detected_sign = int(np.sign(ratio.real))
            phase_correct = abs(ratio - expected_sign) < 1e-6
        else:
            phase_correct = False
            detected_sign = 0

        status = "OK" if phase_correct else "FAIL"
        if not phase_correct:
            all_pass = False
        print(f"  Basis |{idx}>: expected_sign={expected_sign:+d}, "
              f"got_ratio={ratio:.3f} [{status}]")

    if all_pass:
        print(f"  Oracle validation PASSED for machine {machine_id}")
    else:
        print(f"  Oracle validation FAILED for machine {machine_id}")


def validate_grover_distribution(
    machine_id: int,
    machine_ops: List[int],
    ops: List[Operation],
    feasible_windows: Dict[int, Tuple[int, int]],
    shots: int = 2048,
):
    """
    Check that Grover output distribution peaks at marked states.
    """
    print(f"\n[Validation] Grover Distribution for Machine {machine_id}")
    k = len(machine_ops)
    n_orderings = math.factorial(k)
    n_valid, valid_orderings = count_valid_machine_orderings(
        machine_ops, ops, feasible_windows
    )
    n_iter = compute_grover_iterations(n_orderings, n_valid)
    print(f"  n_orderings={n_orderings}, n_valid={n_valid}, n_iter={n_iter}")
    if n_orderings == 2 and n_valid == 1:
        print("  NOTE: for N=2, M=1, Grover gives sin^2(3*pi/4)=1/2 -- "
              "distribution is always 50/50 (no amplitude amplification)")

    if k == 2:
        qc = build_grover_circuit_2ops(
            machine_id, machine_ops, ops, feasible_windows, n_iter
        )
    else:
        qc = build_grover_circuit_general(
            machine_id, machine_ops, ops, feasible_windows, n_iter
        )

    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts()
    print(f"  Counts: {counts}")

    all_orderings = enumerate_machine_orderings(machine_ops)
    for bits, count in sorted(counts.items(), key=lambda x: -x[1]):
        idx = int(bits, 2)
        if idx < len(all_orderings):
            ordering = all_orderings[idx]
            feasible, _ = check_machine_ordering_feasible(
                ordering, ops, feasible_windows
            )
            marker = "*** MARKED ***" if feasible else ""
            print(f"  |{bits}> (idx={idx}, ordering={ordering}): "
                  f"{count}/{shots} = {count/shots:.2%} {marker}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Reference test instance from Venturelli et al. 2015
    # Job 0: op(machine=0, time=1), op(machine=1, time=2)
    # Job 1: op(machine=1, time=1), op(machine=0, time=2)
    # Optimal makespan: 4
    jobs = [
        [(0, 1), (1, 2)],   # Job 0
        [(1, 1), (0, 2)],   # Job 1
    ]
    T_max = 4

    ops = build_operations(jobs)

    print("=" * 60)
    print("Test instance:")
    for op in ops:
        print(f"  {op}")

    # ---- Classical validations ----
    validate_jns_encoding(ops)
    validate_schedule_evaluation(ops)

    # ---- Feasible windows ----
    feasible_windows = compute_feasible_windows(ops, T_max)
    ops_per_machine = group_operations_by_machine(ops)
    print(f"\nFeasible windows: {feasible_windows}")
    print(f"Ops per machine: {ops_per_machine}")

    # ---- Oracle and Grover validations per machine ----
    for m, machine_ops in sorted(ops_per_machine.items()):
        validate_oracle(m, machine_ops, ops, feasible_windows)
        validate_grover_distribution(m, machine_ops, ops, feasible_windows)

    # ---- Full solver ----
    print("\n" + "=" * 60)
    schedule, makespan = solve_jssp(jobs, T_max, shots=1024, verbose=True)

    print("\n" + "=" * 60)
    print(f"RESULT: makespan={makespan} (optimal=4)")
    if schedule is not None and makespan == T_max:
        print("SUCCESS: optimal makespan achieved")
    elif schedule is not None:
        print(f"FEASIBLE: found schedule with makespan={makespan}")
    else:
        print("FAILED: no feasible schedule found")


if __name__ == "__main__":
    main()
