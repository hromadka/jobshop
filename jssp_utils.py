"""
jssp_utils.py
Classical utilities for the JSSP quantum search implementation.

Covers:
  - JNS (Job Number System) encode / decode
  - CPM (Critical Path Method) makespan computation
  - JSSP feasibility predicate Phi_T
  - Lower bound computation
  - Instance generators and pretty-printers
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Operation:
    job: int
    idx: int          # position within job (0-based)
    machine: int
    duration: int


@dataclass
class JSSPInstance:
    """
    A JSSP instance.

    jobs[j][i] = (machine, duration) for the i-th operation of job j.
    """
    jobs: List[List[Tuple[int, int]]]

    # ---------- derived quantities ----------
    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_machines(self) -> int:
        return max(m for job in self.jobs for m, _ in job) + 1

    @property
    def operations(self) -> List[Operation]:
        """Global list of all operations in job-then-position order."""
        ops = []
        k = 0
        for j, job in enumerate(self.jobs):
            for i, (m, d) in enumerate(job):
                ops.append(Operation(job=j, idx=i, machine=m, duration=d))
                k += 1
        return ops

    @property
    def K(self) -> int:
        return sum(len(job) for job in self.jobs)

    @property
    def P_max(self) -> int:
        return sum(d for job in self.jobs for _, d in job)

    @property
    def N_valid(self) -> int:
        k = self.K
        num = math.factorial(k)
        den = math.prod(math.factorial(len(job)) for job in self.jobs)
        return num // den

    @property
    def n_perm(self) -> int:
        nv = self.N_valid
        return max(1, math.ceil(math.log2(nv))) if nv > 1 else 1

    def lower_bound(self) -> int:
        """Machine-based lower bound: max over machines of sum of durations."""
        machine_load: Dict[int, int] = {}
        for job in self.jobs:
            for m, d in job:
                machine_load[m] = machine_load.get(m, 0) + d
        return max(machine_load.values()) if machine_load else 0

    def job_lower_bound(self) -> int:
        """Job-based lower bound: max over jobs of sum of durations."""
        return max(sum(d for _, d in job) for job in self.jobs)

    def __str__(self) -> str:
        lines = [f"JSSP {self.num_jobs}x{self.num_machines}, K={self.K}"]
        lines.append(f"  N_valid={self.N_valid}, n_perm={self.n_perm} qubits")
        lines.append(f"  P_max={self.P_max}, lower_bound={self.lower_bound()}")
        for j, job in enumerate(self.jobs):
            ops_str = ", ".join(f"M{m}:{d}" for m, d in job)
            lines.append(f"  Job {j}: {ops_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JNS encode / decode
# ---------------------------------------------------------------------------

def _multinomial_rank(seq: List[int], job_sizes: List[int]) -> int:
    """
    Encode a valid sequencing (list of job indices, one per operation)
    as a JNS integer in [0, N_valid).

    seq[k] = which job the k-th slot belongs to.
    This is the standard combinatorial number system for multiset permutations.
    """
    K = len(seq)
    # remaining operations per job
    remaining = list(job_sizes)
    N_valid = math.prod(math.factorial(r) for r in remaining)
    N_valid = math.factorial(K) // math.prod(math.factorial(s) for s in job_sizes)

    rank = 0
    n = K
    for k in range(K):
        j = seq[k]
        # count how many valid choices come before job j
        # in the current remaining counts
        prefix = sum(remaining[jj] for jj in range(j))
        # number of completions if we pick each earlier job next
        # = C(n-1, remaining[0], ...) for each earlier job choice
        # simplified: we compute the factoradic contribution
        total_rem = sum(remaining)
        assert total_rem == n - k
        denom = math.factorial(n - k - 1)
        for jj, r in enumerate(remaining):
            denom //= math.factorial(r if jj != j else r - 1)
        rank += prefix_rank(remaining, j, n - k)
        remaining[j] -= 1

    return rank


def prefix_rank(remaining: List[int], chosen: int, n: int) -> int:
    """
    Number of valid interleavings that start with a job index < chosen,
    given current remaining counts and total n positions left.
    """
    total = 0
    rem_copy = list(remaining)
    for j in range(chosen):
        if rem_copy[j] == 0:
            continue
        rem_copy[j] -= 1
        num = math.factorial(n - 1)
        den = math.prod(math.factorial(r) for r in rem_copy)
        total += num // den
        rem_copy[j] += 1
    return total


def jns_decode(x: int, job_sizes: List[int]) -> List[int]:
    """
    Decode integer x in [0, N_valid) to a sequencing list.

    Returns seq where seq[k] = job index of the k-th operation slot.
    This is equivalent to a Lehmer code / factoradic unranking for
    multiset permutations.
    """
    K = sum(job_sizes)
    remaining = list(job_sizes)
    seq = []
    n = K

    for k in range(K):
        # find which job comes k-th in the sequencing
        for j in range(len(job_sizes)):
            if remaining[j] == 0:
                continue
            # count interleavings starting with job j
            remaining[j] -= 1
            count = math.factorial(n - k - 1) // math.prod(
                math.factorial(r) for r in remaining
            )
            if x < count:
                seq.append(j)
                break
            x -= count
            remaining[j] += 1

    return seq


def sequencing_to_ops(seq: List[int], instance: JSSPInstance) -> List[Operation]:
    """
    Convert a job-index sequencing to a list of Operation objects.
    seq[k] = job index for slot k; within each job, operations are
    assigned in order (next unused operation of that job).
    """
    ops = instance.operations
    # Build per-job operation lists
    job_ops: List[List[Operation]] = [[] for _ in range(instance.num_jobs)]
    for op in ops:
        job_ops[op.job].append(op)

    # Pointer for each job
    ptr = [0] * instance.num_jobs
    result = []
    for j in seq:
        result.append(job_ops[j][ptr[j]])
        ptr[j] += 1
    return result


# ---------------------------------------------------------------------------
# CPM (Critical Path Method)
# ---------------------------------------------------------------------------

def cpm_makespan(seq: List[int], instance: JSSPInstance) -> int:
    """
    Given a job-index sequencing, compute the CPM makespan
    (earliest-start schedule).

    Uses overflow-safe arithmetic: all values bounded by P_max.
    """
    ops = sequencing_to_ops(seq, instance)
    K = len(ops)

    # earliest finish time per job (indexed by job id) and per machine
    job_finish: Dict[int, int] = {}
    machine_finish: Dict[int, int] = {}
    finish_times = []

    for op in ops:
        e_job = job_finish.get(op.job, 0)
        e_mach = machine_finish.get(op.machine, 0)
        start = max(e_job, e_mach)
        finish = start + op.duration
        job_finish[op.job] = finish
        machine_finish[op.machine] = finish
        finish_times.append(finish)

    return max(finish_times) if finish_times else 0


def cpm_schedule(seq: List[int], instance: JSSPInstance) -> Dict[int, int]:
    """
    Compute the full CPM earliest-start schedule.

    Returns {global_op_index: start_time}.
    """
    ops = sequencing_to_ops(seq, instance)
    job_finish: Dict[int, int] = {}
    machine_finish: Dict[int, int] = {}
    schedule: Dict[int, int] = {}

    # Build a lookup from (job, idx) to global op index
    global_idx: Dict[Tuple[int,int], int] = {}
    for k, op in enumerate(instance.operations):
        global_idx[(op.job, op.idx)] = k

    for op in ops:
        e_job = job_finish.get(op.job, 0)
        e_mach = machine_finish.get(op.machine, 0)
        start = max(e_job, e_mach)
        finish = start + op.duration
        job_finish[op.job] = finish
        machine_finish[op.machine] = finish
        gk = global_idx[(op.job, op.idx)]
        schedule[gk] = start

    return schedule


def phi_T(x: int, instance: JSSPInstance, T: int) -> bool:
    """
    The JSSP feasibility predicate Phi_T(x).

    Returns True iff:
      - x is a valid JNS index (x < N_valid), AND
      - C_max(JNS_decode(x)) <= min(T, P_max)
    """
    if x >= instance.N_valid:
        return False
    job_sizes = [len(job) for job in instance.jobs]
    seq = jns_decode(x, job_sizes)
    threshold = min(T, instance.P_max)
    return cpm_makespan(seq, instance) <= threshold


def all_feasible(instance: JSSPInstance, T: int) -> List[int]:
    """Return list of all JNS indices x with Phi_T(x) = True."""
    return [x for x in range(instance.N_valid) if phi_T(x, instance, T)]


def verify_schedule(schedule: Dict[int, int], instance: JSSPInstance) -> Tuple[bool, str]:
    """
    Verify a schedule satisfies all JSSP constraints.
    Returns (valid, error_message).
    """
    ops = instance.operations

    # C_ord: job precedence
    for j, job in enumerate(instance.jobs):
        for i in range(len(job) - 1):
            k = sum(len(instance.jobs[jj]) for jj in range(j)) + i
            k_next = k + 1
            if schedule[k] + ops[k].duration > schedule[k_next]:
                return False, f"Job {j} precedence violated: op {i} -> op {i+1}"

    # C_mach: no machine overlap
    by_machine: Dict[int, List[Tuple[int,int,int]]] = {}  # m -> [(start, end, op_idx)]
    for k, op in enumerate(ops):
        if op.machine not in by_machine:
            by_machine[op.machine] = []
        s = schedule[k]
        by_machine[op.machine].append((s, s + op.duration, k))

    for m, intervals in by_machine.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i+1][0]:
                return False, f"Machine {m} overlap between ops {intervals[i][2]} and {intervals[i+1][2]}"

    return True, "OK"


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def load_or_library(path: str) -> List["JSSPInstance"]:
    """
    Load one or more JSSP instances from an OR-Library format file.

    Format per instance:
        J M
        m0 d0 m1 d1 ... (J lines, one per job)

    Each job line contains an even number of integers (machine, duration pairs).
    Jobs may have different numbers of operations (variable-length JSSP).
    M is the total number of machines; it does not fix the operations per job.

    Comment lines starting with '#', '+', or 'c' are skipped.
    Multiple instances may appear in one file separated by non-data lines.

    Returns a list of JSSPInstance objects.
    """
    with open(path) as f:
        raw_lines = f.readlines()

    def is_data(line: str) -> bool:
        s = line.strip()
        return bool(s) and s[0].isdigit()

    instances: List[JSSPInstance] = []
    lines = [l.strip() for l in raw_lines]
    i = 0

    while i < len(lines):
        if not is_data(lines[i]):
            i += 1
            continue

        parts = lines[i].split()
        if len(parts) < 2:
            i += 1
            continue

        try:
            J, M = int(parts[0]), int(parts[1])
        except ValueError:
            i += 1
            continue

        i += 1
        jobs: List[List[Tuple[int, int]]] = []

        for _j in range(J):
            while i < len(lines) and not is_data(lines[i]):
                i += 1
            if i >= len(lines):
                break
            nums = list(map(int, lines[i].split()))
            # Variable-length: parse all pairs on this line
            n_ops = len(nums) // 2
            job = [(nums[2 * k], nums[2 * k + 1]) for k in range(n_ops)]
            jobs.append(job)
            i += 1

        if len(jobs) == J:
            instances.append(JSSPInstance(jobs=jobs))

    return instances


def load_taillard(path: str, zero_indexed: bool = True) -> List["JSSPInstance"]:
    """
    Load one or more JSSP instances from a Taillard format file.

    Format per instance:
        J M
        <J lines of M processing times>
        <J lines of M machine assignments>

    zero_indexed: if False, machine numbers are decremented by 1.
    """
    with open(path) as f:
        lines = [l.strip() for l in f
                 if l.strip() and not l.strip().startswith('c')]

    instances: List[JSSPInstance] = []
    i = 0

    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 2:
            i += 1
            continue
        try:
            J, M = int(parts[0]), int(parts[1])
        except ValueError:
            i += 1
            continue

        i += 1
        if i + 2 * J > len(lines):
            break

        durations = [list(map(int, lines[i + j].split())) for j in range(J)]
        i += J
        machines_raw = [list(map(int, lines[i + j].split())) for j in range(J)]
        i += J

        if not zero_indexed:
            machines_raw = [[m - 1 for m in row] for row in machines_raw]

        jobs = [
            [(machines_raw[j][op], durations[j][op]) for op in range(M)]
            for j in range(J)
        ]
        instances.append(JSSPInstance(jobs=jobs))

    return instances


def load_benchmark_file(path: str) -> List["JSSPInstance"]:
    """
    Auto-detect OR-Library vs Taillard format and load all instances.

    Heuristic:
    - If the total number of data lines equals 1 + 2*J (header + 2 blocks
      of J lines), it is likely Taillard.
    - Otherwise assume OR-Library (interleaved, variable-length jobs).
    OR-Library is the default when ambiguous.
    """
    with open(path) as f:
        raw = f.readlines()

    data_lines = [l.strip() for l in raw
                  if l.strip() and l.strip()[0].isdigit()]

    if len(data_lines) < 2:
        return load_or_library(path)

    try:
        J, M = map(int, data_lines[0].split()[:2])
    except (ValueError, IndexError):
        return load_or_library(path)

    # Taillard has exactly 1 + 2*J data lines per instance (header + 2 blocks)
    # OR-Library has exactly 1 + J data lines per instance (header + job rows)
    n_data = len(data_lines)
    if n_data == 1 + 2 * J:
        # Could be Taillard — check that each block has exactly M tokens
        dur_lines   = data_lines[1:J+1]
        mach_lines  = data_lines[J+1:2*J+1]
        if all(len(l.split()) == M for l in dur_lines + mach_lines):
            return load_taillard(path)

    # Default: OR-Library (handles variable-length jobs)
    return load_or_library(path)


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

def summarise_instance(inst: "JSSPInstance", name: str = "") -> str:
    """One-line summary of an instance."""
    lb = max(inst.lower_bound(), inst.job_lower_bound())
    tag = f"[{name}] " if name else ""
    return (f"{tag}J={inst.num_jobs} M={inst.num_machines} "
            f"K={inst.K} N_valid={inst.N_valid} "
            f"n_perm={inst.n_perm} P_max={inst.P_max} T_lb={lb}")


def make_2x2_instance() -> JSSPInstance:
    """
    2-job, 2-machine instance used as the paper's validation example.
    Job 0: M0(1) -> M1(2)
    Job 1: M1(2) -> M0(2)
    Optimal makespan: C*_max = 4
    N_valid = 6, n_perm = 3 qubits
    S = 4 feasible sequencings at T=4.
    """
    return JSSPInstance(jobs=[
        [(0, 1), (1, 2)],   # job 0
        [(1, 2), (0, 2)],   # job 1
    ])


def make_3x2_instance() -> JSSPInstance:
    """
    3-job, 2-machine instance.
    Job 0: M0(1) -> M1(3)
    Job 1: M1(2) -> M0(2)
    Job 2: M0(3) -> M1(1)
    N_valid = 90, n_perm = 7 qubits
    """
    return JSSPInstance(jobs=[
        [(0, 1), (1, 3)],
        [(1, 2), (0, 2)],
        [(0, 3), (1, 1)],
    ])


def make_2x3_instance() -> JSSPInstance:
    """
    2-job, 3-machine instance.
    Job 0: M0(2) -> M1(1) -> M2(3)
    Job 1: M2(2) -> M0(3) -> M1(1)
    N_valid = 20, n_perm = 5 qubits
    """
    return JSSPInstance(jobs=[
        [(0, 2), (1, 1), (2, 3)],
        [(2, 2), (0, 3), (1, 1)],
    ])


def make_3x3_instance() -> JSSPInstance:
    """
    3-job, 3-machine instance (Taillard-style).
    N_valid = 1680, n_perm = 11 qubits
    """
    return JSSPInstance(jobs=[
        [(0, 3), (1, 2), (2, 2)],
        [(1, 2), (2, 1), (0, 4)],
        [(2, 3), (0, 1), (1, 3)],
    ])
