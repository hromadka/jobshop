# ── dependencies ──────────────────────────────────────────────────────────────
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ── 1. problem definition ─────────────────────────────────────────────────────

class JSSPInstance:
    """
    Represents a JSSP instance.

    jobs: list of lists, each inner list is a sequence of
          (machine, duration) tuples representing the ordered
          operations of that job.

    Example 2x2x2 instance:
        jobs = [
            [(0, 1), (1, 2)],
            [(1, 1), (0, 2)],
        ]
    """
    def __init__(self, jobs: list, T: int):
        self.jobs = jobs
        self.T = T
        self.n_jobs = len(jobs)
        self.n_machines = max(m for job in jobs for m, _ in job) + 1

        # Flatten all operations to a global index k
        self.ops = []               # (job_idx, op_idx, machine, duration)
        self.ops_by_job = {}        # job_idx  -> [global k indices, in order]
        self.ops_by_machine = {}    # machine  -> [global k indices]

        k = 0
        for j, job in enumerate(jobs):
            self.ops_by_job[j] = []
            for i, (m, dur) in enumerate(job):
                self.ops.append((j, i, m, dur))
                self.ops_by_job[j].append(k)
                self.ops_by_machine.setdefault(m, []).append(k)
                k += 1

        self.K = k

        # Compute initial feasible windows from job structure alone
        self.e_k = []   # earliest start per operation
        self.L_k = []   # latest start per operation
        self.T_k = []   # window width = L_k - e_k + 1
        self._compute_initial_windows()

    def _compute_initial_windows(self):
        for j, job in enumerate(self.jobs):
            cumulative = 0
            for i, (m, dur) in enumerate(job):
                e = cumulative
                successor_sum = sum(d for _, d in job[i + 1:])
                L = self.T - dur - successor_sum
                self.e_k.append(e)
                self.L_k.append(L)
                self.T_k.append(max(0, L - e + 1))
                cumulative += dur

    def duration(self, k: int) -> int:
        return self.ops[k][3]

    def machine(self, k: int) -> int:
        return self.ops[k][2]

    def total_qubits(self) -> int:
        return sum(self.T_k)

    def job_qubit_count(self, job_idx: int) -> int:
        return sum(self.T_k[k] for k in self.ops_by_job[job_idx])

    def machine_qubit_count(self, machine: int,
                             windows: dict = None) -> int:
        """
        Qubit count for machine circuit.
        windows: optional dict k -> (e, L) overriding instance windows
                 (used after Stage 1 pruning).
        """
        total = 0
        for k in self.ops_by_machine[machine]:
            if windows and k in windows:
                e, L = windows[k]
                total += max(0, L - e + 1)
            else:
                total += self.T_k[k]
        return total


# ── 2. classical preprocessing ───────────────────────────────────────────────

def compute_feasible_windows(instance: JSSPInstance):
    """
    Validate that all initial feasible windows are non-empty.
    Raises ValueError if the instance is infeasible under deadline T.
    """
    for k in range(instance.K):
        if instance.T_k[k] <= 0:
            raise ValueError(
                f"Operation {k} has empty feasible window: "
                f"e={instance.e_k[k]}, L={instance.L_k[k]}, "
                f"T={instance.T}"
            )


# ── 3. W-state preparation ────────────────────────────────────────────────────

def w_state_circuit(T: int) -> QuantumCircuit:
    """
    Prepares |W_T⟩ = (1/√T) Σ_{t=0}^{T-1} |e_t⟩ on T qubits from |0...0⟩.

    Uses cascade of CRY + CNOT gates.
    Depth O(T), gate count 2(T-1) two-qubit gates.
    """
    if T == 1:
        qc = QuantumCircuit(1, name="W1")
        qc.x(0)
        return qc
    qc = QuantumCircuit(T, name=f"W{T}")
    qc.x(0)
    for i in range(T - 1):
        theta = 2.0 * np.arccos(np.sqrt(1.0 / (T - i)))
        qc.cry(theta, i, i + 1)
        qc.cx(i + 1, i)
    return qc


def tensor_w_state(T_list: list) -> QuantumCircuit:
    """
    Prepares tensor product ⊗_k |W_{T_k}⟩ for a list of window widths.
    Each operation block is initialized independently.
    Total qubits: sum(T_list).
    """
    total = sum(T_list)
    qc = QuantumCircuit(total, name="tensor_W")
    offset = 0
    for Tk in T_list:
        if Tk > 0:
            w = w_state_circuit(Tk)
            qc.compose(w, qubits=list(range(offset, offset + Tk)), inplace=True)
        offset += Tk
    return qc


# ── 4. oracle building block ──────────────────────────────────────────────────

def build_conflict_oracle(n_search: int,
                          conflict_pairs_qubits: list,
                          circuit_name: str) -> QuantumCircuit:
    """
    Generic phase-marking oracle for a list of conflicting qubit pairs.

    Marks states where NO conflict is detected (all conflict pairs = 0).
    Used by both Stage 1 (ordering conflicts) and Stage 2 (machine conflicts).

    Args:
        n_search:              number of search register qubits
        conflict_pairs_qubits: list of (q0, q1) qubit index pairs within the
                               search register that must not both be 1
        circuit_name:          name for the QuantumCircuit

    Qubit layout of returned circuit:
        [0 .. n_search-1]          search register
        [n_search .. n_search+P-1] conflict ancilla (one per pair)
        [n_search+P]               flag ancilla
        [n_search+P+1]             phase qubit (must be initialized to |-> externally)

    Circuit action:
        phase qubit acquires -1 phase iff NO conflict pair is simultaneously 1
        (i.e., the assignment is valid)
    All ancilla restored to |0⟩ after call (uncomputation included).
    """
    P = len(conflict_pairs_qubits)
    n_ancilla = P + 1       # P conflict bits + 1 flag
    n_phase = 1
    total = n_search + n_ancilla + n_phase

    anc_base    = n_search
    flag_qubit  = n_search + P
    phase_qubit = n_search + P + 1

    qc = QuantumCircuit(total, name=circuit_name)

    if P == 0:
        # No conflicts: every assignment is valid — mark all states.
        # X|-> = -|-> (global phase -1, qubit stays in |->).
        qc.x(phase_qubit)
        return qc

    # ── Step 1: detect each conflict pair ────────────────────────────────────
    for idx, (q0, q1) in enumerate(conflict_pairs_qubits):
        qa = anc_base + idx
        qc.ccx(q0, q1, qa)   # ancilla[idx] = x_{q0} AND x_{q1}

    # ── Step 2: NOR all conflict ancilla into flag ───────────────────────────
    # flag = NOR(a_0, ..., a_{P-1}) = 1 iff no conflict detected
    # X all ancilla → AND(NOT a_i) detected by MCX → flag inverted twice
    # Net: flag = NOT(AND(NOT a_i)) flipped by two X(flag) = AND(NOT a_i) = NOR(a_i)
    conflict_ancilla = list(range(anc_base, flag_qubit))
    qc.x(conflict_ancilla)
    qc.x(flag_qubit)
    qc.mcx(conflict_ancilla, flag_qubit)
    qc.x(flag_qubit)
    qc.x(conflict_ancilla)
    # Now flag = 1 iff NO conflict ancilla is 1 (valid assignment)

    # ── Step 3: phase kickback ───────────────────────────────────────────────
    # flag = 1 for valid states — CX fires directly, no inversion needed.
    qc.cx(flag_qubit, phase_qubit)

    # ── Step 4: uncompute step 2 ─────────────────────────────────────────────
    qc.x(conflict_ancilla)
    qc.x(flag_qubit)
    qc.mcx(conflict_ancilla, flag_qubit)
    qc.x(flag_qubit)
    qc.x(conflict_ancilla)

    # ── Step 5: uncompute step 1 ─────────────────────────────────────────────
    for idx, (q0, q1) in reversed(list(enumerate(conflict_pairs_qubits))):
        qa = anc_base + idx
        qc.ccx(q0, q1, qa)

    return qc


# ── 5. Stage 1 oracle and circuit ────────────────────────────────────────────

def stage1_conflict_pairs(instance: JSSPInstance,
                          job_idx: int) -> list:
    """
    Enumerate all conflicting (q0, q1) qubit index pairs within the
    job j search register that violate the ordering constraint h3.

    For consecutive operations (k, k+1) in job j:
        conflict iff x_{k,t} = 1 AND x_{k+1,t'} = 1 AND t' < t + ell_k

    Returns list of (local_qubit_q0, local_qubit_q1) pairs.
    """
    k_list = instance.ops_by_job[job_idx]

    # Local qubit index within job register for variable x_{k, e_k + t_off}
    # Build offset map
    offsets = {}
    off = 0
    for k in k_list:
        offsets[k] = off
        off += instance.T_k[k]

    pairs = []
    for idx in range(len(k_list) - 1):
        k  = k_list[idx]
        k1 = k_list[idx + 1]
        dur_k = instance.duration(k)
        e_k   = instance.e_k[k]
        e_k1  = instance.e_k[k1]
        Tk    = instance.T_k[k]
        Tk1   = instance.T_k[k1]

        for t_off in range(Tk):
            t_abs = e_k + t_off
            for t1_off in range(Tk1):
                t1_abs = e_k1 + t1_off
                if t1_abs < t_abs + dur_k:   # ordering violation
                    q0 = offsets[k]  + t_off
                    q1 = offsets[k1] + t1_off
                    pairs.append((q0, q1))

    return pairs


def stage1_oracle(instance: JSSPInstance, job_idx: int) -> QuantumCircuit:
    """
    Stage 1 phase-marking oracle for job j.
    Marks assignments satisfying h3 (ordering constraint).
    h1 is enforced structurally by W-state initialization and not checked here.
    """
    k_list   = instance.ops_by_job[job_idx]
    n_search = sum(instance.T_k[k] for k in k_list)
    pairs    = stage1_conflict_pairs(instance, job_idx)

    return build_conflict_oracle(
        n_search,
        pairs,
        circuit_name=f"oracle_s1_job{job_idx}"
    )


def restricted_diffusion(T_list: list, name: str) -> QuantumCircuit:
    """
    Grover diffusion operator restricted to the W-state feasible subspace.

    Implements D = W · (2|0><0| - I) · W†
    where W = tensor_w_state(T_list).

    Preserves h1: amplitude never leaves the one-hot subspace.

    Args:
        T_list: list of window widths, one per operation in this register
        name:   circuit name
    """
    n = sum(T_list)
    w_init = tensor_w_state(T_list)
    qc = QuantumCircuit(n, name=name)

    # W†
    qc.compose(w_init.inverse(), inplace=True)

    # 2|0><0| - I
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))

    # W
    qc.compose(w_init, inplace=True)

    return qc


def stage1_circuit(instance: JSSPInstance,
                   job_idx: int,
                   n_iterations: int = None) -> QuantumCircuit:
    """
    Complete Stage 1 Grover circuit for job j.

    Qubit layout:
        [search register]  [conflict ancilla]  [flag ancilla]  [phase qubit]

    Returns circuit with measurement on search register only.
    """
    k_list   = instance.ops_by_job[job_idx]
    T_list   = [instance.T_k[k] for k in k_list]
    n_search = sum(T_list)

    oracle    = stage1_oracle(instance, job_idx)
    diffusion = restricted_diffusion(T_list, name=f"diff_s1_job{job_idx}")

    n_extra  = oracle.num_qubits - n_search   # ancilla + phase
    total_q  = n_search + n_extra

    qr = QuantumRegister(total_q,  name=f"q_j{job_idx}")
    cr = ClassicalRegister(n_search, name=f"c_j{job_idx}")
    qc = QuantumCircuit(qr, cr, name=f"s1_job{job_idx}")

    search_qubits = list(range(n_search))
    phase_qubit   = total_q - 1

    # Initialise phase qubit to |->
    qc.x(phase_qubit)
    qc.h(phase_qubit)

    # Initialise search register with W-states (enforces h1 structurally)
    w_init = tensor_w_state(T_list)
    qc.compose(w_init, qubits=search_qubits, inplace=True)

    if n_iterations is None:
        # π/4·√N is optimal for M=1 solutions; for unknown M, callers should
        # vary n_iterations across retries (1, 2, 3, ..., max) to avoid
        # permanent over-rotation when M > 1.
        search_space = 1
        for Tk in T_list:
            search_space *= Tk
        n_iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(search_space))))

    for _ in range(n_iterations):
        qc.compose(oracle,    qubits=list(range(total_q)), inplace=True)
        qc.compose(diffusion, qubits=search_qubits,        inplace=True)

    qc.measure(search_qubits, list(range(n_search)))
    return qc


# ── 6. Stage 2 oracle and circuit ────────────────────────────────────────────

def stage2_conflict_pairs(instance: JSSPInstance,
                          machine: int,
                          windows: dict) -> list:
    """
    Enumerate all conflicting (q0, q1) qubit index pairs within the
    machine m search register that violate h2 (no-overlap constraint).

    windows: dict k -> (e_k, L_k) giving the pruned feasible window
             for each operation k on this machine after Bridge 1.

    Conflict sets (Kurowski et al. 2023):
        A_m: operation k' starts while k is still running
             0 < t' - t < ell_k  (k' overlaps k from above)
        B_m: two operations start at exactly the same time
             t' = t, both have positive duration

    Returns list of (local_qubit_q0, local_qubit_q1) pairs within
    the machine m register.
    """
    k_list = instance.ops_by_machine[machine]

    # Build local qubit offset map for this machine's register
    offsets = {}
    off = 0
    for k in k_list:
        offsets[k] = off
        e, L = windows[k]
        off += max(0, L - e + 1)

    pairs = []
    for i in range(len(k_list)):
        for j in range(len(k_list)):
            if i == j:
                continue
            k  = k_list[i]
            k1 = k_list[j]
            e_k,  L_k  = windows[k]
            e_k1, L_k1 = windows[k1]
            dur_k  = instance.duration(k)
            dur_k1 = instance.duration(k1)
            Tw_k   = max(0, L_k  - e_k  + 1)
            Tw_k1  = max(0, L_k1 - e_k1 + 1)

            for t_off in range(Tw_k):
                t_abs = e_k + t_off
                for t1_off in range(Tw_k1):
                    t1_abs = e_k1 + t1_off

                    # A_m: k running, k' starts during k's execution
                    a_conflict = (k != k1 and
                                  0 < t1_abs - t_abs < dur_k)

                    # B_m: simultaneous start (i < j to avoid duplicates)
                    b_conflict = (i < j and
                                  t1_abs == t_abs and
                                  dur_k > 0 and dur_k1 > 0)

                    if a_conflict or b_conflict:
                        q0 = offsets[k]  + t_off
                        q1 = offsets[k1] + t1_off
                        # Avoid duplicate pairs
                        pair = (min(q0, q1), max(q0, q1))
                        if pair not in pairs:
                            pairs.append(pair)

    return pairs


def stage2_oracle(instance: JSSPInstance,
                  machine: int,
                  windows: dict) -> QuantumCircuit:
    """
    Stage 2 phase-marking oracle for machine m.
    Marks assignments satisfying h2 (no-overlap constraint).

    windows: dict k -> (e_k, L_k) for operations on this machine,
             pruned by Bridge 1.
    """
    k_list   = instance.ops_by_machine[machine]
    n_search = sum(max(0, windows[k][1] - windows[k][0] + 1)
                   for k in k_list)
    pairs    = stage2_conflict_pairs(instance, machine, windows)

    return build_conflict_oracle(
        n_search,
        pairs,
        circuit_name=f"oracle_s2_mach{machine}"
    )


def stage2_circuit(instance: JSSPInstance,
                   machine: int,
                   windows: dict,
                   n_iterations: int = None) -> QuantumCircuit:
    """
    Complete Stage 2 Grover circuit for machine m.

    windows: dict k -> (e_k, L_k) giving pruned feasible windows
             from Bridge 1. For a fully fixed Stage 1 result,
             e_k = L_k = s_k (single timeslot), so the search space
             is trivial. Windows are used when Stage 1 produced a
             partial result or when we allow Stage 2 to resolve
             conflicts by searching within a window.

    Returns circuit with measurement on search register.
    """
    k_list   = instance.ops_by_machine[machine]
    T_list   = [max(0, windows[k][1] - windows[k][0] + 1) for k in k_list]
    n_search = sum(T_list)

    oracle    = stage2_oracle(instance, machine, windows)
    diffusion = restricted_diffusion(
        T_list, name=f"diff_s2_mach{machine}"
    )

    n_extra = oracle.num_qubits - n_search
    total_q = n_search + n_extra

    qr = QuantumRegister(total_q,   name=f"q_m{machine}")
    cr = ClassicalRegister(n_search, name=f"c_m{machine}")
    qc = QuantumCircuit(qr, cr, name=f"s2_mach{machine}")

    search_qubits = list(range(n_search))
    phase_qubit   = total_q - 1

    # Phase qubit initialisation
    qc.x(phase_qubit)
    qc.h(phase_qubit)

    # W-state initialisation within pruned windows
    w_init = tensor_w_state(T_list)
    qc.compose(w_init, qubits=search_qubits, inplace=True)

    if n_iterations is None:
        search_space = 1
        for Tk in T_list:
            search_space *= max(1, Tk)
        n_iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(search_space))))

    for _ in range(n_iterations):
        qc.compose(oracle,    qubits=list(range(total_q)), inplace=True)
        qc.compose(diffusion, qubits=search_qubits,        inplace=True)

    qc.measure(search_qubits, list(range(n_search)))
    return qc


# ── 7. measurement decoding ───────────────────────────────────────────────────

def decode_job_measurement(bitstring: str,
                           instance: JSSPInstance,
                           job_idx: int) -> dict:
    """
    Decode a measured bitstring for Stage 1 job j.
    Returns dict k -> absolute_start_time, or None if invalid.
    Qiskit reverses bit order in output strings.
    """
    k_list = instance.ops_by_job[job_idx]
    bits   = bitstring[::-1]   # correct for Qiskit reversal

    assignments = {}
    offset = 0
    for k in k_list:
        Tk    = instance.T_k[k]
        block = bits[offset:offset + Tk]
        if block.count('1') != 1:
            return None   # h1 violated
        t_offset = block.index('1')
        assignments[k] = instance.e_k[k] + t_offset
        offset += Tk

    return assignments


def decode_machine_measurement(bitstring: str,
                               instance: JSSPInstance,
                               machine: int,
                               windows: dict) -> dict:
    """
    Decode a measured bitstring for Stage 2 machine m.
    Returns dict k -> absolute_start_time, or None if invalid.
    """
    k_list = instance.ops_by_machine[machine]
    bits   = bitstring[::-1]

    assignments = {}
    offset = 0
    for k in k_list:
        e, L = windows[k]
        Tw   = max(0, L - e + 1)
        block = bits[offset:offset + Tw]
        if block.count('1') != 1:
            return None
        t_offset = block.index('1')
        assignments[k] = e + t_offset
        offset += Tw

    return assignments


# ── 8. classical bridges ──────────────────────────────────────────────────────

def bridge1(instance: JSSPInstance,
            job_assignments: dict) -> tuple:
    """
    Classical Bridge 1.

    Combines Stage 1 job assignments, verifies ordering constraints
    classically, identifies machine conflicts, and computes pruned
    feasible windows for Stage 2.

    Returns:
        start_times:          dict k -> start_time (from Stage 1)
        conflicts_by_machine: dict machine -> list of (k, k') conflict pairs
        stage2_windows:       dict k -> (e_k, L_k) for Stage 2
                              When Stage 1 fully fixes start times,
                              e_k = L_k = s_k (window of width 1).
                              When conflicts exist, windows are widened
                              to allow Stage 2 to search for a resolution.
    """
    # Combine all job assignments
    start_times = {}
    for job_idx, assignments in job_assignments.items():
        start_times.update(assignments)

    # Verify ordering constraints
    for j in range(instance.n_jobs):
        k_list = instance.ops_by_job[j]
        for idx in range(len(k_list) - 1):
            k  = k_list[idx]
            k1 = k_list[idx + 1]
            if start_times[k1] < start_times[k] + instance.duration(k):
                raise ValueError(
                    f"Bridge 1: ordering violated between "
                    f"op {k} and op {k1} in job {j}"
                )

    # Identify machine conflicts and build Stage 2 windows
    conflicts_by_machine = {}
    stage2_windows = {}

    for m, k_list in instance.ops_by_machine.items():
        conflicts = []
        for i in range(len(k_list)):
            for j in range(i + 1, len(k_list)):
                k  = k_list[i]
                k1 = k_list[j]
                sk  = start_times[k]
                sk1 = start_times[k1]
                overlap = not (sk  + instance.duration(k)  <= sk1 or
                               sk1 + instance.duration(k1) <= sk)
                if overlap:
                    conflicts.append((k, k1))

        if conflicts:
            conflicts_by_machine[m] = conflicts
            # For Stage 2: widen windows to allow conflict resolution
            # Use original feasible windows so Stage 2 can search
            for k in k_list:
                stage2_windows[k] = (instance.e_k[k], instance.L_k[k])
        else:
            # No conflicts: fix windows to Stage 1 result
            for k in k_list:
                s = start_times[k]
                stage2_windows[k] = (s, s)

    return start_times, conflicts_by_machine, stage2_windows


def bridge2(instance: JSSPInstance,
            stage1_times: dict,
            machine_assignments: dict) -> tuple:
    """
    Classical Bridge 2.

    Merges Stage 1 and Stage 2 assignments, verifies all constraints,
    computes makespan, and evaluates the Kurowski makespan objective h4.

    Returns:
        schedule:  dict k -> start_time (complete)
        makespan:  int
        h4:        float (Kurowski makespan penalty)
        feasible:  bool
    """
    # Merge: Stage 2 overwrites Stage 1 for machines that had conflicts
    schedule = dict(stage1_times)
    schedule.update(machine_assignments)

    feasible = True

    # Verify h1: every operation has exactly one start time (guaranteed by
    # construction but check for completeness)
    if set(schedule.keys()) != set(range(instance.K)):
        feasible = False

    # Verify h3: job ordering
    for j in range(instance.n_jobs):
        k_list = instance.ops_by_job[j]
        for idx in range(len(k_list) - 1):
            k  = k_list[idx]
            k1 = k_list[idx + 1]
            if schedule[k1] < schedule[k] + instance.duration(k):
                feasible = False

    # Verify h2: machine no-overlap
    for m, k_list in instance.ops_by_machine.items():
        for i in range(len(k_list)):
            for j in range(i + 1, len(k_list)):
                k  = k_list[i]
                k1 = k_list[j]
                sk  = schedule[k]
                sk1 = schedule[k1]
                overlap = not (sk  + instance.duration(k)  <= sk1 or
                               sk1 + instance.duration(k1) <= sk)
                if overlap:
                    feasible = False

    # Verify deadline
    for k in range(instance.K):
        if schedule[k] + instance.duration(k) > instance.T:
            feasible = False

    # Compute makespan
    makespan = max(schedule[k] + instance.duration(k)
                   for k in range(instance.K))

    # Kurowski h4 makespan objective
    # h4 = Σ_{n=1}^{J} (J+1)^{completion_time of last op of job n}
    h4 = 0.0
    J  = instance.n_jobs
    for j in range(J):
        last_k      = instance.ops_by_job[j][-1]
        completion  = schedule[last_k] + instance.duration(last_k)
        h4         += (J + 1) ** completion

    return schedule, makespan, h4, feasible


def compute_makespan(start_times: dict, instance: JSSPInstance) -> int:
    return max(start_times[k] + instance.duration(k)
               for k in range(instance.K))


# ── 9. full algorithm ─────────────────────────────────────────────────────────

def solve_jssp(instance: JSSPInstance,
               shots: int = 2000,
               max_retries: int = 20,
               verbose: bool = True) -> dict:
    """
    Constraint-stratified Grover search for JSSP.

    Stages:
        Preprocessing   classical feasibility check and window computation
        Stage 1         one Grover circuit per job (h3 ordering)
        Bridge 1        classical conflict detection and window pruning
        Stage 2         one Grover circuit per machine with conflicts (h2)
        Bridge 2        schedule reconstruction, verification, makespan

    Returns dict with keys:
        feasible:   bool
        schedule:   dict k -> start_time  (if feasible)
        makespan:   int                   (if feasible)
        h4:         float                 (Kurowski objective, if feasible)
        reason:     str                   (if infeasible)
    """
    simulator = AerSimulator()

    if verbose:
        print("=" * 60)
        print("Constraint-Stratified Grover Search for JSSP")
        print("=" * 60)
        print(f"Jobs: {instance.n_jobs}, Machines: {instance.n_machines}, "
              f"Operations: {instance.K}, Deadline T: {instance.T}")

    # ── Classical Preprocessing ───────────────────────────────────────────────
    try:
        compute_feasible_windows(instance)
    except ValueError as e:
        return {'feasible': False, 'reason': str(e)}

    if verbose:
        print("\nFeasible windows (e_k, L_k):")
        for k in range(instance.K):
            j, i, m, dur = instance.ops[k]
            print(f"  op {k} (job {j}, machine {m}, dur {dur}): "
                  f"[{instance.e_k[k]}, {instance.L_k[k]}] "
                  f"width={instance.T_k[k]}")

    # ── Stage 1: Job Ordering Search ─────────────────────────────────────────
    if verbose:
        print("\nStage 1: Job ordering search")

    job_assignments = {}

    for job_idx in range(instance.n_jobs):
        if verbose:
            k_list = instance.ops_by_job[job_idx]
            n_q = sum(instance.T_k[k] for k in k_list)
            print(f"  Job {job_idx}: {n_q} search qubits, "
                  f"operations {k_list}")

        # Compute max iterations for this job's search space
        k_list_j  = instance.ops_by_job[job_idx]
        sp_j      = max(1, int(np.prod([instance.T_k[k] for k in k_list_j])))
        max_iters = max(1, int(np.floor(np.pi / 4 * np.sqrt(sp_j))))

        success = False
        for attempt in range(max_retries):
            # Cycle through 1..max_iters to avoid permanent over-rotation
            # when M > 1 solutions exist (unknown M strategy).
            iters = (attempt % max_iters) + 1
            qc     = stage1_circuit(instance, job_idx, n_iterations=iters)
            comp   = transpile(qc, simulator)
            counts = (simulator.run(comp, shots=shots)
                      .result().get_counts())

            # Try measurements in order of frequency
            for bitstring in sorted(counts, key=counts.get, reverse=True):
                assignments = decode_job_measurement(
                    bitstring, instance, job_idx
                )
                if assignments is None:
                    continue

                # Verify h3 classically
                k_list = instance.ops_by_job[job_idx]
                valid  = all(
                    assignments[k_list[i + 1]] >=
                    assignments[k_list[i]] + instance.duration(k_list[i])
                    for i in range(len(k_list) - 1)
                )
                if valid:
                    job_assignments[job_idx] = assignments
                    if verbose:
                        print(f"    Attempt {attempt + 1} (iters={iters}): "
                              f"success -> {assignments} "
                              f"(count={counts[bitstring]})")
                    success = True
                    break

            if success:
                break

        if not success:
            return {
                'feasible': False,
                'reason': (f"Stage 1 failed for job {job_idx} "
                           f"after {max_retries} attempts")
            }

    # ── Classical Bridge 1 ────────────────────────────────────────────────────
    if verbose:
        print("\nBridge 1: conflict detection")

    try:
        stage1_times, conflicts, stage2_windows = bridge1(
            instance, job_assignments
        )
    except ValueError as e:
        return {'feasible': False, 'reason': f"Bridge 1 error: {e}"}

    if verbose:
        print(f"  Stage 1 start times: {stage1_times}")
        if conflicts:
            print(f"  Machine conflicts: {conflicts}")
        else:
            print("  No machine conflicts detected")

    # ── Stage 2: Machine Conflict Resolution ─────────────────────────────────
    machine_assignments = {}

    if conflicts:
        if verbose:
            print("\nStage 2: Machine conflict resolution")

        for machine, conflict_pairs in conflicts.items():
            k_list = instance.ops_by_machine[machine]
            wins   = {k: stage2_windows[k] for k in k_list}

            if verbose:
                print(f"  Machine {machine}: operations {k_list}, "
                      f"conflicts {conflict_pairs}")
                print(f"  Windows: { {k: wins[k] for k in k_list} }")
                n_q = sum(max(0, wins[k][1] - wins[k][0] + 1)
                          for k in k_list)
                print(f"  Search qubits: {n_q}")

            sp_m      = max(1, int(np.prod([
                max(1, wins[k][1] - wins[k][0] + 1) for k in k_list
            ])))
            max_iters = max(1, int(np.floor(np.pi / 4 * np.sqrt(sp_m))))

            success = False
            for attempt in range(max_retries):
                iters  = (attempt % max_iters) + 1
                qc     = stage2_circuit(instance, machine, wins,
                                        n_iterations=iters)
                comp   = transpile(qc, simulator)
                counts = (simulator.run(comp, shots=shots)
                          .result().get_counts())

                for bitstring in sorted(counts, key=counts.get, reverse=True):
                    assignments = decode_machine_measurement(
                        bitstring, instance, machine, wins
                    )
                    if assignments is None:
                        continue

                    # Verify h2 classically
                    valid = True
                    for i in range(len(k_list)):
                        for j in range(i + 1, len(k_list)):
                            k  = k_list[i]
                            k1 = k_list[j]
                            sk  = assignments[k]
                            sk1 = assignments[k1]
                            if not (sk  + instance.duration(k)  <= sk1 or
                                    sk1 + instance.duration(k1) <= sk):
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        machine_assignments.update(assignments)
                        if verbose:
                            print(f"    Attempt {attempt + 1} (iters={iters}): "
                                  f"success -> {assignments} "
                                  f"(count={counts[bitstring]})")
                        success = True
                        break

                if success:
                    break

            if not success:
                return {
                    'feasible': False,
                    'reason': (f"Stage 2 failed for machine {machine} "
                               f"after {max_retries} attempts")
                }

    # For machines without conflicts, use Stage 1 times directly
    for m, k_list in instance.ops_by_machine.items():
        if m not in conflicts:
            for k in k_list:
                machine_assignments[k] = stage1_times[k]

    # ── Classical Bridge 2 ────────────────────────────────────────────────────
    if verbose:
        print("\nBridge 2: schedule reconstruction and verification")

    schedule, makespan, h4, feasible = bridge2(
        instance, stage1_times, machine_assignments
    )

    if verbose:
        print(f"  Final schedule: {schedule}")
        print(f"  Makespan: {makespan}")
        print(f"  h4 objective: {h4:.2f}")
        print(f"  Feasible: {feasible}")

    if not feasible:
        return {
            'feasible': False,
            'reason':   'Bridge 2 verification failed',
            'schedule': schedule,
            'makespan': makespan
        }

    return {
        'feasible': True,
        'schedule': schedule,
        'makespan': makespan,
        'h4':       h4
    }


# ── 10. makespan optimisation (binary search over T) ─────────────────────────

def solve_jssp_optimal(jobs: list,
                       T_init: int = None,
                       shots: int = 2000,
                       max_retries: int = 20,
                       verbose: bool = True) -> dict:
    """
    Find the optimal makespan via linear scan to find any feasible T, then
    binary search to tighten T toward the minimum.

    Note: solve_jssp is probabilistic. A 'not feasible' result at some T may
    mean the algorithm failed to find a solution rather than true infeasibility.
    Binary search treats algorithm failures as infeasible, so the returned
    makespan is a valid upper bound but may not be globally optimal.
    """
    if T_init is None:
        T_init = sum(dur for job in jobs for _, dur in job)

    if verbose:
        print(f"Searching for optimal makespan, T_init={T_init}")

    # Linear scan downward to find the first feasible T
    best_result = None
    T = T_init
    while T >= 1:
        instance = JSSPInstance(jobs, T)
        result   = solve_jssp(instance, shots=shots,
                              max_retries=max_retries, verbose=False)
        if result['feasible']:
            best_result = result
            if verbose:
                print(f"  T={T}: feasible, makespan={result['makespan']}")
            break
        if verbose:
            print(f"  T={T}: infeasible")
        T -= 1

    if best_result is None:
        return {'feasible': False, 'reason': 'No feasible solution found'}

    # Binary search to tighten: invariant lo is infeasible, hi is feasible
    lo = 1
    hi = best_result['makespan'] - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        instance = JSSPInstance(jobs, mid)
        result   = solve_jssp(instance, shots=shots,
                              max_retries=max_retries, verbose=False)
        if result['feasible']:
            best_result = result
            hi = mid - 1
            if verbose:
                print(f"  T={mid}: feasible, makespan={result['makespan']}")
        else:
            lo = mid + 1
            if verbose:
                print(f"  T={mid}: infeasible (or search failed)")

    if verbose:
        print(f"  Optimal makespan: {best_result['makespan']}")

    return best_result


# ── 11. validation ────────────────────────────────────────────────────────────

def validate_2x2x2():
    """
    Standard 2x2x2 JSSP validation instance.

    Job 0: op0 on machine 0 for 1 unit, op1 on machine 1 for 2 units
    Job 1: op0 on machine 1 for 1 unit, op1 on machine 0 for 2 units

    Optimal makespan: 3
    Optimal schedule:
        op0 (j0,m0,dur1): start 0  → finishes 1
        op1 (j0,m1,dur2): start 1  → finishes 3
        op2 (j1,m1,dur1): start 0  → finishes 1
        op3 (j1,m0,dur2): start 1  → finishes 3
    All ordering and machine constraints satisfied; makespan = 3.
    """
    jobs = [
        [(0, 1), (1, 2)],
        [(1, 1), (0, 2)],
    ]

    print("=" * 60)
    print("Validation: 2x2x2 JSSP instance")
    print("Expected optimal makespan: 3")
    print("=" * 60)

    result = solve_jssp_optimal(jobs, T_init=6, shots=2000,
                                max_retries=20, verbose=True)

    print("\nFinal result:")
    for key, val in result.items():
        print(f"  {key}: {val}")

    if result.get('feasible') and result.get('makespan') == 3:
        print("\nPASS: optimal makespan found")
    else:
        print("\nFAIL: did not find optimal makespan of 3")

    return result


def validate_circuit_structure():
    """
    Structural validation: verify circuit components independently
    before running the full algorithm.
    """
    jobs = [
        [(0, 1), (1, 2)],
        [(1, 1), (0, 2)],
    ]
    T        = 5
    instance = JSSPInstance(jobs, T)

    print("=" * 60)
    print("Circuit structure validation")
    print("=" * 60)

    # 1. W-state circuits
    for T_test in [2, 3, 4, 5]:
        qc = w_state_circuit(T_test)
        print(f"W-state T={T_test}: depth={qc.depth()}, "
              f"gates={qc.count_ops()}")

    print()

    # 2. Stage 1 circuits
    for job_idx in range(instance.n_jobs):
        qc = stage1_circuit(instance, job_idx)
        print(f"Stage 1 job {job_idx}: total_qubits={qc.num_qubits}, "
              f"depth={qc.depth()}, "
              f"search_qubits={instance.job_qubit_count(job_idx)}")

    print()

    # 3. Stage 2 circuits (using original windows for structure check)
    for m in instance.ops_by_machine:
        wins = {k: (instance.e_k[k], instance.L_k[k])
                for k in instance.ops_by_machine[m]}
        qc = stage2_circuit(instance, m, wins)
        print(f"Stage 2 machine {m}: total_qubits={qc.num_qubits}, "
              f"depth={qc.depth()}, "
              f"search_qubits={instance.machine_qubit_count(m)}")

    print()

    # 4. Conflict pair enumeration
    for job_idx in range(instance.n_jobs):
        pairs = stage1_conflict_pairs(instance, job_idx)
        print(f"Stage 1 job {job_idx}: {len(pairs)} conflict pairs")

    for m in instance.ops_by_machine:
        wins  = {k: (instance.e_k[k], instance.L_k[k])
                 for k in instance.ops_by_machine[m]}
        pairs = stage2_conflict_pairs(instance, m, wins)
        print(f"Stage 2 machine {m}: {len(pairs)} conflict pairs")


if __name__ == "__main__":
    validate_circuit_structure()
    print()
    validate_2x2x2()
