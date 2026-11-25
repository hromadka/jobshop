"""Verified corrected-readout rerun of the six toy benchmarks.

Requires the original jssp_utils.py, qiskit_oracle.py, qiskit_bbht.py and Qiskit.
Imports the corrected solver directly; refuses the legacy reversed-readout source.
Each optimum is found independently by exhaustive enumeration, and each circuit
is audited before Algorithm 2 and 100 independently seeded BBHT trials are run.
Completed instances are checkpointed; --resume skips them only after checking
input/source/version/parameter fingerprints. No previous experimental results
are imported. Both requested LaTeX tables are generated from this run's records.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

INSTANCES = [('th02', ['th02.txt']), ('k032', ['k032.txt']),
             ('g03', ['go03.txt', 'g03.txt']), ('v03', ['v03.txt']),
             ('yn03', ['yn03.txt']), ('p034', ['p034.txt'])]
TRIALS, DELTA, SEED = 100, 0.10, 42

def sequences(sizes):
    remaining = list(sizes)
    prefix = []
    def visit():
        if not any(remaining):
            yield tuple(prefix)
        for job in range(len(remaining)):
            if remaining[job]:
                remaining[job] -= 1
                prefix.append(job)
                yield from visit()
                prefix.pop()
                remaining[job] += 1
    yield from visit()

def reference_schedule(sequence, jobs):
    next_op = [0]*len(jobs)
    job_ready = [0]*len(jobs)
    machine_ready = [0]*(1+max(m for job in jobs for m, _ in job))
    records = []
    for job in sequence:
        step = next_op[job]
        machine, duration = jobs[job][step]
        start = max(job_ready[job], machine_ready[machine])
        end = start+duration
        records.append((job, step, machine, start, end))
        next_op[job] += 1
        job_ready[job] = machine_ready[machine] = end
    return max(job_ready), records

def valid_records(records, jobs):
    indexed = {(j, s): (m, a, b) for j, s, m, a, b in records}
    if len(indexed) != sum(map(len, jobs)) or len(records) != len(indexed):
        return False
    for j, job in enumerate(jobs):
        for s, (m, d) in enumerate(job):
            item = indexed.get((j, s))
            if item is None or item[0] != m or item[1] < 0 or item[2]-item[1] != d:
                return False
            if s and indexed[j, s-1][2] > item[1]:
                return False
    for i, (_, _, m, a, b) in enumerate(records):
        for _, _, other_m, c, d in records[i+1:]:
            if m == other_m and not (b <= c or d <= a):
                return False
    return True

def apply_known_circuit(circuit, vector):
    """Independent exact application of the X/H/controlled-phase oracle gates."""
    import numpy as np
    from qiskit.circuit import ControlledGate
    state = np.asarray(vector, dtype=complex).copy()
    indices = np.arange(len(state))
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if gate.name == 'x':
            state = state[indices ^ (1 << qubits[0])]
        elif gate.name == 'h':
            block = state.reshape(-1, 2, 1 << qubits[0])
            a, b = block[:, 0, :].copy(), block[:, 1, :].copy()
            block[:, 0, :] = (a+b)/math.sqrt(2)
            block[:, 1, :] = (a-b)/math.sqrt(2)
        elif gate.name in ('z', 'cz'):
            mask = sum(1 << q for q in qubits)
            state[(indices & mask) == mask] *= -1
        elif isinstance(gate, ControlledGate) and gate.base_gate.name == 'p':
            if gate.ctrl_state != (1 << gate.num_ctrl_qubits)-1:
                raise ValueError('Unexpected open-control phase gate')
            mask = sum(1 << q for q in qubits)
            state[(indices & mask) == mask] *= np.exp(1j*float(gate.params[0]))
        else:
            raise ValueError(f'Unexpected audit gate: {gate.name}')
    return state*np.exp(1j*float(circuit.global_phase))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def brute_force(inst, utils):
    jobs = [[tuple(pair) for pair in job] for job in inst.jobs]
    sizes = list(map(len, jobs))
    count = math.factorial(sum(sizes))//math.prod(math.factorial(s) for s in sizes)
    if count > 8192:
        raise ValueError(f'{count} valid sequences exceeds this toy rerun limit of 8192')
    if inst.N_valid != count or inst.n_perm != max(1, (count-1).bit_length()):
        raise AssertionError('Instance state-space dimensions are inconsistent')
    reference = {}
    for sequence in sequences(sizes):
        cost, records = reference_schedule(sequence, jobs)
        if not valid_records(records, jobs):
            raise AssertionError('Independent enumeration produced an invalid schedule')
        reference[sequence] = cost
    costs, seen = [], set()
    for index in range(count):
        sequence = tuple(utils.jns_decode(index, sizes))
        if sequence in seen or sequence not in reference:
            raise AssertionError('JNS decoder is not a bijection')
        seen.add(sequence)
        cost = reference[sequence]
        if utils.cpm_makespan(list(sequence), inst) != cost:
            raise AssertionError('Original makespan calculation disagrees with independent evaluation')
        costs.append(cost)
    if seen != set(reference) or len(seen) != count:
        raise AssertionError('Enumeration did not cover every valid sequence')
    optimum = min(reference.values())
    return dict(J=len(jobs), M=inst.num_machines, K=sum(sizes), N_valid=count,
                n=inst.n_perm, C_star=optimum, S=sum(c == optimum for c in costs),
                all_sequences_valid=True, jns_bijection=True,
                histogram=dict(sorted(Counter(reference.values()).items()))), costs


def verify_index(index, deadline, inst, utils, costs, info=None):
    if index is None or not 0 <= index < len(costs):
        raise AssertionError('Solver returned an invalid or padded schedule index')
    sequence = tuple(utils.jns_decode(index, list(map(len, inst.jobs))))
    cost, records = reference_schedule(sequence, inst.jobs)
    if not valid_records(records, inst.jobs) or cost != costs[index] or cost > deadline:
        raise AssertionError('Returned schedule failed independent constraints/deadline verification')
    if info is not None:
        if tuple(info['sequencing']) != sequence:
            raise AssertionError('Returned sequencing and measured index disagree')
        offsets = [sum(len(job) for job in inst.jobs[:j]) for j in range(len(inst.jobs))]
        for job, step, _, start, _ in records:
            if info['schedule'][offsets[job]+step] != start:
                raise AssertionError('Returned start times differ from independently computed schedule')
    return cost, records


def audit_circuits(inst, reference, costs, oracle, solver, shots):
    import numpy as np
    n, optimum = reference['n'], reference['C_star']
    N = 2**n
    marked = np.zeros(N, dtype=bool)
    marked[:len(costs)] = np.asarray(costs) <= optimum
    uniform = np.ones(N, dtype=complex)/math.sqrt(N)
    circuit = oracle.build_oracle_circuit(inst, optimum)
    xmask = 0
    for instruction in circuit.data:
        if instruction.operation.name == 'x':
            xmask ^= 1 << circuit.find_bit(instruction.qubits[0]).index
    phase_state = apply_known_circuit(circuit, uniform)
    if xmask or not np.allclose(phase_state, uniform*np.where(marked, -1, 1), atol=1e-9):
        raise AssertionError('Oracle does not phase-mark exactly the feasible states')
    diffuser = oracle.build_diffusion_circuit(n)
    rng = np.random.default_rng(SEED)
    vector = rng.normal(size=N)+1j*rng.normal(size=N)
    vector /= np.linalg.norm(vector)
    expected = 2*vector.mean()-vector
    actual = apply_known_circuit(diffuser, vector)
    phase = np.vdot(expected, actual)
    if not math.isclose(abs(phase), 1, abs_tol=1e-9) or not np.allclose(actual, phase*expected, atol=1e-9):
        raise AssertionError('Diffuser differs from a reflection beyond global phase')
    after = apply_known_circuit(diffuser, phase_state)
    exact = float(np.sum(np.abs(after[marked])**2))
    theory = math.sin(3*math.asin(math.sqrt(reference['S']/N)))**2
    if not math.isclose(exact, theory, abs_tol=1e-9):
        raise AssertionError('Exact amplification differs from Grover prediction')
    simulator = solver._get_simulator()
    _, probabilities = solver.compute_amplitude_distribution_qiskit(
        inst, optimum, j_iters=1, simulator=simulator, shots=shots, seed=SEED+1)
    if len(probabilities) != len(costs):
        raise AssertionError('Probability array has an unexpected dimension')
    measured = float(np.sum(probabilities[marked[:len(costs)]]))
    tolerance = 6*math.sqrt(theory*(1-theory)/shots)+6/shots
    if abs(measured-theory) > tolerance:
        raise AssertionError('Corrected sampled amplification differs from exact prediction')
    return dict(oracle_all_indices_pass=True, diffuser_pass=True,
                exact_one_iteration_mass=exact, theory=theory, sampled_mass=measured,
                shots=shots, sample_pass=True)


def tables(out, records, classical_only=False):
    reference_rows = [r for r in records if 'reference' in r]
    instance_lines = []
    for r in reference_rows:
        b = r['reference']
        instance_lines.append(f"{r['name']:<4} & {b['n']:>2} & {b['C_star']:>2} & {b['S']:>3} " + r'\\')
    instance_tex = r'''\begin{table}[!ht]
\centering\small
\begin{tabular}{lrrr}
\toprule
Instance & Qubits ($n$) & $C^*_{\max}$ & $S$ \\
\midrule
'''+'\n'.join(instance_lines)+r'''
\bottomrule
\end{tabular}
\caption{Search-register qubit count
($n=\lceil\log_2 N_{\mathrm{valid}}\rceil$), optimal makespan
($C^*_{\max}$, established by independent exhaustive classical enumeration),
and number of feasible JNS sequencings at $T=C^*_{\max}$ ($S$)
for each toy benchmark instance.}
\label{tab:instance-summary}
\end{table}
'''
    (out/'instance_summary.tex').write_text(instance_tex, encoding='utf-8')
    completed = [r for r in records if r.get('complete')]
    if classical_only:
        return
    lines = []
    for r in completed:
        calls = f"{r['algorithm2']['calls']:,}".replace(',', '{,}')
        lines.append(f"{r['name']:<4} & {calls} & {r['bbht']['successes']}/{TRIALS} " + r'\\')
    all_complete = len(completed) == len(INSTANCES)
    success_sentence = ('All 100 trials succeeded for every instance.\n'
                        if all_complete and all(r['bbht']['successes'] == TRIALS for r in completed)
                        else '')
    alg_tex = ('' if all_complete else '% PARTIAL RUN: unfinished instances are omitted.\n')+r'''\begin{table}[!ht]
\centering\small
\begin{tabular}{lrr}
\toprule
Instance & Alg.~2 oracle calls & BBHT successes (100 trials) \\
\midrule
'''+'\n'.join(lines)+r'''
\bottomrule
\end{tabular}
\caption{Algorithm~2 total oracle calls across all binary-search
iterations and BBHT Algorithm~1 successes over 100 independently seeded
trials at $T=C^*_{\max}$ ($\delta=0.10$, $\lambda=6/5$),
using the corrected measurement-to-index mapping.
'''+success_sentence+r'''Every returned schedule passed independent classical checks of
job precedence, machine non-overlap, and the makespan threshold.
Totals include unsuccessful bounded feasibility searches at suboptimal
thresholds (see Section~\ref{sec:resources}).}
\label{tab:alg-results}
\end{table}
'''
    target = out/('algorithm_results.tex' if all_complete else 'algorithm_results_partial.tex')
    target.write_text(alg_tex, encoding='utf-8')
    if all_complete and (out/'algorithm_results_partial.tex').exists():
        (out/'algorithm_results_partial.tex').unlink()


def summaries(out, records, manifest, classical_only):
    completed = [r for r in records if r.get('complete')]
    write_json(out/'results.json', dict(manifest=manifest, classical_only=classical_only,
                                      complete=(len(completed) == len(INSTANCES)), records=records))
    with (out/'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Instance', 'n', 'C_star', 'S', 'Alg2_calls', 'Alg2_rounds',
                         'Alg2_elapsed_s', 'BBHT_successes', 'BBHT_trials', 'BBHT_mean_calls'])
        for r in completed:
            b, a, t = r['reference'], r['algorithm2'], r['bbht']
            writer.writerow([r['name'], b['n'], b['C_star'], b['S'], a['calls'], a['rounds'],
                             a['elapsed_s'], t['successes'], TRIALS, t['mean_calls_successful']])
    text = [f"Host: {manifest['host']}", f'Fresh corrected-readout rerun; seed={SEED}, delta={DELTA}',
            f'Completed: {len(completed)}/{len(INSTANCES)}',
            'Instance n C* S Alg2_calls Alg2_rounds Alg2_seconds BBHT_successes BBHT_mean_calls']
    for r in completed:
        b, a, t = r['reference'], r['algorithm2'], r['bbht']
        mean = t['mean_calls_successful']
        text.append(f"{r['name']} {b['n']} {b['C_star']} {b['S']} {a['calls']} {a['rounds']} "
                    f"{a['elapsed_s']:.6f} {t['successes']}/{TRIALS} {mean}")
    (out/'summary.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')
    tables(out, records, classical_only)


def main():
    global INSTANCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--benchmark-dir', type=Path, default=Path(__file__).resolve().parent/'benchmarks-toys')
    parser.add_argument('--output-dir', type=Path, default=Path('toy_verified_rerun'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--classical-only', action='store_true', help='Preflight only; no experimental table is generated')
    parser.add_argument('--audit-shots', type=int, default=8192)
    parser.add_argument('--instances', nargs='+', choices=[name for name, _ in INSTANCES],
                        help='Optional subset, in the requested order')
    args = parser.parse_args()
    if args.instances:
        if len(set(args.instances)) != len(args.instances):
            parser.error('Duplicate instance names')
        choices = dict(INSTANCES)
        INSTANCES = [(name, choices[name]) for name in args.instances]
    if args.audit_shots < 1024:
        parser.error('Use at least 1024 audit shots')
    folder, out = args.solver_dir.resolve(), args.output_dir.resolve()
    sys.path.insert(0, str(folder))
    import jssp_utils as utils
    paths = {}
    for name, filenames in INSTANCES:
        matches = [args.benchmark_dir/f for f in filenames if (args.benchmark_dir/f).is_file()]
        if not matches:
            parser.error(f'Missing benchmark for {name}: expected {filenames}')
        if len({sha(p) for p in matches}) > 1:
            parser.error(f'Ambiguous differing files for {name}: {matches}')
        paths[name] = matches[0].resolve()
    sources = {name: sha(folder/name) for name in ['jssp_utils.py', 'qiskit_oracle.py', 'qiskit_bbht.py']}
    versions = {}
    solver = oracle = None
    corrected_source = (folder/'qiskit_bbht.py').read_text(encoding='utf-8')
    replacement_count = corrected_source.count('int(bitstr[::-1], 2)')
    if replacement_count:
        parser.error('Legacy reversed readout detected; update qiskit_bbht.py first')
    if not args.classical_only:
        import qiskit
        import qiskit_aer
        versions = dict(qiskit=qiskit.__version__, aer=qiskit_aer.__version__)
    manifest = dict(instances=[name for name, _ in INSTANCES], host=platform.node(), platform=platform.platform(), python=sys.version,
                    runner_sha256=sha(Path(__file__)), source_sha256=sources,
                    benchmark_sha256={name: sha(p) for name, p in paths.items()},
                    versions=versions, seed=SEED, delta=DELTA, lambda_=6/5, trials=TRIALS,
                    audit_shots=args.audit_shots, classical_only=args.classical_only,
                    corrected_reversal_sites=replacement_count,
                    corrected_solver_sha256=hashlib.sha256(corrected_source.encode('utf-8')).hexdigest())
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            parser.error('Output directory is nonempty; use --resume or choose a new directory')
        if not (out/'manifest.json').is_file() or json.loads((out/'manifest.json').read_text()) != manifest:
            parser.error('Resume fingerprint differs: source, data, parameters or environment changed')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', manifest)
    if not args.classical_only:
        import qiskit_oracle as oracle
        import qiskit_bbht as solver
        for index in range(8192):
            if solver._counts_to_int({format(index, '013b'): 1}) != index:
                raise AssertionError('Corrected decoder still misinterprets measurements')
        print('Readout PASS for all 8192 indices', flush=True)
    records, costs_by_name, instances = [], {}, {}
    # Complete all independent reference checks before the long quantum reruns.
    for name, _ in INSTANCES:
        inst = utils.load_benchmark_file(str(paths[name]))[0]
        reference, costs = brute_force(inst, utils)
        instances[name], costs_by_name[name] = inst, costs
        target = out/f'{name}.json'
        if args.resume and target.is_file():
            record = json.loads(target.read_text())
            if record['reference'] != json.loads(json.dumps(reference)):
                raise AssertionError('Saved reference differs from current independent enumeration')
        else:
            record = dict(name=name, reference=reference, complete=False)
        records.append(record)
        write_json(target, record)
        print(f"{name}: reference C*={reference['C_star']} S={reference['S']} "
              f"n={reference['n']} N_valid={reference['N_valid']} PASS", flush=True)
    summaries(out, records, manifest, args.classical_only)
    if args.classical_only:
        print(f'Preflight complete. Instance table: {out / "instance_summary.tex"}', flush=True)
        return
    for record in records:
        name = record['name']
        if record.get('complete'):
            print(f'{name}: resuming past completed instance', flush=True)
            continue
        inst, costs = instances[name], costs_by_name[name]
        target = out/f'{name}.json'
        try:
            record.pop('error', None)
            reference = record['reference']
            record['circuit_audit'] = audit_circuits(inst, reference, costs, oracle, solver, args.audit_shots)
            print(f'{name}: oracle, diffuser and sampled amplification PASS', flush=True)
            write_json(target, record)
            simulator = solver._get_simulator()
            trace = []
            original_search = solver.bbht_search_qiskit
            def traced_search(*positional, **kwargs):
                answer = original_search(*positional, **kwargs)
                threshold = positional[1] if len(positional) > 1 else kwargs['T']
                trace.append(dict(T=threshold, x=answer[0], calls=answer[1], rounds=answer[2],
                                  reason=answer[3].get('reason')))
                record['algorithm2_progress'] = trace
                write_json(target, record)
                return answer
            solver.bbht_search_qiskit = traced_search
            print(f'{name}: running corrected Algorithm 2...', flush=True)
            try:
                found, info, calls, rounds, elapsed = solver.find_optimal_makespan_qiskit(
                    inst, delta=DELTA, seed=SEED, verbose=True, simulator=simulator)
            finally:
                solver.bbht_search_qiskit = original_search
            if found != reference['C_star'] or info is None:
                raise AssertionError('Algorithm 2 did not recover the independently proven optimum')
            verify_index(info['x'], found, inst, utils, costs, info)
            record['algorithm2'] = dict(C_star=found, calls=calls, rounds=rounds,
                                        elapsed_s=elapsed, independent_verification=True,
                                        threshold_trace=trace, schedule_info=info)
            print(f'{name}: C*={found} valid=True calls={calls} rounds={rounds} elapsed={elapsed:.6f}s', flush=True)
            trials = []
            for trial in range(TRIALS):
                index, cl, rd, _ = solver.bbht_search_qiskit(
                    inst, reference['C_star'], delta=DELTA, seed=SEED+trial, simulator=simulator)
                if index is not None:
                    verify_index(index, reference['C_star'], inst, utils, costs)
                trials.append(dict(trial=trial+1, seed=SEED+trial, x=index,
                                   success=index is not None, calls=cl, rounds=rd))
                record['bbht_trials'] = trials
                if (trial+1) % 10 == 0:
                    write_json(target, record)
                    print(f'{name}: BBHT {trial+1}/{TRIALS}', flush=True)
            successful = [r for r in trials if r['success']]
            record['bbht'] = dict(successes=len(successful), trials=TRIALS,
                                  mean_calls_successful=(sum(r['calls'] for r in successful)/len(successful)
                                                         if successful else None),
                                  all_returned_schedules_verified=True)
            record['complete'] = True
            write_json(target, record)
            summaries(out, records, manifest, False)
        except BaseException as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            write_json(target, record)
            summaries(out, records, manifest, False)
            raise
    summaries(out, records, manifest, False)
    print((out/'summary.txt').read_text(encoding='utf-8'), flush=True)
    print(f'LaTeX tables: {out / "instance_summary.tex"} and {out / "algorithm_results.tex"}', flush=True)


if __name__ == '__main__':
    main()
