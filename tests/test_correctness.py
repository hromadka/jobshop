"""Scientific regression checks; the multi-hour reference rerun is not a unit test."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import jssp_utils as utils
import qiskit_bbht as solver
import qiskit_oracle as oracle
import run_all_toys_verified as suite

ROOT = Path(__file__).resolve().parents[1]

class CorrectnessTests(unittest.TestCase):
    def test_all_readout_indices_and_spaces(self):
        for index in range(8192):
            bits = format(index, '013b')
            self.assertEqual(solver._counts_to_int({bits: 1}), index)
            self.assertEqual(solver._counts_to_int({bits[:5]+' '+bits[5:]: 1}), index)

    def test_actual_asymmetric_readout(self):
        circuit = QuantumCircuit(3, 3)
        circuit.x(0)
        circuit.measure(range(3), range(3))
        counts = solver._run_circuit(circuit, AerSimulator(max_parallel_threads=1), seed=42)
        self.assertEqual(solver._counts_to_int(counts), 1)

    def test_all_six_independent_optima_and_encoding_bijections(self):
        expected = {'th02': (3, 4, 4, 6), 'k032': (4, 3, 5, 12),
                    'g03': (10, 11, 117, 560), 'v03': (11, 6, 544, 1680),
                    'yn03': (11, 12, 468, 1680), 'p034': (13, 28, 628, 4200)}
        for name, filenames in suite.INSTANCES:
            with self.subTest(instance=name):
                inst = utils.load_benchmark_file(str(ROOT/'benchmarks-toys'/filenames[0]))[0]
                reference, _ = suite.brute_force(inst, utils)
                self.assertEqual(tuple(reference[k] for k in ('n', 'C_star', 'S', 'N_valid')), expected[name])

    def test_oracle_diffuser_padding_and_amplification(self):
        for name, mass in [('th02', .5), ('k032', 245/256), ('yn03', .9943023920059204)]:
            with self.subTest(instance=name):
                inst = utils.load_benchmark_file(str(ROOT/'benchmarks-toys'/(name+'.txt')))[0]
                optimum = {'th02': 4, 'k032': 3, 'yn03': 12}[name]
                N = 2**inst.n_perm
                marked = np.array([utils.phi_T(x, inst, optimum) for x in range(N)])
                uniform = np.ones(N, dtype=complex)/math.sqrt(N)
                phase_state = suite.apply_known_circuit(oracle.build_oracle_circuit(inst, optimum), uniform)
                np.testing.assert_allclose(phase_state, uniform*np.where(marked, -1, 1), atol=1e-10)
                after = suite.apply_known_circuit(oracle.build_diffusion_circuit(inst.n_perm), phase_state)
                probabilities = abs(after)**2
                self.assertAlmostEqual(float(probabilities.sum()), 1)
                theory = math.sin(3*math.asin(math.sqrt(marked.sum()/N)))**2
                self.assertAlmostEqual(float(probabilities[marked].sum()), theory)
                self.assertAlmostEqual(theory, mass, places=6)
                self.assertFalse(marked[inst.N_valid:].any())
                if name == 'th02':
                    np.testing.assert_allclose(probabilities, np.full(N, 1/8))
                if name == 'k032':
                    np.testing.assert_allclose(probabilities[marked], 49/256)
                    np.testing.assert_allclose(probabilities[~marked], 1/256)

    def test_sampled_probability_mapping(self):
        inst = utils.load_benchmark_file(str(ROOT/'benchmarks-toys/k032.txt'))[0]
        simulator = AerSimulator(method='statevector', max_parallel_threads=1)
        _, p = solver.compute_amplitude_distribution_qiskit(inst, 3, j_iters=1,
                        shots=8192, seed=43, simulator=simulator)
        marked = np.array([utils.phi_T(x, inst, 3) for x in range(inst.N_valid)])
        self.assertAlmostEqual(float(p[marked].sum()), 245/256, delta=.015)

    def test_reference_records_consistent(self):
        run = json.loads((ROOT/'results/verified_toys/results.json').read_text())
        self.assertTrue(run['complete'])
        expected_calls = [2, 3, 198179, 409459, 408511, 4885966]
        for record, calls in zip(run['records'], expected_calls):
            self.assertEqual(record['algorithm2']['calls'], calls)
            self.assertEqual(sum(t['calls'] for t in record['algorithm2']['threshold_trace']), calls)
            self.assertEqual(sum(t['rounds'] for t in record['algorithm2']['threshold_trace']),
                             record['algorithm2']['rounds'])
            trials = record['bbht_trials']
            self.assertEqual(len(trials), 100)
            self.assertTrue(all(t['success'] for t in trials))
            self.assertAlmostEqual(sum(t['calls'] for t in trials)/100,
                                   record['bbht']['mean_calls_successful'])

    def test_guard_before_large_enumeration(self):
        inst = utils.JSSPInstance([[(0, 1)]*6 for _ in range(3)])
        with self.assertRaisesRegex(ValueError, '8192'):
            suite.brute_force(inst, utils)

    def test_small_full_run_and_resume_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'experiment'
            command = [sys.executable, str(ROOT/'run_all_toys_verified.py'),
                       '--instances', 'th02', '--output-dir', str(out), '--audit-shots', '1024']
            def execute(extra=()):
                return subprocess.run(command+list(extra), cwd=ROOT, capture_output=True,
                                      text=True, encoding='utf-8', timeout=120)
            first = execute()
            self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
            record = json.loads((out/'th02.json').read_text())
            self.assertTrue(record['complete'])
            self.assertEqual(record['algorithm2']['C_star'], 4)
            self.assertEqual(record['bbht']['successes'], 100)
            before = (out/'th02.json').read_bytes()
            resumed = execute(['--resume'])
            self.assertEqual(resumed.returncode, 0, resumed.stdout+resumed.stderr)
            self.assertEqual((out/'th02.json').read_bytes(), before)
            mismatched = execute(['--resume', '--audit-shots', '2048'])
            self.assertEqual(mismatched.returncode, 2)
            self.assertIn('fingerprint differs', mismatched.stderr)

if __name__ == '__main__':
    unittest.main()
