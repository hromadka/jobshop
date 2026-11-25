# Quantum search experiments for job shop scheduling

Gate-model JSSP experiments using JNS encodings, a BBHT feasibility search,
and classical binary search over integer makespan thresholds. The maintained
implementation is `jssp_utils.py`, `qiskit_oracle.py`, and `qiskit_bbht.py`.
The scripts at the repository root reproduce the six toy experiments.

**Scope:** the present oracle enumerates candidate schedules classically before
building phase-marking gates. It does not reversibly decode indices and compute
makespans inside the circuit. These simulations demonstrate circuit behavior;
they do not demonstrate an end-to-end quantum speedup. Classical preprocessing,
circuit construction, compilation and simulation all have costs.

## Install

Use Python 3.10 or later in a fresh virtual environment. No Spark-specific
hardware or GPU is required. The default simulator is CPU statevector Aer.

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The reference Spark run used Python 3.12.3, Qiskit 2.6.2 and Aer 0.18.4.
Local checks also use Python 3.10, Qiskit 2.6.2 and Aer 0.17.2.
For closer reproduction, install `qiskit==2.6.2 qiskit-aer==0.18.4`.
The supported dependency ranges are not an exact lockfile; timing and sampled
outcomes can vary with package versions and platform.

## Verify and rerun the toy benchmarks

```sh
python -m unittest discover -s tests -v
python run_all_toys_verified.py --classical-only --output-dir toy_preflight_results
python run_all_toys_verified.py --instances th02 k032 --output-dir quick_results
python run_all_toys_verified.py --output-dir toy_verified_rerun
```

The last command runs all six instances and 100 independently seeded BBHT trials
per instance. On the reference Spark, Algorithm 2 alone took about four hours
across all six, dominated by p034. The explicit toy oracle guard allows at most
8192 valid sequences; it includes p034's 4200 sequences.

`run_section9.py` is an alias for the verified runner and accepts the same options.
Use `--benchmark-dir PATH` to supply other copies of the benchmark inputs.
`g03` is the paper label; its input filename is `go03.txt`.

Each run independently enumerates the complete toy state space, checks the JNS
bijection, compares makespans with a separate evaluator, audits oracle phases and
diffusion, checks sampled amplification, and independently verifies every returned
schedule. It writes JSON checkpoints, CSV and LaTeX tables. A classical preflight
does not fabricate quantum results. Resume with the identical command plus
`--resume`; changed source, inputs, parameters or environment are rejected.
An interrupted instance restarts; completed instances are retained.

BBHT rounds restart from a uniform state. Oracle calls count the sum of Grover
iterations across these rounds; they are neither circuit counts nor hardware
runtime estimates. A finite search budget returning no witness does not prove
infeasibility. Independent exhaustive enumeration establishes the toy optima.

## Corrected reference results and figures

`results/verified_toys/` contains the actual completed corrected-readout Spark
run, including individual trials, threshold traces, source hashes and LaTeX
tables. These are historical measurements, not a claim that the current sources
have identical hashes. The public runner now imports the fixed solver directly;
the original verification runner made a corrected copy during execution.

| Instance | Search qubits | Optimum | Feasible sequences | Alg. 2 calls | BBHT successes |
|---|---:|---:|---:|---:|---:|
| th02 | 3 | 4 | 4 | 2 | 100/100 |
| k032 | 4 | 3 | 5 | 3 | 100/100 |
| g03 | 10 | 11 | 117 | 198179 | 100/100 |
| v03 | 11 | 6 | 544 | 409459 | 100/100 |
| yn03 | 11 | 12 | 468 | 408511 | 100/100 |
| p034 | 13 | 28 | 628 | 4885966 | 100/100 |

```sh
python generate_section9_figures.py
python generate_paper_figures.py
```

These regenerate vector PDFs and numerical source data under `figures/`.
Both accept `--results-dir`, `--benchmark-dir` and `--output-dir`.
Section 9 plots compare the uniform distribution with the actual final successful
BBHT round at the optimum, reconstructed from its seed and checked query totals.
All basis states, including padding, are shown; probabilities sum to one.
The compact circuit PDFs are register-level schematics, not gate decompositions.

The four paper figures show a verified g03 schedule, the algorithm schematic,
k032 amplification, and threshold-by-threshold query costs. The k032 illustration
uses one Grover iteration explicitly. Figure 2's LaTeX insert uses a single column.
For th02, S/N=1/2 and every integer iteration retains uniform measurement
probabilities: it illustrates no amplification. A final successful round need
not maximize amplification and can have zero Grover iterations (as for v03).

**Readout fix:** Qiskit count strings already put the most significant bit on
the left. Both sampled-index conversions now use `int(bitstring, 2)` after
removing spaces. Reversing these strings caused the earlier incorrect probability
plots and biased candidate validation. The earlier uncorrected tables are superseded.

## Toy-only scope

The maintained source supports th02, k032, g03, v03, yn03 and p034.
Its only experiment runners are `run_all_toys_verified.py` and `run_section9.py`.
Large-benchmark neighborhood runners, screening utilities and incumbent-data
preparation tools are excluded from this source distribution.

## Repository organization and license

The notebooks, `jssp_quantum.py`, `stratified.py`, and `pre-qiskit-2.0/` are
historical exploratory material and are not the maintained paper reproduction
entry points. Their generated outputs are not experimental evidence for the
corrected runs. Virtual environments, bytecode and notebook checkpoints are
excluded from the public source tree. No local environment is distributed.

Project-owned code is released under the MIT license. Third-party benchmark
data, cited sources and dependencies retain their own terms; see
`THIRD_PARTY_NOTICES.md`. Nothing here publishes to GitHub automatically.
