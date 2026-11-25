"""Publication figures from verified Spark records and exact original circuits."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle, Arc
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import argparse

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=ROOT/'results/verified_toys')
parser.add_argument('--benchmark-dir', type=Path, default=ROOT/'benchmarks-toys')
parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/toy_figures')
args = parser.parse_args()
OUT = args.output_dir.resolve()
QA = OUT/'previews'
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
RECORDS = args.results_dir.resolve()
BENCHMARKS = args.benchmark_dir.resolve()
SOURCES = SOURCE = ROOT
sys.path.insert(0, str(ROOT))
import run_all_toys_verified as audit
import jssp_utils as utils

def same_text_hash(path, expected):
    # Git checkout line endings may differ; preserve the recorded byte hash.
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return expected in {hashlib.sha256(x).hexdigest()
                       for x in (raw, lf, lf.replace(b'\n', b'\r\n'))}
import qiskit_oracle as oracle
from qiskit import QuantumCircuit
data = json.loads((RECORDS/'results.json').read_text())
manifest = data['manifest']
assert data['complete'] and len(data['records']) == 6
plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
names = {'th02': 'Hromadka 2x2', 'k032': 'Kurowski 3x2',
         'g03': 'Google OR-Tools 3x3', 'v03': 'Venturelli 3x3',
         'yn03': 'Yamada-Nakano 3x3', 'p034': 'Pinedo 3x4'}
provenance, pages, captions = [], [], []


def final_round(record):
    b = record['reference']
    trace = record['algorithm2']['threshold_trace']
    assert sum(t['calls'] for t in trace) == record['algorithm2']['calls']
    assert sum(t['rounds'] for t in trace) == record['algorithm2']['rounds']
    t = [t for t in trace if t['T'] == b['C_star'] and t['x'] is not None][-1]
    assert t['x'] == record['algorithm2']['schedule_info']['x']
    seed = manifest['seed']+t['T']
    rng = np.random.default_rng(seed)
    width, js = 1.0, []
    for _ in range(t['rounds']):
        js.append(int(rng.integers(0, math.ceil(width))))
        width = min(manifest['lambda_']*width, math.sqrt(2**b['n']))
    assert sum(js) == t['calls']
    return dict(T=t['T'], seed=seed, round=t['rounds'], j=js[-1],
                calls_in_threshold_search=t['calls'], sampled_x=t['x'],
                circuit_seed=(seed*10000+t['rounds']-1) % (2**31),
                reconstruction='Replayed original numpy.default_rng growth schedule; '
                'sum of sampled iterations equals recorded threshold oracle calls.')


def probability_figure(name, b, final, p0, pf, feasible):
    N, valid, S = len(p0), b['N_valid'], b['S']
    j = final['j']
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.15), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.30, top=.75, wspace=.13)
    fig.suptitle(f"{name}: {names[name]}", fontsize=13, y=.975)
    fig.text(.5, .875,
             rf"$T={b['C_star']}$, $n={b['n']}$, $N_{{\mathrm{{valid}}}}={valid}$, "
             rf"$N={N}$, $S={S}$", ha='center', fontsize=10)
    colors = np.full(N, '#a0a0a0', dtype=object)
    colors[feasible] = '#303030'
    colors[valid:] = '#d0d0d0'
    upper = float(max(np.max(p0), np.max(pf)))*1.20
    x = np.arange(N)
    titles = [r'(a) Initial uniform state ($j=0$)',
              rf'(b) Final successful round ($j={j}$)']
    for panel, (ax, probabilities, title) in enumerate(zip(axes, [p0, pf], titles)):
        ax.axvspan(valid-.5, N-.5, facecolor='#f2f2f2', zorder=0)
        if N <= 32:
            bars = ax.bar(x, probabilities, width=.76, color=colors,
                          edgecolor='#666666', linewidth=.45, zorder=3)
            for index, bar in enumerate(bars):
                if feasible[index]:
                    bar.set_hatch('///')
                    bar.set_edgecolor('white')
                elif index >= valid:
                    bar.set_hatch('..')
            ax.set_xticks(x)
        else:
            segments = np.stack((np.column_stack((x, np.zeros(N))),
                                 np.column_stack((x, probabilities))), axis=1)
            ax.add_collection(LineCollection(segments, colors=colors,
                                             linewidths=.35, zorder=3))
            ticks = np.unique(np.round(np.linspace(0, N-1, 6)).astype(int))
            ax.set_xticks(ticks)
        ax.axvline(valid-.5, color='#666666', linestyle=':', linewidth=.8, zorder=4)
        ax.set_xlim(-.6, N-.4)
        ax.set_ylim(0, upper)
        ax.set_title(title, pad=12, fontsize=10)
        ax.set_xlabel('Basis-state index $x$ (including padding)', labelpad=7, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y', color='#dddddd', linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        if upper < .01:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
        center = .29 if panel == 0 else .76
        mass = float(probabilities[feasible].sum())
        pad_mass = float(probabilities[valid:].sum())
        fig.text(center, .155, f'Feasible mass = {mass:.6f} ({100*mass:.2f}%)',
                 ha='center', fontsize=9)
        fig.text(center, .105,
                 rf'$\sum_{{x=0}}^{{{N-1}}}p(x)={probabilities.sum():.6f}$'
                 + f'; padding mass = {pad_mass:.6f}', ha='center', fontsize=8.5)
    axes[0].set_ylabel('Exact ideal measurement probability', labelpad=8, fontsize=9)
    handles = [Patch(facecolor='#303030', hatch='///', edgecolor='white', label=f'Feasible ({S})'),
               Patch(facecolor='#a0a0a0', edgecolor='#666666', label=f'Valid, infeasible ({valid-S})'),
               Patch(facecolor='#d0d0d0', hatch='..', edgecolor='#666666', label=f'Padding ({N-valid})')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005),
               ncol=3, frameon=False, fontsize=8.5, handlelength=2)
    path = OUT/f'{name}_probs.pdf'
    fig.savefig(path, metadata={'Title': f'{name}: initial and actual final BBHT-round probabilities',
                                'Subject': 'Exact ideal probabilities; all basis states included; corrected readout.'})
    fig.savefig(QA/f'{name}_probs_matplotlib.png', dpi=160)
    plt.close(fig)
    return path


def circuit_figure(name, b, final, oc, dc):
    n, j = b['n'], final['j']
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(j):
        qc.append(oc.to_gate(label=rf'$O_{{{b["C_star"]}}}$'), range(n))
        qc.append(dc.to_gate(label='$D$'), range(n))
    qc.measure(range(n), range(n))
    assert qc.num_qubits == n
    fig, ax = plt.subplots(figsize=(7.4, 2.45))
    fig.subplots_adjust(left=.025, right=.985, top=.70, bottom=.22)
    ax.set_xlim(0, 10)
    ax.set_ylim(-.65, .65)
    ax.axis('off')
    fig.suptitle(f'{name}: actual final successful BBHT round', fontsize=12, y=.98)
    fig.text(.5, .835,
             rf"$T={b['C_star']}$, $j={j}$, $n={n}$; "
             f"round {final['round']} of the optimal-threshold search", ha='center', fontsize=9)
    ax.text(.12, 0, rf'$|0\rangle^{{\otimes {n}}}$', ha='left', va='center', fontsize=12)
    ax.plot([1.15, 8.05], [0, 0], color='black', linewidth=1.15, zorder=1)
    ax.plot([1.48, 1.60], [-.08, .08], color='black', linewidth=1.0)
    ax.text(1.55, .21, str(n), ha='center', fontsize=9)
    def box(center, label, width=.95):
        ax.add_patch(Rectangle((center-width/2, -.22), width, .44,
                               facecolor='white', edgecolor='black', linewidth=1.1, zorder=3))
        ax.text(center, 0, label, ha='center', va='center', fontsize=12, zorder=4)
    box(2.5, rf'$H^{{\otimes {n}}}$', 1.2)
    if j:
        for k in range(j):
            # Recorded runs have j=1; their circuit objects above retain exact structure.
            box(4.25+2.2*k, rf'$O_{{{b["C_star"]}}}$')
            box(5.65+2.2*k, '$D$')
        ax.add_patch(Rectangle((3.55, -.42), 2.8, .89, fill=False,
                               linestyle='--', linewidth=.7, edgecolor='#666666'))
        ax.text(4.95, .52, rf'$j_{{\mathrm{{final}}}}={j}$', ha='center', fontsize=9)
    else:
        ax.text(4.9, .20, 'uniform sampling (j=0)', ha='center', fontsize=9)
    center = 7.4
    ax.add_patch(Rectangle((center-.48, -.26), .96, .52,
                           facecolor='white', edgecolor='black', linewidth=1.1, zorder=3))
    ax.add_patch(Arc((center, -.05), .60, .37, theta1=0, theta2=180,
                     edgecolor='black', linewidth=1.0, zorder=4))
    ax.plot([center, center+.19], [-.05, .17], color='black', linewidth=1.0, zorder=4)
    ax.plot([7.88, 8.62], [-.025, -.025], color='black', linewidth=.8)
    ax.plot([7.88, 8.62], [.025, .025], color='black', linewidth=.8)
    ax.annotate('', xy=(8.72, 0), xytext=(8.50, 0), arrowprops=dict(arrowstyle='->', lw=.8))
    ax.text(8.9, 0, rf'$x={final["sampled_x"]}$', ha='left', va='center', fontsize=11)
    if j:
        note = (r'Each wire bundles all $n$ qubits. $O_T$: feasibility phase oracle; $D$: diffusion.'
                '\nBoxes denote subcircuits; the output is the recorded successful readout.')
    else:
        note = (r'Each wire bundles all $n$ qubits. The final round used no oracle or diffusion.'
                '\nThe output is the recorded successful readout from uniform sampling.')
    fig.text(.5, .045, note, ha='center', va='bottom', fontsize=8.5, linespacing=1.6)
    path = OUT/f'{name}_circuit.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=.12,
                metadata={'Title': f'{name}: final-round BBHT circuit, j={j}',
                          'Subject': 'Register-level circuit schematic with oracle/diffusion subcircuit boxes.'})
    plt.close(fig)
    return path


for record in data['records']:
    name, b = record['name'], record['reference']
    filename = 'go03.txt' if name == 'g03' else name+'.txt'
    path = BENCHMARKS/filename
    assert same_text_hash(path, manifest['benchmark_sha256'][name]), name
    inst = utils.load_benchmark_file(str(path))[0]
    reference, costs = audit.brute_force(inst, utils)
    for field in ('n', 'C_star', 'S', 'N_valid'):
        assert reference[field] == b[field]
    final = final_round(record)
    N, S = 2**b['n'], b['S']
    feasible = np.zeros(N, dtype=bool)
    feasible[:inst.N_valid] = np.asarray(costs) <= b['C_star']
    initial = np.ones(N, dtype=complex)/math.sqrt(N)
    state = initial.copy()
    oc = oracle.build_oracle_circuit(inst, b['C_star'])
    dc = oracle.build_diffusion_circuit(b['n'])
    marked_state = audit.apply_known_circuit(oc, initial)
    assert np.allclose(marked_state, initial*np.where(feasible, -1, 1), atol=1e-9)
    for _ in range(final['j']):
        state = audit.apply_known_circuit(oc, state)
        state = audit.apply_known_circuit(dc, state)
    p0, pf = np.abs(initial)**2, np.abs(state)**2
    assert np.isclose(p0.sum(), 1, atol=1e-10) and np.isclose(pf.sum(), 1, atol=1e-10)
    expected_mass = math.sin((2*final['j']+1)*math.asin(math.sqrt(S/N)))**2
    assert math.isclose(float(pf[feasible].sum()), expected_mass, abs_tol=1e-9)
    if name == 'th02':
        assert np.allclose(p0, 1/8) and np.allclose(pf, 1/8)
    if name == 'k032' and final['j'] == 1:
        assert np.allclose(pf[feasible], 49/256)
        assert np.allclose(pf[~feasible], 1/256)
    probs_pdf = probability_figure(name, b, final, p0, pf, feasible)
    circuit_pdf = circuit_figure(name, b, final, oc, dc)
    pages.extend([circuit_pdf, probs_pdf])
    with (OUT/f'{name}_probabilities.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['basis_index', 'category', 'p_initial', 'p_final_round'])
        for index in range(N):
            kind = 'padding' if index >= inst.N_valid else ('feasible' if feasible[index] else 'valid_infeasible')
            writer.writerow([index, kind, f'{p0[index]:.17g}', f'{pf[index]:.17g}'])
    provenance.append(dict(name=name, reference=b, final_round=final,
                           p_initial_sum=float(p0.sum()), p_final_sum=float(pf.sum()),
                           initial_feasible_mass=float(p0[feasible].sum()),
                           final_feasible_mass=float(pf[feasible].sum()),
                           final_padding_mass=float(pf[inst.N_valid:].sum()),
                           includes_all_basis_states=True,
                           probability_method='Exact ideal statevector, evaluated from the original oracle/diffuser gates; not shot counts.'))
    if name == 'th02':
        explanation = (r'Here $S/N=4/8=1/2$, so every basis state has probability $1/8$ '
                       r'in both panels. The feasible mass remains $1/2$: '
                       r'this instance illustrates no measurement-probability amplification.')
    elif final['j'] == 0:
        explanation = (r'The recorded final successful round used $j=0$, '
                       r'so its premeasurement distribution is uniform and identical to the initial distribution.')
    else:
        explanation = (rf'The feasible mass increases from ${S}/{N}={float(p0[feasible].sum()):.6f}$ '
                       rf'to ${float(pf[feasible].sum()):.6f}$ in this recorded round.')
    captions.append(r'\begin{figure*}[t]'+'\n'+r'\centering'+'\n'
                    +rf'\includegraphics[width=\textwidth]{{{name}_probs.pdf}}'+'\n'
                    +rf'\caption{{Exact ideal measurement probabilities for {name} at $T={b["C_star"]}$. '
                    +rf'The left panel shows $j=0$; the right shows $j={final["j"]}$, '
                    +r'the actual final successful BBHT round in the corrected Algorithm~2 rerun. '
                    +explanation+' '
                    +rf'All ${N}$ basis states, including ${N-inst.N_valid}$ padded states, are displayed; '
                    +r'each panel sums to one. These are unconditional premeasurement probabilities, '
                    +r'not probabilities conditioned on successful readout.}'+'\n'
                    +rf'\label{{fig:{name}-final-probabilities}}'+'\n'+r'\end{figure*}'+'\n')
    print(f'{name}: final_j={final["j"]}, P_feasible={float(pf[feasible].sum()):.9f}; PDFs saved', flush=True)

(OUT/'figure_captions.tex').write_text('% Requires \\usepackage{graphicx}\n\n'+'\n'.join(captions), encoding='utf-8')
figure9 = r'''% Replacement for the amplification example (figure numbering is automatic).
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{k032_probs.pdf}
\caption{Amplitude amplification for k032 at $T=C^*_{\max}=3$.
There are $N=16$ basis states: five feasible states, seven valid infeasible
states, and four padded states. Initially ($j=0$), every state has probability
$1/16$, giving feasible mass $5/16=0.3125$. After $j=1$, the actual final
successful BBHT round at this threshold in the corrected Algorithm~2 rerun,
each feasible state has probability $49/256$ and each unmarked state,
including padding, has probability $1/256$. Thus the feasible mass is
$245/256\approx0.9570$. All states are shown and each panel sums to one.
The bars show exact ideal premeasurement probabilities.}
\label{fig:amplification}
\end{figure*}
'''
if any(p['name'] == 'k032' and p['final_round']['j'] == 1 for p in provenance):
    (OUT/'figure9_replacement.tex').write_text(figure9, encoding='utf-8')
(OUT/'figure_provenance.json').write_text(json.dumps(dict(
    source='Completed corrected-readout Spark run', manifest=manifest, figures=provenance,
    interpretation='Final means the last successful round at the optimal threshold, '
    'not the sum of Grover iterations across BBHT restarts or binary-search thresholds.'), indent=2), encoding='utf-8')
(OUT/'README.txt').write_text('''CORRECTED TOY BENCHMARK FIGURES

Each *_circuit.pdf is a compact register-level circuit schematic using the actual final
successful BBHT round at the optimal threshold in the completed Spark run.
Each quantum wire bundles all n search qubits. Oracle and diffusion boxes
denote their original subcircuits; gate-level decompositions are not shown.

Each *_probs.pdf compares j=0 with that round's actual iteration count.
Probabilities are exact ideal statevector probabilities, NOT new sampled
measurement counts and NOT conditioned on the successful observed readout.
All 2^n states are displayed, including padding. CSVs contain every probability.
Benchmark fingerprints match the completed run (allowing Git newline conversion).
The current oracle and diffuser are independently checked against the reference.
Iteration schedules were reconstructed from seeds/rounds and checked against
the recorded oracle-call totals. See figure_provenance.json.

Actual final iteration counts are recorded in figure_provenance.json.
The final successful round need not maximize amplification; any j=0 round
retains the initial uniform distribution.

FIGURE 9 CORRECTION
For th02, S/N=1/2: every integer Grover iteration retains uniform measurement
probabilities 1/8. Its caption makes no amplification claim.
Use k032_probs.pdf and figure9_replacement.tex for the amplification example:
feasible mass increases from 5/16 to 245/256. Its four padded states are shown.
figure_captions.tex contains captions for all six probability figures.
''', encoding='utf-8')

print('Saved 12 PDFs, captions, probability CSVs and provenance.', flush=True)
