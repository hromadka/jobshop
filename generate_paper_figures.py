"""Four review figures based solely on the completed verified Spark experiment."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import textwrap
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Arc

import argparse

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=ROOT/'results/verified_toys')
parser.add_argument('--benchmark-dir', type=Path, default=ROOT/'benchmarks-toys')
parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/paper_figures')
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
run = json.loads((RECORDS/'results.json').read_text())
records = {r['name']: r for r in run['records']}
assert run['complete'] and all(r['complete'] for r in records.values())

plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
figures = []


def save(fig, stem, title, caption, focus, height_hint):
    fig.savefig(OUT/(stem+'.pdf'), metadata={'Title': title,
                    'Subject': 'Draft publication figure for review; verified Spark toy benchmark results.'})
    fig.savefig(OUT/(stem+'.svg'))
    fig.savefig(QA/(stem+'_canvas.png'), dpi=150)
    plt.close(fig)
    figures.append(dict(stem=stem, title=title, caption=caption,
                        review_focus=focus, height_hint=height_hint))


def gantt():
    r = records['g03']
    path = BENCHMARKS/'go03.txt'
    assert same_text_hash(path, run['manifest']['benchmark_sha256']['g03'])
    inst = utils.load_benchmark_file(str(path))[0]
    info = r['algorithm2']['schedule_info']
    index = info['x']
    sequence = utils.jns_decode(index, [len(job) for job in inst.jobs])
    cost, ops = audit.reference_schedule(sequence, inst.jobs)
    assert cost == r['reference']['C_star'] == 11 and audit.valid_records(ops, inst.jobs)
    offsets = [sum(len(job) for job in inst.jobs[:j]) for j in range(len(inst.jobs))]
    for job, step, machine, start, end in ops:
        assert info['schedule'][str(offsets[job]+step)] == start
    with (OUT/'figure_01_schedule_data.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['job', 'operation', 'machine', 'start', 'finish'])
        writer.writerows(sorted(ops))
    colors = ['#dedede', '#939393', '#343434']
    textcolors = ['black', 'black', 'white']
    hatches = ['', '///', '..']
    fig = plt.figure(figsize=(7.4, 4.55))
    fig.suptitle('g03: an independently verified optimal schedule', fontsize=13, y=.98)
    fig.text(.5, .91, r'$C_{\max}=11$; 3 jobs, 3 machines, 8 operations', ha='center', fontsize=10)
    handles = [Patch(facecolor=colors[j], edgecolor='#555555', hatch=hatches[j],
                     label=rf'Job $J_{j}$') for j in range(3)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.885),
               ncol=3, frameon=False, fontsize=9)
    ax = fig.add_axes([.09, .385, .86, .385])
    for job, step, machine, start, end in ops:
        y = 2-machine
        ax.barh(y, end-start, left=start, height=.60, color=colors[job],
                edgecolor='#555555', linewidth=.65, hatch=hatches[job], zorder=3)
        ax.text((start+end)/2, y, rf'$J_{job}O_{step}$', ha='center', va='center',
                fontsize=9.5, color=textcolors[job], zorder=4)
    # Show one job's precedence arrows to avoid crossing unrelated operations.
    job0 = sorted((o for o in ops if o[0] == 0), key=lambda o:o[1])
    for u, v in zip(job0, job0[1:]):
        y1, y2 = 2-u[2], 2-v[2]
        ax.annotate('', xy=(v[3], y2+.34), xytext=(u[4], y1-.34),
                    arrowprops=dict(arrowstyle='->', linestyle='--', color='black',
                                    linewidth=.9, connectionstyle='arc3,rad=0.10'), zorder=5)
    ax.axvline(11, color='black', linestyle='--', linewidth=1.1)
    ax.text(11.08, 2.48, r'$C_{\max}=11$', fontsize=9, ha='left', va='center')
    ax.set_xlim(-.15, 12.6)
    ax.set_ylim(-.6, 2.8)
    ax.set_yticks([2,1,0], ['$M_0$', '$M_1$', '$M_2$'])
    ax.set_xticks(range(12))
    ax.set_xlabel('Time', labelpad=5)
    ax.grid(axis='x', color='#dddddd', linewidth=.6, zorder=0)
    fig.text(.09, .282, 'Dashed arrows illustrate precedence for job $J_0$.', fontsize=9)
    fig.text(.09, .22, rf'Decoded JNS sequencing ($x={index}$):', fontsize=10)
    chips = fig.add_axes([.09,.10,.86,.09])
    chips.set_xlim(0,8)
    chips.set_ylim(0,1)
    chips.axis('off')
    for i, job in enumerate(sequence):
        chips.add_patch(Rectangle((i+.04,.07),.92,.84,facecolor=colors[job],
                                   edgecolor='#555555',hatch=hatches[job],linewidth=.6))
        chips.text(i+.5,.49,rf'$J_{job}$',ha='center',va='center',fontsize=11,color=textcolors[job])
    fig.text(.5,.025,'Different JNS sequences can decode to the same physical schedule.',
             fontsize=9,ha='center')
    caption = (f'Optimal schedule for g03 returned by the corrected Algorithm 2 run on the Spark '
               f'(JNS index x={index}). Blocks show operations O_k of job J_j on each machine. '
               'Operations on a machine do not overlap; the dashed arrows illustrate the '
               'within-job precedence of J0. The last operation finishes at makespan 11, '
               'which was independently established as optimal by exhaustive enumeration. '
               'The lower strip shows the decoded JNS job sequence; successive appearances '
               'of a job select its successive operations. Multiple JNS sequences may encode '
               'the same physical schedule.')
    save(fig,'figure_01_g03_schedule','A concrete feasible and optimal schedule',caption,
         'Does the schedule make the constraints and the relationship to the encoding clear?',350)


def workflow():
    fig = plt.figure(figsize=(8.0, 4.9))
    fig.suptitle('Threshold optimization and one BBHT search round', fontsize=13, y=.975)
    fig.text(.035,.90,'(a) Classical control and experimental oracle construction',fontsize=11)
    ax = fig.add_axes([.025,.585,.95,.275])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')
    centers = [.125,.375,.625,.875]
    labels = ['Choose threshold $T$\n(binary search)',
              'Construct explicit $O_T$\nby classical enumeration\n(repeated for trial circuits)',
              'BBHT feasibility search\n(repeated rounds)',
              'Update threshold bounds\nfrom the search outcome']
    for i,(center,label) in enumerate(zip(centers,labels)):
        ax.add_patch(Rectangle((center-.112,.32),.224,.58,facecolor='#ededed' if i !=2 else 'white',
                               edgecolor='black',linewidth=1.0))
        ax.text(center,.61,label,ha='center',va='center',fontsize=9,linespacing=1.5)
        if i <3:
            ax.annotate('',xy=(centers[i+1]-.113,.61),xytext=(center+.113,.61),
                        arrowprops=dict(arrowstyle='->',lw=1.0))
    ax.plot([.875,.875,.125],[.32,.10,.10],color='black',lw=.85)
    ax.annotate('',xy=(.125,.32),xytext=(.125,.10),arrowprops=dict(arrowstyle='->',lw=.85))
    ax.text(.5,.14,'Repeat until the threshold bounds meet',ha='center',va='bottom',fontsize=8.5,
            bbox=dict(facecolor='white',edgecolor='none',pad=1))
    fig.text(.035,.515,'(b) One round inside the BBHT search',fontsize=11)
    circuit = fig.add_axes([.025,.225,.95,.235])
    circuit.set_xlim(0,14)
    circuit.set_ylim(-.6,.7)
    circuit.axis('off')
    circuit.text(.02,0,r'$|0\rangle^{\otimes n}$',ha='left',va='center',fontsize=12)
    circuit.plot([1.35,8.0],[0,0],color='black',lw=1.0,zorder=1)
    circuit.plot([1.55,1.69],[-.075,.075],color='black',lw=.8)
    circuit.text(1.62,.20,'$n$',ha='center',fontsize=9)
    def gate(center,label,width=1.0):
        circuit.add_patch(Rectangle((center-width/2,-.22),width,.44,facecolor='white',
                                    edgecolor='black',lw=1.0,zorder=3))
        circuit.text(center,0,label,ha='center',va='center',fontsize=12,zorder=4)
    gate(2.55,r'$H^{\otimes n}$',1.25)
    gate(4.4,r'$O_T$')
    gate(5.85,r'$D$')
    circuit.add_patch(Rectangle((3.65,-.41),2.95,.91,fill=False,edgecolor='#666666',ls='--',lw=.7))
    circuit.text(5.125,.56,r'$j$ Grover iterations',ha='center',fontsize=9)
    circuit.add_patch(Rectangle((7.15,-.25),.95,.50,facecolor='white',edgecolor='black',lw=1,zorder=3))
    circuit.add_patch(Arc((7.625,-.05),.62,.36,theta1=0,theta2=180,lw=1,zorder=4))
    circuit.plot([7.625,7.82],[-.05,.17],color='black',lw=1,zorder=4)
    for yy in [-.025,.025]:
        circuit.plot([8.1,9.2],[yy,yy],color='black',lw=.75)
    circuit.annotate('',xy=(9.3,0),xytext=(9.05,0),arrowprops=dict(arrowstyle='->',lw=.8))
    circuit.text(8.65,.20,'$x$',ha='center',fontsize=10)
    circuit.add_patch(Rectangle((9.35,-.30),4.35,.60,facecolor='#ededed',edgecolor='black',lw=1))
    circuit.text(11.525,0,'Classically decode $x$; verify\nconstraints and makespan',
                 ha='center',va='center',fontsize=9.5,linespacing=1.5)
    fig.text(.5,.145,r'$O_T|x\rangle=(-1)^{\Phi_T(x)}|x\rangle$, '
             r'$D=2|s\rangle\langle s|-I$, $|s\rangle=H^{\otimes n}|0\rangle^{\otimes n}$',
             ha='center',fontsize=10)
    fig.text(.5,.073,'Verified hit: return a feasible index. Otherwise resample $j$ until the round budget is exhausted.',
             ha='center',fontsize=9)
    fig.text(.5,.028,'Oracle construction is classical; the implementation rebuilds it for trial circuits with nonzero j.',
             ha='center',fontsize=8.5)
    caption = ('Algorithm structure used in the experiments. (a) Classical binary search chooses '
               'a makespan threshold T. The experimental implementation enumerates valid JNS '
               'indices to construct an explicit phase oracle; this is classical preprocessing. '
               'The implementation rebuilds that oracle for trial circuits with nonzero j. '
               '(b) A BBHT round prepares the uniform search state, applies j oracle/diffusion '
               'iterations, measures an index and classically verifies the resulting schedule. '
               'The BBHT controller samples j and repeats until success or its round budget is '
               'exhausted. Budget exhaustion is a bounded-search outcome, not by itself a proof '
               'of infeasibility. Each quantum wire represents the full n-qubit register. '
               'This schematic does not imply an end-to-end runtime advantage over classical solvers.')
    save(fig,'figure_02_algorithm_schematic','What is classical and what is quantum',caption,
         'Is the explicit classical oracle preparation prominent enough, and is the feedback loop clear?',375)


def amplification():
    source = OUT/'figure_03_probability_data.csv'
    inst = utils.load_benchmark_file(str(BENCHMARKS/'k032.txt'))[0]
    reference, costs = audit.brute_force(inst, utils)
    assert (reference['n'], reference['C_star'], reference['S']) == (4, 3, 5)
    with source.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['basis_index','category','p_initial','p_final_round'])
        for x in range(16):
            marked = x < len(costs) and costs[x] <= 3
            kind = 'padding' if x >= len(costs) else ('feasible' if marked else 'valid_infeasible')
            writer.writerow([x, kind, 1/16, 49/256 if marked else 1/256])
    rows = list(csv.DictReader(source.open(encoding='utf-8')))
    assert len(rows) == 16
    x = np.array([int(r['basis_index']) for r in rows])
    initial = np.array([float(r['p_initial']) for r in rows])
    final = np.array([float(r['p_final_round']) for r in rows])
    feasible = np.array([r['category'] == 'feasible' for r in rows])
    assert feasible.sum() == 5 and sum(r['category'] == 'padding' for r in rows) == 4
    assert np.allclose(initial, 1/16)
    assert np.allclose(final[feasible], 49/256) and np.allclose(final[~feasible], 1/256)
    assert np.isclose(initial.sum(),1) and np.isclose(final.sum(),1)
    fig, axes = plt.subplots(1,2,figsize=(7.4,3.9),sharey=True)
    fig.subplots_adjust(left=.095,right=.985,bottom=.31,top=.74,wspace=.13)
    fig.suptitle('k032: amplification of five feasible sequencings',fontsize=13,y=.975)
    fig.text(.5,.865,r'$T=C^*_{\max}=3$, $n=4$, $N=16$, $N_{\mathrm{valid}}=12$, $S=5$',
             ha='center',fontsize=10)
    colors = ['#303030' if f else ('#d0d0d0' if i>=12 else '#a0a0a0')
              for i,f in enumerate(feasible)]
    for panel,(ax,p,title) in enumerate(zip(axes,[initial,final],
        [r'(a) Initial uniform state ($j=0$)',r'(b) One Grover iteration ($j=1$)'])):
        ax.axvspan(11.5,15.5,color='#f2f2f2',zorder=0)
        bars=ax.bar(x,p,width=.76,color=colors,edgecolor='#666666',linewidth=.45,zorder=3)
        for i,bar in enumerate(bars):
            if feasible[i]:
                bar.set_hatch('///')
                bar.set_edgecolor('white')
            elif i>=12:
                bar.set_hatch('..')
        ax.axvline(11.5,color='#666666',ls=':',lw=.8)
        ax.set_xlim(-.6,15.6)
        ax.set_ylim(0,.235)
        ax.set_xticks(range(16))
        ax.tick_params(axis='x',labelsize=8)
        ax.set_yticks([0,.05,.10,.15,.20])
        ax.set_title(title,fontsize=10,pad=10)
        ax.set_xlabel('Basis-state index $x$',labelpad=6,fontsize=9)
        ax.grid(axis='y',color='#dddddd',lw=.5,zorder=0)
        ax.set_axisbelow(True)
        cx=.29 if panel==0 else .76
        label = r'$P(\mathrm{feasible})=5/16=31.25\%$' if panel==0 else r'$P(\mathrm{feasible})=245/256=95.70\%$'
        fig.text(cx,.16,label,ha='center',fontsize=9)
        fig.text(cx,.105,r'$\sum_{x=0}^{15}p(x)=1$; '+
                 ('padding mass = 25.00%' if panel==0 else 'padding mass = 1.5625%'),
                 ha='center',fontsize=8.5)
    axes[0].set_ylabel('Exact ideal measurement probability',fontsize=9,labelpad=8)
    fig.legend(handles=[Patch(facecolor='#303030',hatch='///',edgecolor='white',label='Feasible (5)'),
                        Patch(facecolor='#a0a0a0',edgecolor='#666666',label='Valid, infeasible (7)'),
                        Patch(facecolor='#d0d0d0',hatch='..',edgecolor='#666666',label='Padding (4)')],
               loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=8.5)
    caption = ('Exact ideal measurement probabilities for k032 at the independently verified '
               'optimal threshold T=3. All 16 basis states are shown: five feasible JNS '
               'sequencings, seven valid infeasible sequencings and four padded states. '
               'Initially each state has probability 1/16. After one ideal Grover iteration, '
               'each feasible state has probability '
               '49/256 and each unmarked state has probability 1/256. The total feasible '
               'mass therefore rises from 5/16 to 245/256. Each panel sums to one. These '
               'are unconditional premeasurement probabilities, not shot counts or a '
               'distribution conditioned on a successful outcome.')
    save(fig,'figure_03_k032_amplification','A checkable demonstration of amplification',caption,
         'This replaces the th02 amplification example: for th02, S/N=1/2 and probabilities remain uniform.',300)
    (OUT/'figure_03_probability_data.csv').write_bytes(source.read_bytes())


def query_cost():
    names = ['th02','k032','g03','v03','yn03','p034']
    fig, axes = plt.subplots(2,3,figsize=(8.1,6.1),sharey=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.785,wspace=.22,hspace=.44)
    fig.suptitle('Where Algorithm 2 spends its oracle calls',fontsize=13,y=.98)
    fig.legend(handles=[Patch(facecolor='#a0a0a0',edgecolor='#555555',label='Returned a feasible schedule'),
                        Patch(facecolor='#303030',edgecolor='white',hatch='///',label='Round budget exhausted')],
               loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=9)
    output_rows=[]
    total, exhausted=0,0
    for ax,name in zip(axes.flat,names):
        record=records[name]
        reference=record['reference']
        trace=record['algorithm2']['threshold_trace']
        assert sum(t['calls'] for t in trace)==record['algorithm2']['calls']
        assert sum(t['rounds'] for t in trace)==record['algorithm2']['rounds']
        for order,t in enumerate(trace):
            is_exhausted=t['reason']=='budget_exhausted'
            if is_exhausted:
                assert t['T']<reference['C_star'] and t['x'] is None
                exhausted+=t['calls']
            else:
                assert t['x'] is not None
            total+=t['calls']
            bar=ax.bar(order,t['calls'],width=.68,
                       color='#303030' if is_exhausted else '#a0a0a0',
                       edgecolor='white' if is_exhausted else '#555555',
                       linewidth=.6,hatch='///' if is_exhausted else None,zorder=3)
            if t['calls']>=1000:
                ax.text(order,math.sqrt(t['calls']),f"{t['calls']:,}",ha='center',va='center',
                        rotation=90,fontsize=8,color='white',zorder=5,
                        bbox=dict(facecolor='#303030',edgecolor='none',pad=1))
            else:
                if t['calls']==0:
                    ax.plot(order,0,'o',markersize=3,color='#555555',zorder=5)
                ax.annotate(str(t['calls']),xy=(order,t['calls']),xytext=(0,4),
                            textcoords='offset points',ha='center',va='bottom',fontsize=9,zorder=5)
            output_rows.append(dict(instance=name,test=order+1,T=t['T'],calls=t['calls'],
                                    rounds=t['rounds'],outcome=t['reason'],
                                    classical_optimum=reference['C_star'],
                                    at_optimum=t['T']==reference['C_star']))
        ax.set_title(f'{name}\nTotal = {record["algorithm2"]["calls"]:,}',fontsize=10,pad=8)
        ax.set_yscale('symlog',linthresh=1,linscale=.8)
        ax.set_ylim(-.08,1e7)
        ax.set_yticks([0,1,10,1000,100000,10000000],['0','1','10',r'$10^3$',r'$10^5$',r'$10^7$'])
        ax.set_xticks(range(len(trace)),[str(t['T'])+('*' if t['T']==reference['C_star'] else '') for t in trace])
        ax.set_xlim(-.65,len(trace)-.35)
        ax.grid(axis='y',color='#dddddd',lw=.5,zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both',labelsize=8.5)
    axes[0,0].set_ylabel('Oracle calls',fontsize=9)
    axes[1,0].set_ylabel('Oracle calls',fontsize=9)
    fig.text(.54,.145,'Threshold $T$ in the order tested; * marks the independently verified optimum.',
             ha='center',fontsize=9)
    fig.text(.5,.089,'Common symmetric-log y scale, linear near zero. Dots mark zero-call successes.',
             ha='center',fontsize=8.5)
    fig.text(.5,.039,'Zero-call successes still perform state preparation, measurement and classical verification.',
             ha='center',fontsize=8.5)
    with (OUT/'figure_04_threshold_data.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    caption = (f'Algorithm 2 oracle calls at every tested makespan threshold in the '
               f'corrected run (seed {run["manifest"]["seed"]}). Thresholds appear in their execution order; '
               f'an asterisk marks the independently verified optimum. The common '
               f'symmetric-log scale is linear near zero, and zero-call outcomes are '
               f'marked by dots. Dark hatched bars indicate BBHT searches that exhausted '
               f'their round budget; independent classical enumeration establishes that '
               f'these particular thresholds are infeasible. They account for {exhausted:,} '
               f'of {total:,} total oracle calls. Budget exhaustion alone is not a proof '
               f'of infeasibility. Counts are abstract oracle queries and exclude classical '
               f'oracle construction, verification and compilation; they do not determine QPU time.')
    save(fig,'figure_04_threshold_query_cost','Why the optimization query counts are large',caption,
         'Does the common scale and threshold-level breakdown explain the resource table without implying a hardware speedup?',400)


gantt()
workflow()
amplification()
query_cost()

# LaTeX inserts use the supplied PDF filenames, with ordinary automatic numbering.
tex_captions = [
    r'''Optimal g03 schedule returned by corrected Algorithm~2 ($x=MEASUREDINDEX$), with
    makespan $C_{\max}=11$, independently proved optimal by exhaustive enumeration.
    Blocks identify job and operation; dashed arrows illustrate job $J_0$ precedence.
    The lower strip gives the decoded JNS sequence. Distinct JNS sequences can
    produce the same physical schedule.''',
    r'''Structure of the experimental algorithm. Classical binary search chooses $T$;
    the explicit oracle is constructed by classical enumeration of JNS indices
    and rebuilt for trial circuits with nonzero $j$.
    A BBHT round prepares the uniform state, applies $j$ oracle/diffusion iterations,
    measures an index, and verifies the decoded schedule classically. BBHT resamples
    $j$ until success or round-budget exhaustion. Each wire represents the full
    $n$-qubit register. Budget exhaustion alone does not prove infeasibility;
    classical preprocessing must be counted in total execution cost.''',
    r'''Exact ideal measurement probabilities for k032 at $T=C^*_{\max}=3$.
    All $N=16$ states are shown: five feasible JNS sequencings, seven valid
    infeasible sequencings, and four padded states. Initially each state has
    probability $1/16$. After one ideal Grover iteration ($j=1$),
    each feasible state has probability $49/256$ and each unmarked state has
    probability $1/256$. Thus feasible mass rises from $5/16$ to $245/256$.
    Each panel sums to one. These are unconditional premeasurement probabilities,
    not shot counts or probabilities conditioned on successful readout.''',
    r'''Algorithm~2 oracle calls by threshold in the corrected Spark run (seed 42).
    Thresholds appear in execution order; an asterisk marks the independently
    verified optimum. The common symmetric-log scale is linear near zero;
    dots mark zero-call successes. Dark hatched bars show searches that exhausted
    their round budgets at thresholds independently established as infeasible.
    They account for $5{,}902{,}107$ of $5{,}902{,}120$ total calls.
    Budget exhaustion alone is not an infeasibility proof. Oracle counts exclude
    classical preparation, verification and compilation, and do not determine QPU time.'''
]
labels=['fig:toy-schedule','fig:algorithm-structure','fig:amplification','fig:threshold-effort']
inserts=[]
for entry,caption,label in zip(figures,tex_captions,labels):
    caption = caption.replace('MEASUREDINDEX', str(records['g03']['algorithm2']['schedule_info']['x']))
    total_calls = sum(r['algorithm2']['calls'] for r in records.values())
    exhausted_calls = sum(t['calls'] for r in records.values()
                          for t in r['algorithm2']['threshold_trace'] if t['reason'] == 'budget_exhausted')
    caption = caption.replace('5{,}902{,}107', f'{exhausted_calls:,}'.replace(',', '{,}'))
    caption = caption.replace('5{,}902{,}120', f'{total_calls:,}'.replace(',', '{,}'))
    caption = caption.replace('Spark run (seed 42)', f'run (seed {run["manifest"]["seed"]})')
    environment = 'figure' if entry['stem'].startswith('figure_02') else 'figure*'
    width = r'\columnwidth' if environment == 'figure' else r'\textwidth'
    inserts.append(r'\begin{'+environment+'}[t]'+'\n'+r'\centering'+'\n'
                   +rf'\includegraphics[width={width}]{{{entry["stem"]}.pdf}}'+'\n'
                   +r'\caption{'+textwrap.dedent(caption).strip()+'}\n'
                   +rf'\label{{{label}}}'+'\n'+r'\end{'+environment+'}'+'\n')
(OUT/'paper_figure_captions.tex').write_text('% Preamble: \\usepackage{graphicx}\n\n'+'\n'.join(inserts),encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps(dict(
    source_manifest=run['manifest'],source_records_sha256=hashlib.sha256(
        (RECORDS/'results.json').read_bytes()).hexdigest(),
    figures=figures),indent=2),encoding='utf-8')
(OUT/'README.txt').write_text('''FOUR PROPOSED MAIN-TEXT FIGURES - REVIEW DRAFT

1. Actual g03 schedule from the verified Spark run, with its JNS job sequence.
2. Algorithm and circuit schematic distinguishing classical oracle construction.
3. k032 amplification with all padded states and exact normalization.
4. Query effort by binary-search threshold, from the actual run records.

Each figure is supplied as a vector PDF and an editable-text SVG.
paper_figure_captions.tex contains proposed two-column LaTeX figure inserts.
The probability figure is exact ideal circuit data, not new measurement counts.
The resource figure displays abstract oracle queries, not hardware gate counts.
CSV files and provenance.json provide the numerical sources.
''',encoding='utf-8')

print('Created four PDF/SVG figures, CSVs, captions and provenance.', flush=True)
