# Completed corrected-readout reference run

These are the actual six-instance Spark records supplied by the experiment,
not a new execution performed during repository preparation. Only the local
hostname in the manifest was replaced with a generic hardware description.
Numerical measurements, seeds, individual trials, binary-search traces and
source/input fingerprints are unchanged. Timings are host-dependent.

The original runner repaired two reversed-readout sites in a temporary solver
copy. The public core now contains that correction and the public runner imports
it directly, so its source hashes and manifest differ from this historical run.
Do not use this folder as a destination for a new run or attempt to resume it
with changed source. Choose a new output directory.

The six independently established optima are 4, 3, 11, 6, 12 and 28.
All 600 BBHT trials succeeded and all returned schedules were independently
verified. The search register count excludes working qubits that a scalable
reversible schedule evaluator would require.
