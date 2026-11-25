# Third-party sources

The MIT license covers project-owned code. It does not relicense dependencies,
benchmark descriptions, third-party schedules, or any existing third-party files
in the historical exploratory directories. Preserve any original notices.

The toy inputs are small factual benchmark operation tables. Their labels and
provenance are retained in `benchmarks-toys/`. Relevant sources include:

- g03/go03: [Google OR-Tools job shop example](https://developers.google.com/optimization/scheduling/job_shop).
- k032: Kurowski et al. (2023), *Application of Quantum Approximate Optimization
  Algorithm to Job Shop Scheduling Problem*, EJOR 310, 518–528,
  https://doi.org/10.1016/j.ejor.2023.03.013.
- v03: Venturelli et al. (2016), as identified in the paper and input comments.
- yn03 and p034: Yamada–Nakano and Pinedo toy instances as identified in the
  paper and input comments. Instance labels identify the particular toy inputs;
  they do not imply full-size benchmark suites.
- FT06 and historical OR-Library data:
  [OR-Library job shop data](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html).

Qiskit, Qiskit Aer, NumPy, Matplotlib and pylatexenc are installed dependencies;
their upstream licenses apply. They are not vendored into this release.
