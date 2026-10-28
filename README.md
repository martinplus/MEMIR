# Artifact for “Ant-Colony-Inspired Dwell Memory for Traffic-Aware Idle Repositioning in Multi-Agent Pickup and Delivery”

This bundle contains everything needed to **check** every number reported in the
manuscript and, if desired, to **re-run** the simulations that produced them.

The implementation is pure standard-library Python 3: no third-party packages,
no compilation, no network access. Verified on CPython 3.12 and 3.13.

---

## 1. Contents

```
reproduce.py            driver: recomputes the reported tables from the raw files
expected.json           the values as printed in the manuscript
README.md               this file
CHECKSUMS.txt           sha256 of every file in the bundle
mapf_sim/               simulator, experiment drivers, analysis and figure scripts
  warehouse.py          KIVA-style map generator, region partition (Definition 5)
  movingai.py           MovingAI benchmark adapter (porting rules R1–R6)
  maps.py               the nine layouts used in the paper
  sim.py                the MAPD executor (Token Passing + the third layer)
  rest.py               resting policies, candidate set, haul-flow prior
  demand.py             arrival field, two-timescale filters, drift detector
  assign.py, seq.py     L1 assignment and batching
  planner.py            space–time A* and the task planner
  stats_util.py         bootstrap intervals, exact signed-rank test, Holm, MDE
  run_suite.py          experiment suites A–P (the original submission)
  run_r2.py             experiment suites Q_* (added during revision)
  analyze_suite.py      table generation
  plot_paper_en.py      figure generation (needs matplotlib, figures only)
results_final/
  raw_*.csv             25 files, 8 499 runs, one row per (suite, cell, seed)
```

## 2. Checking the reported numbers (about one minute)

```
python reproduce.py            # or:  python3 reproduce.py
python reproduce.py --print    # also prints the recomputed main and scale tables
```

The driver recomputes, from `results_final/raw_*.csv` alone,

* the main comparison (nine fleet sizes × five policies, idle fraction and
  percentage difference against `static`);
* the three-leg decomposition at eight, sixteen and thirty-two agents
  ($T_1=\texttt{leg\_assign}$, $T_2=\texttt{leg\_approach}$,
  $T_3=\texttt{leg\_haul}$; the columns sum to the service time exactly, which
  is how Lemma 1 was checked on every run);
* the complete scale study, all twenty-four cells;
* the TER-Safety fallback comparison, including the certified-fallback residual
  and the `dwell_fb_fail` counter;
* the comparison of Token Passing with and without the dummy suffix, which is
  what Section 4.7 quotes for the interaction with the planner;
* the total number of runs;

and compares each against `expected.json`. A check passes when the absolute
difference does not exceed the tolerance implied by the decimals printed in the
paper (0.05 for one decimal, 0.005 for two). The expected output ends with

```
183 checks passed, 0 failed
```

Conventions, matching the manuscript: all metrics are computed after the
warm-up prefix (`service_post`); a reported difference is the **ratio of means**
under common random numbers, expressed in percent, with seeds matched cell by
cell; the idle fraction is the realised fraction under `memir`.

## 3. Re-running the simulations (hours, not minutes)

The experiments are deterministic given a seed. From the repository root:

```
cd mapf_sim
python run_suite.py A_main          # one suite; omit the argument to run all
WORKERS=16 python run_suite.py      # parallel, 16 processes
python run_r2.py Q_guard2           # suites added during revision
```

`WORKERS` defaults to 10. Output is appended to `results_final/raw_<suite>.csv`.
One run costs from under a minute on the reference layout to about fifty minutes
on the largest benchmark maps, which is why the full set (8 499 runs) is a
multi-hour job. The figure scripts additionally require matplotlib and are not
needed to check any number:

```
python plot_paper_en.py             # writes paper/figs/*.pdf and *.svg
```

## 4. Notes for reviewers

* **Determinism.** Every run is seeded, and common random numbers are used
  throughout: the same seed gives the same task stream to every policy in a
  cell, which is what makes the paired tests of the paper valid.
* **No tuning on test data.** Only two quantities were chosen by search
  ($\psi$ and the memory timescale); see the manuscript for the sweep.
* **Errors are visible.** Rows whose run raised an exception carry a non-empty
  `error` field and are excluded from every aggregate. `reproduce.py` counts
  them; there are none in the shipped files.
