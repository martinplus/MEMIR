#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the tables of the MEMIR manuscript from the raw result files.

Pure standard-library Python: no numpy, no scipy, no network.  Run

    python reproduce.py            # check every table against expected.json
    python reproduce.py --print    # additionally print the recomputed tables

The script does two things.

1. It recomputes, from ``results_final/raw_*.csv`` alone, the numbers that
   appear in the manuscript, and compares them with the values recorded in
   ``expected.json`` (which were typed from the manuscript).  A check succeeds
   when the absolute difference does not exceed the tolerance implied by the
   number of decimals printed in the paper (0.05 for one decimal, 0.005 for
   two, 0.05 for values printed as integers of a rate).
2. It re-runs the additivity check of Lemma 1 (three-leg decomposition) over
   every run of the main sweep.

Conventions used throughout, matching the manuscript:

* All metrics are computed after the warm-up prefix (``service_post``).
* A reported difference is the ratio of means under common random numbers,
  ``mean(service_post of M) / mean(service_post of static) - 1``, expressed in
  percent.  Seeds are matched between the two policies cell by cell.
* The leg identity is ``T1 = leg_assign`` (release -> assignment),
  ``T2 = leg_approach`` (assignment -> pickup), ``T3 = leg_haul``
  (pickup -> delivery).

Re-running the *simulations* rather than the tables is a separate step and takes
far longer; ``README.md`` gives the commands.
"""
from __future__ import print_function

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_final")
EXPECT = os.path.join(HERE, "expected.json")

OK, BAD = [], []


def rows_of(name):
    with open(os.path.join(RAW, name), encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if not r.get("error")]


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else float("nan")


def by(rows, *keys):
    out = {}
    for r in rows:
        out[tuple(r[k] for k in keys)] = out.get(tuple(r[k] for k in keys),
                                                 []) + [r]
    return out


def mpost(rs):
    return mean([float(r["service_post"]) for r in rs])


def delta(rs_m, rs_s):
    """percentage difference, ratio of means under common random numbers"""
    return (mpost(rs_m) / mpost(rs_s) - 1.0) * 100.0


def check(tag, got, want, tol):
    ok = abs(got - want) <= tol
    (OK if ok else BAD).append(tag)
    print("  [%s] %-34s got %9.3f  paper %9.3f  tol %.3f"
          % ("ok" if ok else "XX", tag, got, want, tol))
    return ok


# ------------------------------------------------------------------ Lemma 1
def check_additivity():
    print("\nLemma 1 (three-leg additivity), all runs of the main sweep")
    rows = rows_of("raw_A_main.csv")
    res = max(abs(float(r["leg_assign"]) + float(r["leg_approach"])
                  + float(r["leg_haul"]) - float(r["service_post"]))
              for r in rows)
    n = len(rows)
    print("  runs = %d, largest |T1+T2+T3 - service| = %.3g timesteps"
          % (n, res))
    check("additivity residual", res, 0.01, 0.02)
    check("main sweep runs", float(n), 810.0, 0.5)


# ------------------------------------------------------------------- Table 7
MAIN_FLEET = [4, 6, 8, 10, 12, 16, 24, 32, 40]
MAIN_POL = ["nearest", "random", "memir0", "memir", "oracle"]


def table_main():
    print("\nMain comparison (reference layout W1), ten seeds per cell")
    rows = rows_of("raw_A_main.csv")
    d = by(rows, "n_agents", "method")
    idle = {}
    out = {}
    for n in MAIN_FLEET:
        key = (str(n), "memir")
        idle[n] = mean([float(r["idle_ratio"]) for r in d[key]]) * 100.0
        for p in MAIN_POL:
            out[(n, p)] = delta(d[(str(n), p)], d[(str(n), "static")])
    return idle, out


def check_main(idle, out, exp):
    for n in MAIN_FLEET:
        check("idle %d" % n, idle[n], float(exp["idle"][str(n)]), 0.05)
        for p in MAIN_POL:
            check("|A|=%d %s" % (n, p), out[(n, p)],
                  float(exp["main"][str(n)][p]), 0.05)


# ------------------------------------------------------------------ Table 10
def table_legs():
    print("\nThree-leg decomposition at three fleet sizes")
    rows = rows_of("raw_A_main.csv")
    d = by(rows, "n_agents", "method")
    out = {}
    for n in (8, 16, 32):
        for p in ("nearest", "static", "memir0", "memir"):
            rs = d[(str(n), p)]
            out[(n, p)] = tuple(mean([float(r[k]) for r in rs])
                                for k in ("leg_assign", "leg_approach",
                                          "leg_haul", "service_post"))
    return out


def check_legs(out, exp):
    for n in (8, 16, 32):
        for p in ("nearest", "static", "memir0", "memir"):
            vals = out[(n, p)]
            for i, leg in enumerate(("T1", "T2", "T3", "service")):
                check("legs %d %s %s" % (n, p, leg), vals[i],
                      float(exp["legs"][str(n)][p][leg]), 0.005)


# ------------------------------------------------------------------ Table 20
FREE = {"W1": 711, "W2": 1230, "W3": 213, "W4": 2551, "W5": 5511,
        "W6": 9591, "B1": 5699, "B2": 9776, "B3": 22599}


def table_scale():
    print("\nScale study, all twenty-four cells")
    rows = rows_of("raw_O_scale.csv")
    d = by(rows, "map", "n_agents", "lmbda", "method")
    order = {}
    for k in d:
        if k[3] == "static":
            order[k[:3]] = None
    out = {}
    for (mp, n, lam) in sorted(order, key=lambda z: (z[0], int(z[1]),
                                                     float(z[2]))):
        S = d[(mp, n, lam, "static")]
        M = d[(mp, n, lam, "memir")]
        g = delta(M, S)
        idle = mean([float(r["idle_ratio"]) for r in M]) * 100.0
        t2 = (mean([float(r["leg_approach"]) for r in S]),
              mean([float(r["leg_approach"]) for r in M]))
        paf = (mean([float(r["plan_attempt_fail"]) for r in S]),
               mean([float(r["plan_attempt_fail"]) for r in M]))
        out["%s_%s_%s" % (mp, n, lam)] = dict(
            gain=g, idle=idle, t2=t2, paf=paf,
            density=int(n) / FREE[mp] * 100.0)
    return out


def check_scale(out, exp):
    for key, want in sorted(exp["scale"].items()):
        got = out[key]
        check("scale %s gain" % key, got["gain"], float(want["gain"]), 0.05)
        check("scale %s idle" % key, got["idle"], float(want["idle"]), 0.05)


# ------------------------------------------------------------------- Table 4
def table_guard():
    print("\nTER-Safety fallbacks, per 10000 rest decisions")
    rows = rows_of("raw_Q_guard2.csv")
    d = by(rows, "n_agents", "method")
    out = {}
    for n in (16, 24, 32, 40):
        S = d[(str(n), "static")]
        MI = d[(str(n), "min_p8")]
        FR = d[(str(n), "free_p8")]
        PA = d[(str(n), "park_p8")]

        def rate(rs):
            return (mean([float(r["dwell_blocked"]) for r in rs])
                    / mean([float(r["decide_cnt"]) for r in rs])) * 1e4
        out[n] = dict(min=rate(MI), park=rate(PA), free=rate(FR),
                      vs_static=delta(FR, S), vs_min=delta(FR, MI),
                      fb_fail=mean([float(r["dwell_fb_fail"]) for r in FR]))
    return out


def check_guard(out, exp):
    for n in (16, 24, 32, 40):
        w = exp["guard"][str(n)]
        for k in ("min", "park", "free", "vs_static", "vs_min", "fb_fail"):
            check("guard %d %s" % (n, k), out[n][k], float(w[k]),
                  0.005 if k in ("vs_static", "vs_min") else 0.05)


# ------------------------------------------------------- planner variants
def table_planner():
    print("\nToken Passing with and without the dummy suffix")
    rows = rows_of("raw_C_planner.csv")
    d = by(rows, "method", "n_agents")
    out = {}
    for mode in ("tp", "tp_dummy"):
        for n in (8, 16, 32):
            out["%s_%d" % (mode, n)] = delta(
                d[(mode + "+memir", str(n))], d[(mode + "+static", str(n))])
    return out


def check_planner(out, exp):
    for k, w in sorted(exp["planner"].items()):
        check("planner %s" % k, out[k], float(w), 0.05)


# --------------------------------------------------------------- run totals
def run_counts():
    tot = 0
    files = 0
    for name in sorted(os.listdir(RAW)):
        if not (name.startswith("raw_") and name.endswith(".csv")):
            continue
        if name in ("raw_smoke.csv",) or name.endswith(".bak"):
            continue
        n = len(rows_of(name))
        tot += n
        files += 1
    print("\nRaw result files")
    print("  %d files, %d runs" % (files, tot))
    return files, tot


def main():
    with open(EXPECT, encoding="utf-8") as fh:
        exp = json.load(fh)
    show = "--print" in sys.argv

    check_additivity()
    idle, mout = table_main()
    check_main(idle, mout, exp)
    legs = table_legs()
    check_legs(legs, exp)
    scale = table_scale()
    check_scale(scale, exp)
    guard = table_guard()
    check_guard(guard, exp)
    plan = table_planner()
    check_planner(plan, exp)
    files, tot = run_counts()
    check("total runs", float(tot), float(exp["total_runs"]), 0.5)

    if show:
        print("\n--- main comparison, percent vs static ---")
        for n in MAIN_FLEET:
            print("  |A|=%-3d idle=%5.1f%%  " % (n, idle[n])
                  + "  ".join("%s %+6.2f" % (p, mout[(n, p)])
                              for p in MAIN_POL))
        print("\n--- scale study ---")
        for k in sorted(scale):
            v = scale[k]
            print("  %-14s density %5.2f%%  idle %5.1f%%  gain %+6.2f  "
                  "T2 %.1f->%.1f  PAF %.0f->%.0f"
                  % (k, v["density"], v["idle"], v["gain"], v["t2"][0],
                     v["t2"][1], v["paf"][0], v["paf"][1]))

    print("\n%d checks passed, %d failed" % (len(OK), len(BAD)))
    if BAD:
        for b in BAD:
            print("  FAILED:", b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
