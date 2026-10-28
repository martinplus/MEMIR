"""延迟三段分解：定位 n 较大时 memir 反劣于 static 的那一段。

service_time = L1(release->assign) + L2(assign->pickup) + L3(pickup->delivery)
memir 的 resp_dist 更小只保证 L2 更短；若 L1 或 L3 变大，总延迟仍可能更差。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maps import get_map                     # noqa: E402
from sim import MAPDSim, SimConfig           # noqa: E402

BASE = dict(T=4000, warmup=1000, zipf_alpha=2.0, arrival_mode="open",
            lmbda=0.08, capacity=1, rest_set="any", drift_mode="static",
            drift_param=2000, planner_mode="tp", assign_mode="hungarian",
            map_name="W1", dwell_safety=True, dwell_probe=8, preempt=0,
            tp_dummy=False, reeval_interval=40, move_thresh=0.0)

POL = {
    "nearest": dict(rest_policy="nearest", demand_mode="none"),
    "random": dict(rest_policy="random", demand_mode="none"),
    "static": dict(rest_policy="static", demand_mode="none"),
    "memir": dict(rest_policy="exp", demand_mode="ema", tau_phi3=100,
                  rest_beta=0.5),
}


def run(j):
    j = dict(j)
    m = j.pop("map_name", "W1")
    pol = j.pop("__pol", "")
    n = j.pop("__n", 0)
    seed = j.get("seed", 0)
    wh = get_map(m)
    try:
        r = MAPDSim(wh, SimConfig(**j)).run()
    except Exception as e:                       # noqa: BLE001
        return dict(pol=pol, n=n, seed=seed, err=repr(e))
    return dict(pol=pol, n=n, seed=seed, err="",
                svc=r["service_time"], l1=r["leg_assign"], l2=r["leg_approach"],
                l3=r["leg_haul"], resp=r["resp_dist"], mv=r["rest_moves"],
                noidle=r["no_idle"], det=r["detour"], pend=r["mean_pending"],
                idle=r["idle_ratio"])


def main():
    ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else [8, 16, 24, 32, 40]
    seeds = 6
    jobs = []
    for n in ns:
        for pol, pk in POL.items():
            for s in range(seeds):
                j = dict(seed=s, n_agents=n, map_name="W1")
                j.update({k: v for k, v in BASE.items() if k != "map_name"})
                j.update(pk)
                j.update(__pol=pol, __n=n)
                jobs.append(j)
    print("jobs=%d" % len(jobs), flush=True)
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 40 == 0:
                print("  [%4d/%4d %5.0fs]" % (i, len(jobs), time.time() - t0),
                      flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        if r["err"]:
            print("ERR", r)
            continue
        agg[(r["n"], r["pol"])].append(r)
    pols = list(POL)
    hdr = "%-4s %-8s | %7s %7s %7s %7s %7s | %7s %7s %7s" % (
        "n", "pol", "svc", "L1_wait", "L2_appr", "L3_haul", "resp",
        "moves", "noidle", "detour")
    print("\n" + hdr)
    print("-" * len(hdr))
    for n in ns:
        for pol in pols:
            g = agg.get((n, pol))
            if not g:
                continue
            f = lambda k: sum(x[k] for x in g) / len(g)   # noqa: E731
            print("%-4d %-8s | %7.2f %7.2f %7.2f %7.2f %7.2f | %7.0f %7.0f %7.3f"
                  % (n, pol, f("svc"), f("l1"), f("l2"), f("l3"), f("resp"),
                     f("mv"), f("noidle"), f("det")))
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
