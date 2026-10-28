"""Reviewer 2 新增机制的冒烟测试。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maps import get_map                        # noqa: E402
from rest import build_haul_load                # noqa: E402
from run_suite import BASE, POL, _mk            # noqa: E402
from sim import MAPDSim, SimConfig              # noqa: E402

wh = get_map("W1")
print("map W1: free=%d parking=%d pickup=%d station=%d"
      % (len(wh.free_cells), len(wh.parking_cells), len(wh.pickup_cells),
         len(wh.station_cells)))

cells = [c for c in wh.free_cells
         if c[1] < wh.station_top and c not in set(wh.pickup_cells)]
print("cand cells (rest_set=any):", len(cells))
pk_in = sum(1 for c in wh.parking_cells if c in set(cells))
print("parking cells inside candidate set:", pk_in, "/", len(wh.parking_cells))

# ---- flow 的四种定义：两两相关系数 ----
import math                                     # noqa: E402
fields = {}
for m, sd in (("single", 0), ("rand", 1), ("rand", 2), ("any", 0),
              ("multi", 0)):
    t0 = time.time()
    f = build_haul_load(wh, cells, mode=m, seed=sd)
    fields[(m, sd)] = f
    nz = sum(1 for v in f.values() if v > 0)
    print("flow %-7s seed=%d  nonzero=%5d  offload=%.1fs"
          % (m, sd, nz, time.time() - t0))

keys = list(fields)


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


print("\npairwise Pearson between flow definitions:")
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a = [fields[keys[i]][c] for c in cells]
        b = [fields[keys[j]][c] for c in cells]
        print("   %-12s vs %-12s  r=%.4f"
              % (str(keys[i]), str(keys[j]), pearson(a, b)))

# ---- 仿真冒烟 ----
def one(tag, n=16, seed=0, T=800, warmup=200, **kw):
    j = dict(BASE)
    j.update(POL["memir"])
    j.update(kw)
    j.update(T=T, warmup=warmup, map_name="W1", n_agents=n, seed=seed)
    cfg = SimConfig(**j)
    r = MAPDSim(get_map("W1"), cfg).run()
    print("%-22s svc=%.2f viol=%d chk=%d fb=%d fbfail=%d blk=%d "
          "planfail=%d attfail=%d conf=%d"
          % (tag, r["service_post"], r["dwell_viol"], r["dwell_checks"],
             r["dwell_fb"], r["dwell_fb_fail"], r["dwell_blocked"],
             r["plan_fail"], r["plan_attempt_fail"], r["conflicts"]))
    return r


print("\n--- Q_guard ---")
one("min_p8", dwell_fallback="min", dwell_probe=8)
one("park_p2", dwell_fallback="park", dwell_probe=2)
one("park_p8", dwell_fallback="park", dwell_probe=8)
one("park_p64", dwell_fallback="park", dwell_probe=64)
print("--- Q_flow ---")
one("flow single")
one("flow multi", flow_def="multi")
one("flow any", flow_def="any")
print("--- Q_stress ---")
one("lag200", comm_lag=200)
one("noise0.35", obs_noise=0.35)
one("dynobs2", dyn_obs=2)
one("dynobs6", dyn_obs=6)
one("het5", het_lat=5)
one("fb0.0", demand_feedback=0.0)
one("fb0.6", demand_feedback=0.6)
one("static fb0.0", rest_policy="static", demand_mode="none")
one("static fb0.6", demand_feedback=0.6, rest_policy="static", demand_mode="none")
