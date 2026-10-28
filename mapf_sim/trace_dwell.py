"""采集「空闲 agent 实际停留在哪些格子」，用于绘制 Fig.5 地图俯视图。

动机：三种泊位策略在相同任务流、相同地图上运行，唯一差别是"停在哪"。
    nearest / static   —— 均匀或就近，散在停车位与边缘
    memir0 (ψ=0)       —— 1-中位数解，向需求的加权中位聚集 → 恰好压在搬运干道上
    memir  (ψ=30)      —— 加入 flow(c) 先验后让开干道，但保留空间结构

产出 figs/fig5_dwell.json：
    { n, map, seeds, layout:{W,H,grid,station_top,park_rows},
      flow: [[x,y,load]...], dwell: {policy: [[x,y,count]...]} }
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maps import get_map                        # noqa: E402
from rest import build_haul_load, rest_cells_for  # noqa: E402
from run_suite import BASE, POL, _mk            # noqa: E402
from sim import MAPDSim, SimConfig              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_final", "figs")
os.makedirs(OUT, exist_ok=True)

STATES = ("idle", "park")
POLS = ("nearest", "static", "memir0", "memir")


def _trace(sim, t):
    """逐 timestep 记账。

    dwell —— 处于驻留状态的 agent 所在格 +1（静止障碍源）
    haul  —— 已取货、正送往 delivery endpoint 的 agent 所在格 +1（经验载货交通）
    """
    h = sim._dwell_hist
    g = sim._haul_hist
    if t < sim.cfg.warmup:
        return
    for a in sim.agents:
        if a["state"] in STATES:
            h[a["pos"]] += 1
        elif a["state"] == "busy" and a.get("picked"):
            g[a["pos"]] += 1


def _job(job):
    j = dict(job)
    map_name =     j.pop("map_name", "W1")
    pol = j.pop("method", "static")   # _mk 用 method 记录策略名
    seed = j.pop("seed", 0)
    j.pop("exp", None)
    wh = get_map(map_name)
    cfg = SimConfig(seed=seed, **j)
    sim = MAPDSim(wh, cfg)
    sim._dwell_hist = defaultdict(int)
    sim._haul_hist = defaultdict(int)
    sim._trace_cb = _trace
    r = sim.run()
    return dict(map=map_name, policy=pol, seed=seed, n=j["n_agents"],
                hist=dict(sim._dwell_hist), haul=dict(sim._haul_hist),
                service=r["service_post"], wait=r["wait_ratio"],
                detour=r["detour"],
                leg_haul=r["leg_haul"], leg_approach=r["leg_approach"])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seeds = list(range(6))
    map_name = "W1"
    jobs = []
    for pol in ("nearest", "static", "memir0", "memir"):
        for s in seeds:
            jobs.append(_mk("Fig5", pol, s, n, pol=pol, map_name=map_name))

    res = []
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            res.append(f.result())
            print("  done %-8s seed=%d svc=%.2f" %
                  (f.result()["policy"], f.result()["seed"],
                   f.result()["service"]), flush=True)

    wh = get_map(map_name)
    cells = rest_cells_for(wh, "any")
    flow = build_haul_load(wh, cells)

    dwell = {}
    haul_tot = defaultdict(float)
    stat = {}
    for pol in POLS:
        rs = [r for r in res if r["policy"] == pol]
        tot = defaultdict(float)
        for r in rs:
            for c, v in r["hist"].items():
                tot[c] += v
            for c, v in r["haul"].items():
                haul_tot[c] += v
        k = sum(tot.values()) or 1.0
        dwell[pol] = [[c[0], c[1], v / k] for c, v in sorted(tot.items())]
        # 经验干扰 = 驻留质量 × 载货交通的经验重叠（同一地图上的载货交通pooled）
        stat[pol] = dict(service=sum(r["service"] for r in rs) / len(rs),
                         leg_haul=sum(r["leg_haul"] for r in rs) / len(rs),
                         leg_approach=sum(r["leg_approach"] for r in rs)
                         / len(rs),
                         wait=sum(r["wait"] for r in rs) / len(rs) * 100,
                         detour=sum(r["detour"] for r in rs) / len(rs))
    hk = sum(haul_tot.values()) or 1.0
    haul = {tuple(c): v / hk for c, v in haul_tot.items()}
    for pol in POLS:
        rs = [r for r in res if r["policy"] == pol]
        tot = defaultdict(float)
        for r in rs:
            for c, v in r["hist"].items():
                tot[c] += v
        k = sum(tot.values()) or 1.0
        ov = sum((v / k) * haul.get(c, 0.0) for c, v in tot.items())
        ovU = ov * len(haul)
        stat[pol]["overlap"] = ovU
        print("%-8s service=%6.2f  L2=%5.2f  L3=%6.2f  wait=%5.3f%%  "
              "detour=%.4f  overlap(normalized)=%6.2f  cells=%d"
              % (pol, stat[pol]["service"], stat[pol]["leg_approach"],
                 stat[pol]["leg_haul"], stat[pol]["wait"],
                 stat[pol]["detour"], ovU, len(dwell[pol])))

    out = dict(
        map=map_name, n=n, seeds=len(seeds),
        layout=dict(W=wh.W, H=wh.H,
                    grid=[[1 if c else 0 for c in row] for row in wh.grid],
                    station_top=wh.station_top, park_rows=wh.park_rows,
                    margin=wh.station_top),
        flow=[[c[0], c[1], float(v)] for c, v in sorted(flow.items())],
        haul=[[c[0], c[1], v] for c, v in sorted(haul.items())],
        dwell=dwell, stat=stat)
    p = os.path.join(OUT, "fig5_dwell_%s_%d.json" % (map_name, n))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("->", p)


if __name__ == "__main__":
    main()
