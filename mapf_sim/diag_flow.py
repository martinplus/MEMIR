"""驻留干扰避让：扫描 flow_cost，检验能否同时保住 L2 增益并压回 L3 惩罚。

mechanism (diag_legs 实测)：空闲 agent 簇聚于热点 → 其所在格恰位于
pickup→station 的**主干搬运通道**上 → 送货 agent 必须绕行/等待 →
L3 从 29.4 涨到 33.7，完全吃掉 L2 省下的 ~2.6。

修正：把「该格承载多少条搬运最短路」做成结构化先验，加入 1-中位数目标：
    c* = argmin_c [ Σ_r p̂_r d̄(c,r) + β·d(pos,c) + ψ·flow(c) ]
flow(c)∈[0,1] 离线算一次；ψ=0 退化为原方法。
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

MEM = dict(rest_policy="exp", demand_mode="ema", tau_phi3=100, rest_beta=0.5)


def run(j):
    j = dict(j)
    m = j.pop("map_name", "W1")
    tag = j.pop("__tag", "")
    n = j.pop("__n", 0)
    seed = j.get("seed", 0)
    wh = get_map(m)
    try:
        r = MAPDSim(wh, SimConfig(**j)).run()
    except Exception as e:                       # noqa: BLE001
        return dict(tag=tag, n=n, seed=seed, err=repr(e))
    return dict(tag=tag, n=n, seed=seed, err="",
                svc=r["service_time"], l1=r["leg_assign"], l2=r["leg_approach"],
                l3=r["leg_haul"], resp=r["resp_dist"], mv=r["rest_moves"],
                det=r["detour"], pend=r["mean_pending"], idle=r["idle_ratio"])


def main():
    ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else [16, 24, 32, 40]
    seeds = 6
    combos = [("psi0", dict(flow_cost=0.0)),
              ("psi20", dict(flow_cost=20.0)),
              ("psi40", dict(flow_cost=40.0)),
              ("psi80", dict(flow_cost=80.0)),
              ("psi160", dict(flow_cost=160.0)),
              ("psi1e4", dict(flow_cost=1e4))]
    ref = [("static", dict(rest_policy="static", demand_mode="none"))]
    jobs = []
    for n in ns:
        for tag, kw in combos:
            for s in range(seeds):
                j = dict(seed=s, n_agents=n, map_name="W1")
                j.update({k: v for k, v in BASE.items() if k != "map_name"})
                j.update(MEM)
                j.update(kw)
                j.update(__tag=tag, __n=n)
                jobs.append(j)
        for tag, kw in ref:
            for s in range(seeds):
                j = dict(seed=s, n_agents=n, map_name="W1")
                j.update({k: v for k, v in BASE.items() if k != "map_name"})
                j.update(kw)
                j.update(__tag=tag, __n=n)
                jobs.append(j)
    print("jobs=%d" % len(jobs), flush=True)
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 50 == 0:
                print("  [%4d/%4d %5.0fs]" % (i, len(jobs), time.time() - t0),
                      flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        if r["err"]:
            print("ERR", r)
            continue
        agg[(r["n"], r["tag"])].append(r)
    tags = [t for t, _ in combos] + [t for t, _ in ref]
    hdr = "%-4s %-9s | %7s %7s %7s %7s %7s %6s" % (
        "n", "variant", "svc", "L1", "L2", "L3", "resp", "moves")
    print("\n" + hdr)
    print("-" * len(hdr))
    for n in ns:
        for tg in tags:
            g = agg.get((n, tg))
            if not g:
                continue
            f = lambda k: sum(x[k] for x in g) / len(g)   # noqa: E731
            print("%-4d %-9s | %7.2f %7.2f %7.2f %7.2f %7.2f %6.0f" %
                  (n, tg, f("svc"), f("l1"), f("l2"), f("l3"), f("resp"),
                   f("mv")))
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
