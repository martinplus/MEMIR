"""诊断：倒 U 右端（大舰队 ACO 失效）究竟是机制（拥挤外部性）
还是 artifact（rest cell 永久占位阻塞通道导致任务规划失败）。

对照：rest_set = any（停在通道，永久占位，可能阻塞）
      rest_set = park（只停 non-task endpoint，标准 well-formed 设定）
若 park 下 ACO 恢复优于 static，则右端失效是 artifact；
若 park 下 ACO 仍劣于 static，则是真实竞争外部性。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warehouse import Warehouse          # noqa: E402
from sim import MAPDSim, SimConfig       # noqa: E402

WH = None


def get_wh():
    global WH
    if WH is None:
        WH = Warehouse(width=41, height=21, station_stride=6)
    return WH


def one(n, rs, seed, T=3000, warmup=1000, **kw):
    wh = get_wh()
    cfg = SimConfig(n_agents=n, T=T, warmup=warmup, zipf_alpha=2.0,
                    arrival_mode="open", lmbda=0.08, capacity=1,
                    rest_set=rs, drift_mode="static", seed=seed, **kw)
    sim = MAPDSim(wh, cfg)
    t0 = time.time()
    r = sim.run()
    r["sec"] = time.time() - t0
    return r


def main():
    seeds = [0, 1, 2]
    for n in (16, 24, 32):
        for rs in ("any", "park"):
            cases = [
                ("nearest", dict(rest_policy="nearest", demand_mode="none")),
                ("static", dict(rest_policy="static", demand_mode="none")),
                ("aco", dict(rest_policy="exp", demand_mode="ema", tau_phi3=100)),
                ("oracle", dict(rest_policy="oracle", demand_mode="oracle")),
            ]
            for name, kw in cases:
                rows = [one(n, rs, seed=s, **kw) for s in seeds]
                svc = [r["service_post"] for r in rows]
                pf = [r["plan_fail"] for r in rows]
                tp = [r["throughput_post"] for r in rows]
                mv = [r["rest_moves"] for r in rows]
                m = lambda v: sum(v) / len(v)
                print(f"n={n:>2} rest={rs:<4} {name:<8} "
                      f"service={m(svc):7.2f} plan_fail={m(pf):9.1f} "
                      f"tp={m(tp):6.2f} moves={m(mv):7.1f} "
                      f"sec={m([r['sec'] for r in rows]):5.1f}")
        print("-" * 78)


if __name__ == "__main__":
    main()
