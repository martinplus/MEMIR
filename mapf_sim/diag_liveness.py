"""Liveness 修复验证：dwell-safety + preempt 是否消除 |A| 较大时的规划失败级联。

对照矩阵（全部在 rest_set=any 下，保住泊位自由度）：
    base     : 现状（无任何守护）
    safety   : + Dwell-Safety 不变量（残余图必须连通）
    preempt  : + 让行（规划失败时驱逐阻塞的空闲 agent）
    both     : 两者都开
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warehouse import Warehouse          # noqa: E402
from sim import MAPDSim, SimConfig       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_open", "diag_liveness.txt")
WH = None


def get_wh():
    global WH
    if WH is None:
        WH = Warehouse(width=41, height=21, station_stride=6)
    return WH


def one(n, variant, seed, T=2500, warmup=800, **kw):
    ds = "safety" in variant
    pe = 2 if "preempt" in variant else 0
    cfg = SimConfig(n_agents=n, T=T, warmup=warmup, zipf_alpha=2.0,
                    arrival_mode="open", lmbda=0.08, capacity=1,
                    rest_set="any", drift_mode="static", seed=seed,
                    dwell_safety=ds, preempt=pe, **kw)
    r = MAPDSim(get_wh(), cfg).run()
    return r


VARIANTS = ["base", "safety", "preempt", "both"]
CASES = [
    ("static", dict(rest_policy="static", demand_mode="none")),
    ("aco", dict(rest_policy="exp", demand_mode="ema", tau_phi3=100)),
]


def main():
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    for n in (24, 32):
        for name, kw in CASES:
            for var in VARIANTS:
                rows = [one(n, var, seed=s, **kw) for s in (0, 1)]
                m = lambda key: sum(r[key] for r in rows) / len(rows)
                emit(f"n={n:>2} {name:<7} {var:<8} "
                     f"service={m('service_post'):7.2f} "
                     f"tp={m('throughput_post'):6.2f} "
                     f"fail={m('plan_fail'):8.1f} "
                     f"att={m('plan_attempt_fail'):8.1f} "
                     f"yield={m('preempt_yields'):7.1f} "
                     f"chk={m('dwell_checks'):8.1f} "
                     f"dwellblk={m('dwell_blocked'):8.1f} "
                     f"dviol={m('dwell_viol'):8.1f} "
                     f"conf={m('conflicts'):5.1f} "
                     f"ms/dec={m('decide_ns')/1e6/max(m('decide_cnt'),1):6.3f} "
                     f"wall={m('wall_sec'):5.1f}s")
            emit("")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
