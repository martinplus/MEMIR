"""liveness 修复消融 II：修好 variant 命名，并扫描 dwell_probe 宽度。

对照（全部 rest_set=any，保住泊位自由度）：
    base            : 无守护
    safety          : TER-Safety（候选里取 TER 违规最小者）
    preempt         : 让行（规划失败时驱逐阻塞的空闲 agent）
    safety_preempt  : 两者都开
另外扫描 safety 的候选检查宽度 probe ∈ {2,4,8,16}。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warehouse import Warehouse          # noqa: E402
from sim import MAPDSim, SimConfig       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_open", "diag_liveness3.txt")
WH = None


def get_wh():
    global WH
    if WH is None:
        WH = Warehouse(width=41, height=21, station_stride=6)
    return WH


def one(n, seed, T=2500, warmup=800, **kw):
    cfg = SimConfig(n_agents=n, T=T, warmup=warmup, zipf_alpha=2.0,
                    arrival_mode="open", lmbda=0.08, capacity=1,
                    rest_set="any", drift_mode="static", seed=seed, **kw)
    return MAPDSim(get_wh(), cfg).run()


ACO = dict(rest_policy="exp", demand_mode="ema", tau_phi3=100)
STA = dict(rest_policy="static", demand_mode="none")
VARIANTS = [
    ("base", dict(dwell_safety=False, preempt=0)),
    ("safety", dict(dwell_safety=True, preempt=0)),
    ("preempt", dict(dwell_safety=False, preempt=2)),
    ("safety_preempt", dict(dwell_safety=True, preempt=2)),
]


def main():
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    def show(tag, rows, extra=""):
        m = lambda key: sum(r[key] for r in rows) / len(rows)
        emit(f"{tag:<34} svc={m('service_post'):7.2f} "
             f"fail={m('plan_fail'):6.1f} att={m('plan_attempt_fail'):7.1f} "
             f"yld={m('preempt_yields'):6.1f} blk={m('dwell_blocked'):7.1f} "
             f"conf={m('conflicts'):4.1f} "
             f"ms/dec={m('decide_ns')/1e6/max(m('decide_cnt'),1):6.3f} "
             f"wall={m('wall_sec'):5.1f}s {extra}")

    seeds = (0, 1, 2)
    # ---- 1. 四变体 × {static, aco} × n ----
    for n in (16, 24, 32):
        for name, kw in (("static", STA), ("aco", ACO)):
            for var, vk in VARIANTS:
                t0 = time.perf_counter()
                rows = [one(n, s, **kw, **vk) for s in seeds]
                show(f"n={n:<3} {name:<6} {var}", rows,
                     f"real={time.perf_counter()-t0:5.1f}s")
            emit("")
    # ---- 2. probe 敏感度（只跑 aco @ 24/32）----
    for n in (24, 32):
        for probe in (2, 4, 8, 16):
            rows = [one(n, s, **ACO, dwell_safety=True, preempt=0,
                        dwell_probe=probe) for s in seeds]
            show(f"probe n={n:<3} p={probe:<3} aco", rows)
        emit("")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
