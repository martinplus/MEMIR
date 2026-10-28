"""诊断（精简版）：倒 U 右端是否 artifact。

对比 rest_set=any（停在通道 + 永久占位，可能阻断窄通道）
    vs  rest_set=park（只停 non-task endpoint，标准 well-formed 泊位）

输出追加到 ../results_open/diag_rest6.txt
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warehouse import Warehouse          # noqa: E402
from sim import MAPDSim, SimConfig       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_open", "diag_rest6.txt")
WH = None


def get_wh():
    global WH
    if WH is None:
        WH = Warehouse(width=41, height=21, station_stride=6)
    return WH


def one(n, rs, seed, T=2500, warmup=800, **kw):
    cfg = SimConfig(n_agents=n, T=T, warmup=warmup, zipf_alpha=2.0,
                    arrival_mode="open", lmbda=0.08, capacity=1,
                    rest_set=rs, drift_mode="static", seed=seed, **kw)
    t0 = time.time()
    r = MAPDSim(get_wh(), cfg).run()
    r["sec"] = time.time() - t0
    return r


CASES = [
    ("static", dict(rest_policy="static", demand_mode="none")),
    ("aco", dict(rest_policy="exp", demand_mode="ema", tau_phi3=100)),
    ("oracle", dict(rest_policy="oracle", demand_mode="oracle")),
]


def main():
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    for n in (24, 32):
        for rs in ("any", "park"):
            for name, kw in CASES:
                rows = [one(n, rs, seed=s, **kw) for s in (0, 1)]
                svc = sum(r["service_post"] for r in rows) / len(rows)
                pf = sum(r["plan_fail"] for r in rows) / len(rows)
                tp = sum(r["throughput_post"] for r in rows) / len(rows)
                emit(f"n={n:>2} rest={rs:<4} {name:<7} service={svc:7.2f} "
                     f"plan_fail={pf:8.1f} tp={tp:6.2f} "
                     f"sec/run={rows[0]['sec']:5.1f}")
        emit("-" * 62)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
