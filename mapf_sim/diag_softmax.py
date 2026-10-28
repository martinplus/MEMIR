"""大 fleet 下「过度集中」问题：两种 ACO 修正是否恢复覆盖？

MEMIR 在 |A|>=32 时略逊于 static（均匀铺开），原因是加权中位把过多空闲 agent
压向热点。ACO 社区的两个标准机制可能修复：
    (a) softmax 概率构造：按 exp(-E(c)/T) 采样，保留覆盖而非全押一处
    (b) ACS 局部衰减 ξ：被占用的热门区域信息素就地衰减 -> 生态位分化
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_suite import BASE, _run_job                       # noqa: E402
from stats_util import paired_bootstrap, wilcoxon_signed_rank, pct  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_open", "diag_softmax.txt")
SEEDS = tuple(range(10))

MODELS = [
    ("nearest", dict(rest_policy="nearest", demand_mode="none")),
    ("static", dict(rest_policy="static", demand_mode="none")),
    ("memir_greedy", dict(rest_policy="exp", demand_mode="ema")),
    ("memir_xi0.05", dict(rest_policy="exp", demand_mode="ema", rest_xi=0.05)),
    ("memir_xi0.15", dict(rest_policy="exp", demand_mode="ema", rest_xi=0.15)),
    ("memir_xi0.30", dict(rest_policy="exp", demand_mode="ema", rest_xi=0.30)),
    ("memir_soft_T0.25", dict(rest_policy="softmax", demand_mode="ema",
                              rest_temp=0.25)),
    ("memir_soft_T0.5", dict(rest_policy="softmax", demand_mode="ema",
                             rest_temp=0.5)),
    ("memir_soft_T1", dict(rest_policy="softmax", demand_mode="ema",
                           rest_temp=1.0)),
    ("memir_soft_T2", dict(rest_policy="softmax", demand_mode="ema",
                           rest_temp=2.0)),
]
AGENTS = (8, 16, 24, 32, 40)


def main():
    jobs = []
    for n in AGENTS:
        for tag, kw in MODELS:
            for s in SEEDS:
                j = dict(BASE)
                j.pop("map_name", None)
                j.update(kw)
                j.update(T=3000, warmup=800)
                jobs.append(dict(exp="softmax", method=tag, seed=s,
                                 map_name="W1", n_agents=n, **j))
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.get("error"):
                print("ERR", r.get("method"), r["error"][:120], flush=True)
                continue
            res.setdefault((r["n_agents"], r["method"]), []).append(r)
            if i % 100 == 0:
                print("[%4d/%4d %5.0fs]" % (i, len(jobs), time.time() - t0),
                      flush=True)

    lines = []
    for n in AGENTS:
        st = res.get((n, "static"), [])
        ref = {r["seed"]: r["service_post"] for r in st}
        refm = sum(ref.values()) / max(len(ref), 1)
        print("--- |A|=%d  (static=%.2f, nearest=%.2f) ---"
              % (n, refm,
                 sum(r["service_post"] for r in res.get((n, "nearest"), []))
                 / max(len(res.get((n, "nearest"), [])), 1)), flush=True)
        lines.append("|A|=%d" % n)
        for tag, _ in MODELS:
            rows = res.get((n, tag), [])
            if not rows:
                continue
            svc = [r["service_post"] for r in rows]
            m = sum(svc) / len(svc)
            idle = sum(r["idle_ratio"] for r in rows) / len(rows) * 100
            extra = ""
            if tag not in ("nearest", "static") and ref:
                ns = {r["seed"]: r["service_post"] for r in rows}
                sd = sorted(set(ref) & set(ns))
                d = [ns[s] - ref[s] for s in sd]
                dm, lo, hi, pb = paired_bootstrap(d)
                w, z, pw = wilcoxon_signed_rank(d)
                extra = ("Δstatic=%+6.2f%% (Δ=%+.2f CI[%+.2f,%+.2f] p=%.3g)"
                         % (pct(m, refm), dm, lo, hi, pw))
            s = "%-18s svc=%7.2f idle=%5.1f%%  %s" % (tag, m, idle, extra)
            print(s, flush=True)
            lines.append(s)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
