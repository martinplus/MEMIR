"""大 fleet 下 MEMIR 略逊于 uniform staging 的两种修正机制。

诊断 1（diag_softmax 已做）：ACS 局部衰减 ξ 在 |A|=16/24 显著改善，但 |A|>=32 反噬。
根因假设：空闲率 >90% 时，**重定位本身的行程代价**已超过预测的边际收益，
而 static 从不移动（零搬迁代价）。因此正解是给重定位加代价正则。

本脚本测试两类"动还是不动"的机制：
    (a) β rigrave：目标函数 = Σ_r p̂_r d̄(c,r) + β·d(pos,c)（带搬迁代价的加权中位）
    (b) move_thresh：迟滞，只有当新位置的期望收益超过阈值时才动
    (c) softmax / Plackett-Luce 概率构造（ACO 标准做法，保留覆盖率）
并以 nearest / static / greedy 作为参照。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_suite import BASE, _run_job                       # noqa: E402
from stats_util import paired_bootstrap, wilcoxon_signed_rank, pct  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_open", "diag_move_cost.txt")
SEEDS = tuple(range(10))

MEM = dict(rest_policy="exp", demand_mode="ema", tau_phi3=100)
MODELS = [
    ("nearest", dict(rest_policy="nearest", demand_mode="none")),
    ("static", dict(rest_policy="static", demand_mode="none")),
    ("memir_greedy", dict(MEM)),
    ("memir_b0.2", dict(MEM, rest_beta=0.2)),
    ("memir_b0.5", dict(MEM, rest_beta=0.5)),
    ("memir_b1.0", dict(MEM, rest_beta=1.0)),
    ("memir_th1", dict(MEM, move_thresh=1.0)),
    ("memir_th3", dict(MEM, move_thresh=3.0)),
    ("memir_soft_T0.5", dict(MEM, rest_policy="softmax", rest_temp=0.5)),
    ("memir_soft_T2", dict(MEM, rest_policy="softmax", rest_temp=2.0)),
    ("memir_b0.5_soft1", dict(MEM, rest_beta=0.5, rest_policy="softmax",
                              rest_temp=1.0)),
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
                jobs.append(dict(exp="movecost", method=tag, seed=s,
                                 map_name="W1", n_agents=n, **j))
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.get("error"):
                print("ERR", r.get("method"), r["error"][:140], flush=True)
                continue
            res.setdefault((r["n_agents"], r["method"]), []).append(r)
            if i % 110 == 0:
                print("[%4d/%4d %5.0fs]" % (i, len(jobs), time.time() - t0),
                      flush=True)

    lines = []
    for n in AGENTS:
        st = res.get((n, "static"), [])
        ref = {r["seed"]: r["service_post"] for r in st}
        refm = sum(ref.values()) / max(len(ref), 1)
        mv = sum(r.get("rest_moves", 0) for r in res.get((n, "memir_greedy"), [])) \
            / max(len(res.get((n, "memir_greedy"), [])), 1)
        near = res.get((n, "nearest"), [])
        print("--- |A|=%d  static=%.2f  nearest=%.2f  (greedy 搬迁次数=%.0f) ---"
              % (n, refm,
                 sum(r["service_post"] for r in near) / max(len(near), 1), mv),
              flush=True)
        lines.append("|A|=%d" % n)
        for tag, _ in MODELS:
            rows = res.get((n, tag), [])
            if not rows:
                continue
            svc = [r["service_post"] for r in rows]
            m = sum(svc) / len(svc)
            mv = sum(r.get("rest_moves", 0) for r in rows) / len(rows)
            extra = ""
            if tag not in ("nearest", "static") and ref:
                ns = {r["seed"]: r["service_post"] for r in rows}
                sd = sorted(set(ref) & set(ns))
                d = [ns[s] - ref[s] for s in sd]
                dm, lo, hi, pb = paired_bootstrap(d)
                w, z, pw = wilcoxon_signed_rank(d)
                extra = ("Δstatic=%+6.2f%% CI[%+.2f,%+.2f] p=%.3g"
                         % (pct(m, refm), lo, hi, pw))
            s = "%-18s svc=%7.2f moves=%6.0f  %s" % (tag, m, mv, extra)
            print(s, flush=True)
            lines.append(s)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
