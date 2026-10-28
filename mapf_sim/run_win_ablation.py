"""A_main 的补充消融：即时窗口 vs 跨窗累积记忆（2026-09-27）。

背景：A_main 里 memir 用 demand_mode="ema"（跨窗口指数累积，时间常数 τ=100）。
为了把"跨窗记忆"本身的贡献从"用外生场做预测"里剥离出来，补跑
demand_mode="window"（α=1，纯当前窗口滑动平均，窗口之间不保留任何积累）。
两者除记忆积分长度外完全一致：同样的 p̂/β/ψ、同样的 TP 执行器、同样的 CRN。

    memir_win   = 即时窗口 + ψ=30
    memir_win0  = 即时窗口 + ψ=0

于是跨窗记忆的净贡献 = gain(memir) − gain(memir_win)，ψ 的贡献 = gain(memir_win) − gain(memir_win0)。

用法： python mapf_sim/run_win_ablation.py
结果追加进 results_final/raw_A_main.csv（不覆盖原有行）。
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_suite as RS                                          # noqa: E402


def main():
    seeds = list(range(10))
    jobs = []
    for n in (4, 6, 8, 10, 12, 16, 24, 32, 40):
        for meth, wl in (("memir_win", 30.0), ("memir_win0", 0.0)):
            for s in seeds:
                jobs.append(RS._mk("A_main", meth, s, n, pol="memir",
                                   demand_mode="window", flow_cost=wl))
    print("jobs =", len(jobs), flush=True)
    t0 = time.time()
    rows = RS._pool(jobs, 12, "A_main/window", t0)
    bad = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    print("done %d ok, %d failed, %.0fs" % (len(rows), len(bad),
                                            time.time() - t0), flush=True)
    for r in bad[:5]:
        print("   ERR", r.get("n_agents"), r.get("seed"),
              str(r.get("error"))[:160], flush=True)

    path = os.path.join(RS.OUT, "raw_A_main.csv")
    with open(path, newline="", encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    fields = sorted({k for r in old for k in r} | {k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in old:
            w.writerow(r)
        for r in rows:
            w.writerow(r)
    print("appended %d rows -> %s (total %d)"
          % (len(rows), path, len(old) + len(rows)), flush=True)


if __name__ == "__main__":
    main()
