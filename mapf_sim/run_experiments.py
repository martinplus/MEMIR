"""批量实验驱动：基线 × 信息素 × Oracle 的全面对比。零第三方依赖，支持多进程。

输出 CSV 到 results/ 目录。
"""
import os
import sys
import csv
import time
import statistics
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warehouse import Warehouse
from sim import MAPDSim, SimConfig

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
WH_CACHE = {}


def get_wh(station_stride=6):
    if station_stride not in WH_CACHE:
        WH_CACHE[station_stride] = Warehouse(width=41, height=21,
                                             station_stride=station_stride)
    return WH_CACHE[station_stride]


def one_run(job):
    """job: dict。返回一行结果。"""
    wh = get_wh(job.get("station_stride", 6))
    # 注意：oracle_tau 必须传入，否则 fixed_tau=None 会退化为全 1 场，
    # 使 Oracle 与基线完全等价（曾导致结论无效）。
    cfg = SimConfig(**{k: v for k, v in job.items()
                       if k not in ("method", "station_stride")})
    r = MAPDSim(wh, cfg).run()
    return dict(
        method=job["method"], n_agents=cfg.n_agents, alpha=cfg.zipf_alpha,
        drift_mode=cfg.drift_mode, drift_param=cfg.drift_param,
        tau_phi=cfg.tau_phi, modulate=cfg.modulate, ph_mode=cfg.ph_mode,
        prio=cfg.prio, seed=cfg.seed,
        throughput=r["throughput_post"], throughput_all=r["throughput"],
        completed=r["completed_post"], service_time=r["service_time"],
        detour=r["detour"], wait_ratio=r["wait_ratio"],
        conflicts=r["conflicts"], plan_fail=r["plan_fail"],
    )


def build_jobs(oracle_tau=None, T=1200, seeds=range(8), warmup=300):
    """构造完整实验矩阵。"""
    jobs = []
    loads = [20, 40, 60]
    alphas = [0.0, 1.0, 2.5]
    taus = [30, 300, 3000, 30000]

    for n in loads:
        for a in alphas:
            base_kw = dict(n_agents=n, zipf_alpha=a, T=T, warmup=warmup,
                           n_options=3, n_deliveries=3, drift_mode="static")
            # ---- 基线 ----
            for s in seeds:
                jobs.append(dict(base_kw, method="greedy", seed=s,
                                 ph_mode="none", modulate="none",
                                 assign_mode="greedy"))
                jobs.append(dict(base_kw, method="hungarian", seed=s,
                                 ph_mode="none", modulate="none",
                                 assign_mode="hungarian"))
                # ---- 信息素（不同时间常数）----
                for tp in taus:
                    jobs.append(dict(base_kw, method="ph_ema", seed=s,
                                     ph_mode="ema", modulate="mean",
                                     tau_phi=tp, assign_mode="hungarian"))
                # ---- EMA 无记忆对照（纯窗口平均，alpha=1）----
                jobs.append(dict(base_kw, method="ph_window", seed=s,
                                 ph_mode="window", modulate="mean",
                                 tau_phi=300, assign_mode="hungarian"))
                # ---- Oracle ----
                if oracle_tau and (n, a) in oracle_tau:
                    jobs.append(dict(base_kw, method="oracle", seed=s,
                                     ph_mode="oracle", modulate="mean",
                                     oracle_tau=oracle_tau[(n, a)],
                                     assign_mode="hungarian"))
    return jobs


def collect_oracle(loads, alphas, T=1200, seeds=range(4)):
    """先跑基线收集各 (负载, alpha) 下的真实拥堵场，用于 Oracle。"""
    out = {}
    for n in loads:
        for a in alphas:
            acc = None
            cnt = None
            for s in seeds:
                wh = get_wh()
                cfg = SimConfig(n_agents=n, zipf_alpha=a, T=T, warmup=300,
                                seed=s, n_options=3, n_deliveries=3,
                                ph_mode="none", modulate="none")
                r = MAPDSim(wh, cfg).run()
                o = r["oracle_tau"]
                if acc is None:
                    acc = [0.0] * len(o)
                    cnt = [0] * len(o)
                for i in range(len(o)):
                    if o[i] > 1.0:
                        acc[i] += o[i]
                        cnt[i] += 1
            out[(n, a)] = [(acc[i] / cnt[i]) if cnt[i] else 1.0
                           for i in range(len(acc))]
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    loads = [20, 40, 60]
    alphas = [0.0, 1.0, 2.5]

    print("[1/3] collecting oracle fields ...")
    orc = collect_oracle(loads, alphas)
    for k, v in sorted(orc.items()):
        print("   n=%d alpha=%.1f  tau range %.2f-%.2f" % (k[0], k[1], min(v), max(v)))

    jobs = build_jobs(orc)
    print("[2/3] running %d jobs ..." % len(jobs))

    nproc = min(8, os.cpu_count() or 1)
    rows = []
    if nproc > 1:
        with mp.Pool(nproc) as pool:
            for i, r in enumerate(pool.imap_unordered(one_run, jobs, chunksize=4)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print("   %d/%d  %.0fs" % (i + 1, len(jobs), time.time() - t0))
    else:
        for i, j in enumerate(jobs):
            rows.append(one_run(j))

    path = os.path.join(OUT, "main_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[3/3] wrote %s (%d rows) in %.0fs" % (path, len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
