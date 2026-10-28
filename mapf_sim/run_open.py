"""Phase 3：open-loop 非饱和 MAPD + 空闲 agent 预定位 —— 完整实验。

研究问题：信息素式经验记忆在什么条件下、以什么速率、带来多少服务时间增益？

核心假设（可预见性假设）：
    记忆的价值 = 「未来任务分布不可观的程度」。
    closed-loop 饱和系统里整个任务池可见 → 记忆冗余（Phase 1/2 已证）；
    open-loop 非饱和系统里未来到达不可观 → 记忆有预测价值。
    由此推出三条可证伪预测：
      P1  增益随舰队规模单调下降（agent 越多，覆盖越容易，预测越不必要）
      P2  增益随空间不均匀度单调上升（分布越偏，猜位置越有价值）
      P3  α=0（均匀分布）时增益 → 0（无可预测的结构）

输出：results_open/raw.csv, summary.csv
"""
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warehouse import Warehouse          # noqa: E402
from sim import MAPDSim, SimConfig       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_open")
SEEDS = list(range(10))


def make_wh():
    return Warehouse(width=41, height=21, station_stride=6)


def jobs():
    """返回 [(group, method, cfg_dict), ...]"""
    J = []

    def add(group, method, **kw):
        J.append((group, method, kw))

    # ---------------------------------------------------------------- E1
    # 基线对比：小舰队 open-loop，静止环境
    E1 = dict(n_agents=8, T=6000, warmup=2000, zipf_alpha=2.0,
              arrival_mode="open", lmbda=0.08, capacity=1,
              rest_set="any", drift_mode="static")
    add("E1_baseline", "nearest", rest_policy="nearest", demand_mode="none", **E1)
    add("E1_baseline", "random", rest_policy="random", demand_mode="none", **E1)
    add("E1_baseline", "static", rest_policy="static", demand_mode="none", **E1)
    add("E1_baseline", "oracle", rest_policy="oracle", demand_mode="oracle", **E1)
    add("E1_baseline", "aco_t100", rest_policy="exp", demand_mode="ema",
        tau_phi3=100, **E1)
    add("E1_baseline", "aco_t500", rest_policy="exp", demand_mode="ema",
        tau_phi3=500, **E1)
    add("E1_baseline", "aco_t2000", rest_policy="exp", demand_mode="ema",
        tau_phi3=2000, **E1)
    add("E1_baseline", "aco_window", rest_policy="exp", demand_mode="window", **E1)
    add("E1_baseline", "aco_rl_credit", rest_policy="exp", demand_mode="ema",
        tau_phi3=500, rl_mode="adv", rl_eta=0.05, elig=0.995, **E1)
    add("E1_baseline", "aco_rl_meta", rest_policy="exp", demand_mode="ema",
        meta_mode=True, tau_fast=100, meta_ratio=8, meta_scale=0.30, **E1)

    # ---------------------------------------------------------------- E2
    # 舰队规模扫描（P1：增益随规模下降）。λ 固定，故饱和度随之下降。
    for n in (4, 6, 8, 12, 16, 24, 32):
        E2 = dict(n_agents=n, T=6000, warmup=2000, zipf_alpha=2.0,
                  arrival_mode="open", lmbda=0.08, capacity=1,
                  rest_set="any", drift_mode="static")
        g = "E2_fleet"
        add(g, "nearest_n%d" % n, rest_policy="nearest", demand_mode="none", **E2)
        add(g, "static_n%d" % n, rest_policy="static", demand_mode="none", **E2)
        add(g, "oracle_n%d" % n, rest_policy="oracle", demand_mode="oracle", **E2)
        add(g, "aco_n%d" % n, rest_policy="exp", demand_mode="ema",
            tau_phi3=100, **E2)
        add(g, "meta_n%d" % n, rest_policy="exp", demand_mode="ema",
            meta_mode=True, tau_fast=100, meta_ratio=8, meta_scale=0.30, **E2)

    # ---------------------------------------------------------------- E3
    # 空间不均匀度扫描（P2/P3：α=0 时增益应 → 0）
    for a in (0.0, 0.5, 1.0, 2.0, 3.0):
        E3 = dict(n_agents=8, T=6000, warmup=2000, zipf_alpha=a,
                  arrival_mode="open", lmbda=0.08, capacity=1,
                  rest_set="any", drift_mode="static")
        g = "E3_alpha"
        add(g, "nearest_a%.1f" % a, rest_policy="nearest", demand_mode="none", **E3)
        add(g, "static_a%.1f" % a, rest_policy="static", demand_mode="none", **E3)
        add(g, "oracle_a%.1f" % a, rest_policy="oracle", demand_mode="oracle", **E3)
        add(g, "aco_a%.1f" % a, rest_policy="exp", demand_mode="ema",
            tau_phi3=100, **E3)

    # ---------------------------------------------------------------- E4
    # 时间常数扫描：倒 U 型检验（H2）与「最优 τ 随环境迁移」
    for lam in (0.02, 0.04, 0.08):
        E4 = dict(n_agents=8, T=8000, warmup=2500, zipf_alpha=2.0,
                  arrival_mode="open", lmbda=lam, capacity=1,
                  rest_set="any", drift_mode="static")
        g = "E4_tau_l%.2f" % lam
        add(g, "nearest", rest_policy="nearest", demand_mode="none", **E4)
        for tp in (20, 50, 100, 300, 1000, 3000):
            add(g, "aco_t%d" % tp, rest_policy="exp", demand_mode="ema",
                tau_phi3=tp, **E4)
        add(g, "meta", rest_policy="exp", demand_mode="ema",
            meta_mode=True, tau_fast=100, meta_ratio=8, meta_scale=0.30, **E4)

    # ---------------------------------------------------------------- E5
    # 环境漂移：固定 τ 失效，元适应应当保持鲁棒（H5）
    for dm, per in (("static", 0), ("shift", 1500), ("shift", 750),
                    ("shift", 400), ("periodic", 1500), ("periodic", 750)):
        E5 = dict(n_agents=8, T=6000, warmup=2000, zipf_alpha=2.0,
                  arrival_mode="open", lmbda=0.08, capacity=1,
                  rest_set="any", drift_mode=dm, drift_param=per)
        g = "E5_drift"
        m = "%s%d" % (dm, per)
        add(g, "nearest_" + m, rest_policy="nearest", demand_mode="none", **E5)
        for tp in (100, 500, 2000):
            add(g, "aco_t%d_%s" % (tp, m), rest_policy="exp",
                demand_mode="ema", tau_phi3=tp, **E5)
        add(g, "meta_" + m, rest_policy="exp", demand_mode="ema",
            meta_mode=True, tau_fast=100, meta_ratio=8, meta_scale=0.30, **E5)
        add(g, "meta60_" + m, rest_policy="exp", demand_mode="ema",
            meta_mode=True, tau_fast=60, meta_ratio=8, meta_scale=0.30, **E5)
        add(g, "oracle_" + m, rest_policy="oracle", demand_mode="oracle", **E5)

    # ---------------------------------------------------------------- E6
    # 消融：ACO 各组件与两种 RL 的贡献
    E6 = dict(n_agents=8, T=6000, warmup=2000, zipf_alpha=2.0,
              arrival_mode="open", lmbda=0.08, capacity=1,
              rest_set="any", drift_mode="shift", drift_param=750)
    g = "E6_ablation"
    add(g, "nearest", rest_policy="nearest", demand_mode="none", **E6)
    add(g, "aco_plain", rest_policy="exp", demand_mode="ema", tau_phi3=100, **E6)
    for xi in (0.05, 0.15, 0.30):
        add(g, "aco_xi%.2f" % xi, rest_policy="exp", demand_mode="ema",
            tau_phi3=100, rest_xi=xi, **E6)
    add(g, "aco_rl_credit", rest_policy="exp", demand_mode="ema", tau_phi3=100,
        rl_mode="adv", rl_eta=0.05, elig=0.995, **E6)
    add(g, "aco_rl_meta", rest_policy="exp", demand_mode="ema", meta_mode=True,
        tau_fast=100, meta_ratio=8, meta_scale=0.30, **E6)
    add(g, "aco_rl_both", rest_policy="exp", demand_mode="ema", meta_mode=True,
        tau_fast=100, meta_ratio=8, meta_scale=0.30,
        rl_mode="adv", rl_eta=0.05, elig=0.995, **E6)
    return J


FIELDS = None


def main():
    os.makedirs(OUT, exist_ok=True)
    wh = make_wh()
    J = jobs()
    total = len(J) * len(SEEDS)
    print("jobs=%d  seeds=%d  total runs=%d" % (len(J), len(SEEDS), total))
    fout = open(os.path.join(OUT, "raw.csv"), "w", newline="", encoding="utf-8")
    w = None
    t0 = time.time()
    done = 0
    for group, method, cfg in J:
        # oracle 模式需要事后真实分布：先用 nearest 采样累计计数构造
        oracle_lam = None
        if cfg.get("demand_mode") == "oracle":
            cum = [0.0] * wh.n_regions
            probe = dict(cfg)
            probe.pop("rest_policy", None)
            probe["demand_mode"] = "none"
            for s in SEEDS:
                c = SimConfig(seed=s, rest_policy="nearest", **probe)
                r = MAPDSim(wh, c).run()
                for i in range(wh.n_regions):
                    cum[i] += r["demand_cum"][i]
            tot = sum(cum)
            oracle_lam = [x / tot for x in cum] if tot > 0 else None
        for s in SEEDS:
            c = SimConfig(seed=s, oracle_lam=oracle_lam, **cfg)
            r = MAPDSim(wh, c).run()
            row = dict(group=group, method=method, seed=s)
            for k, v in r.items():
                if isinstance(v, (list, tuple)):
                    continue
                row[k] = v
            row["tau_phi3"] = cfg.get("tau_phi3", "")
            row["zipf_alpha"] = cfg.get("zipf_alpha", "")
            row["n_agents"] = cfg.get("n_agents", "")
            row["lmbda"] = cfg.get("lmbda", "")
            row["drift_mode"] = cfg.get("drift_mode", "")
            row["drift_param"] = cfg.get("drift_param", "")
            row["meta_mode"] = cfg.get("meta_mode", False)
            if w is None:
                FIELDS = list(row.keys())
                w = csv.DictWriter(fout, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
            w.writerow(row)
            done += 1
        el = time.time() - t0
        print("[%5d/%5d %6.0fs] %-14s %s" % (done, total, el, group, method),
              flush=True)
    fout.close()
    print("done in %.0fs -> %s" % (time.time() - t0, os.path.join(OUT, "raw.csv")))


if __name__ == "__main__":
    main()
