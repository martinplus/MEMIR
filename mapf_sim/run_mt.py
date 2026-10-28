"""MT-MAPD 完整实验（Phase 2）。零第三方依赖，单进程顺序执行。

实验分组：
    E1  容量扫描       cap 1..6，验证 MT vs ST 的吞吐增益与边际行程规模经济
    E2  序列构造器对比  random / nn / cluster / 2opt / oracle
    E3  信息素参数扫描  γ × τ_φ × ξ（含 2opt 组合）
    E4  条件矩阵       空间不均匀度 α × agent 数
    E5  非平稳漂移      static / periodic / shift × τ_φ

输出：results_mt/raw.csv（每次运行一行）
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from warehouse import Warehouse          # noqa: E402
from sim import MAPDSim, SimConfig       # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "results_mt")
N_SEEDS = 6
T = 1200
WARMUP = 400
PF = 8.0

FIELDS = ["group", "method", "cap", "seq_mode", "ph2_mode", "tau_phi2", "gamma",
          "xi", "alpha", "n_agents", "drift_mode", "drift_param", "seed",
          "throughput_post", "completed_post", "service_time", "geom_per_batch",
          "margin_per_task", "steps_per_task", "avg_batch", "wait_ratio",
          "conflicts", "plan_fail", "tau2_spread"]


def make_wh():
    return Warehouse(width=41, height=21, station_stride=6)


def jobs():
    out = []

    def add(group, method, cap, seq, **kw):
        for s in range(N_SEEDS):
            j = dict(group=group, method=method, cap=cap, seq_mode=seq,
                     ph2_mode=kw.pop("ph2_mode", "none"),
                     tau_phi2=kw.pop("tau_phi2", 500.0),
                     gamma=kw.pop("gamma", 1.0),
                     xi=kw.pop("xi", 0.0),
                     alpha=kw.pop("alpha", 1.0),
                     n_agents=kw.pop("n_agents", 40),
                     drift_mode=kw.pop("drift_mode", "static"),
                     drift_param=kw.pop("drift_param", 0),
                     seed=s)
            j.update(kw)
            out.append(j)

    # ---- E1 容量扫描 ----
    for cap in [1, 2, 3, 4, 5, 6]:
        add("E1_capacity", "cap%d_nn" % cap, cap, "nn")

    # ---- E2 序列构造器 ----
    for cap in [3, 4]:
        for m in ["random", "nn", "cluster", "nn2opt", "cluster2opt", "oracle"]:
            add("E2_constructor", "%s_c%d" % (m, cap), cap, m)

    # ---- E3 信息素参数扫描 ----
    for tp in [100.0, 500.0, 2000.0]:
        for g in [2.0, 10.0]:
            for xi in [0.0, 0.1]:
                add("E3_pheromone", "ph_c4_t%d_g%g_x%g" % (tp, g, xi),
                    4, "ph", ph2_mode="ema", tau_phi2=tp, gamma=g, xi=xi)
                add("E3_pheromone", "ph2opt_c4_t%d_g%g_x%g" % (tp, g, xi),
                    4, "ph2opt", ph2_mode="ema", tau_phi2=tp, gamma=g, xi=xi)

    # ---- E4 条件矩阵 ----
    for alpha in [0.0, 1.0, 2.0]:
        for n in [20, 40, 60]:
            add("E4_conditions", "nn2opt_a%g_n%d" % (alpha, n), 4, "nn2opt",
                alpha=alpha, n_agents=n)
            add("E4_conditions", "ph2opt_a%g_n%d" % (alpha, n), 4, "ph2opt",
                ph2_mode="ema", tau_phi2=500.0, gamma=10.0, alpha=alpha,
                n_agents=n)
            add("E4_conditions", "oracle_a%g_n%d" % (alpha, n), 4, "oracle",
                alpha=alpha, n_agents=n)

    # ---- E5 非平稳漂移 ----
    for dm, dp in [("static", 0), ("periodic", 400), ("periodic", 1200),
                   ("shift", 400), ("shift", 1200)]:
        add("E5_drift", "nn2opt_%s%d" % (dm, dp), 4, "nn2opt",
            drift_mode=dm, drift_param=dp)
        for tp in [100.0, 500.0, 2000.0]:
            add("E5_drift", "ph2opt_%s%d_t%d" % (dm, dp, tp), 4, "ph2opt",
                ph2_mode="ema", tau_phi2=tp, gamma=10.0,
                drift_mode=dm, drift_param=dp)

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wh = make_wh()
    print(wh.summary())
    allj = jobs()
    print("总运行数: %d" % len(allj))
    t0 = time.time()
    path = os.path.join(OUT_DIR, "raw.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        done = 0
        for j in allj:
            cfg = SimConfig(
                n_agents=j["n_agents"], T=T, seed=j["seed"], warmup=WARMUP,
                zipf_alpha=j["alpha"], capacity=j["cap"], seq_mode=j["seq_mode"],
                ph2_mode=j["ph2_mode"], tau_phi2=j["tau_phi2"], gamma=j["gamma"],
                xi=j["xi"], pending_factor=PF,
                drift_mode=j["drift_mode"], drift_param=j["drift_param"])
            r = MAPDSim(wh, cfg).run()
            row = {k: j[k] for k in FIELDS if k in j}
            for k in ["throughput_post", "completed_post", "service_time",
                      "geom_per_batch", "margin_per_task", "steps_per_task",
                      "avg_batch", "wait_ratio", "conflicts", "plan_fail",
                      "tau2_spread"]:
                v = r.get(k, 0)
                row[k] = round(v, 4) if isinstance(v, float) else v
            w.writerow(row)
            done += 1
            if done % 60 == 0:
                print("  %d/%d  %.0fs" % (done, len(allj), time.time() - t0))
                f.flush()
    print("完成: %d 行 -> %s  (%.0fs)" % (done, path, time.time() - t0))


if __name__ == "__main__":
    main()
