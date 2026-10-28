"""论文最终实验套件（并行版）。零第三方依赖。

产出：results_final/raw_<exp>.csv，每行一次 run（含全部标量指标），
供 analyze_suite.py 做配对统计。

实验分组：
    A_main       主对照：5 种泊位策略 × 6 个 fleet 规模（Table 1）
    B_liveness   liveness 守护消融 + any/park 对照（Table 2）
    C_planner    tp（全时域）vs rhcr（窗口滚动）× 泊位策略（Table 3）
    D_tau        记忆时间常数 τ 敏感性 + 元适应（Table 4 / Fig）
    E_hetero     需求异质性 α × 到达强度 λ（Table 5）
    F_drift      非平稳漂移（Table 6）
    G_maps       地图泛化 W2 / W3（Table 7）
    H_runtime    规模可扩展性（runtime per decision）
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maps import get_map                      # noqa: E402
from sim import MAPDSim, SimConfig            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_final")

BASE = dict(T=4000, warmup=1000, zipf_alpha=2.0, arrival_mode="open",
            lmbda=0.08, capacity=1, rest_set="any", drift_mode="static",
            drift_param=2000, planner_mode="tp", assign_mode="hungarian",
            map_name="W1", dwell_safety=True, dwell_probe=8, preempt=0,
            tp_dummy=False, reeval_interval=40, move_thresh=0.0,
            rest_deplete=0.0, svc_radius=4.0, flow_cost=0.0)

# ---- 泊位策略（L3 层）----
# flow_cost ψ：驻留干扰避让。空闲 agent 停在 pickup→station 主干搬运通道上会
#   拖慢送货段（diag_legs 实测 L3 从 29.4 涨到 33.7，完全吃掉响应段的收益）。
#   ψ 存在内部最优：过小则 L3 失控，过大则白白牺牲 L2。ψ* 随 fleet 从 ~20 微移到 ~40。
POL = {
    # TP 原生行为：停在最近的 non-task endpoint（Ma et al. 2017 的 dummy path 语义）
    "nearest": dict(rest_policy="nearest", demand_mode="none"),
    "random": dict(rest_policy="random", demand_mode="none"),
    "static": dict(rest_policy="static", demand_mode="none"),
    # β：搬迁代价必须和等待代价用同一种货币计（否则大 fleet 下 white whale 式
    # 频繁改位会吞噬预测收益）；0.5 由 diag_move_cost 的扫描确定。
    "memir0": dict(rest_policy="exp", demand_mode="ema", tau_phi3=100,
                   rest_beta=0.5, flow_cost=0.0),
    "memir": dict(rest_policy="exp", demand_mode="ema", tau_phi3=100,
                  rest_beta=0.5, flow_cost=30.0),
    "oracle": dict(rest_policy="oracle", demand_mode="oracle", flow_cost=0.0),
    "oracle_f": dict(rest_policy="oracle", demand_mode="oracle",
                     flow_cost=30.0),
    # ---- 近年 / 对照社区基线（2026-09-26 补）----
    # stay：完全不预定位，检验"预定位这件事本身"是否值得做
    "stay": dict(rest_policy="stay", demand_mode="none"),
    # pmedian：greedy p-median / 设施选址（运筹学社区放置 k 个服务点的金标准）。
    #   用的是**与 MEMIR 完全相同**的需求信息 p̂ + β + ψ，因此是公平对照：
    #   回答"MEMIR 的增益是不是只要把 agent 分散开就能拿到"。
    "pmedian0": dict(rest_policy="pmedian", demand_mode="ema", tau_phi3=100,
                     rest_beta=0.5, flow_cost=0.0),
    "pmedian": dict(rest_policy="pmedian", demand_mode="ema", tau_phi3=100,
                    rest_beta=0.5, flow_cost=30.0),
    # popsample：按局部需求质量比例采样驻留格（AMoD / 网约车社区的需求吸引式
    #   重定位基线，如 Lin et al. KDD 2018 一类的 demand-proportional 策略）
    "popsample": dict(rest_policy="popsample", demand_mode="ema",
                      tau_phi3=100, rest_beta=0.5),
    "popsample_f": dict(rest_policy="popsample", demand_mode="ema",
                        tau_phi3=100, rest_beta=0.5, pop_flow=3.0),
    # ---- 学习类基线：直接输出驻留位置（详见 learn.py）----
    "rl_lin": dict(rest_policy="learn", demand_mode="ema", tau_phi3=100,
                   learn_kind="lin", learn_mode="rl", learn_lr=0.05,
                   learn_temp=0.5, learn_topk=20),
    "rl_lin_g": dict(rest_policy="learn", demand_mode="ema", tau_phi3=100,
                     learn_kind="lin", learn_mode="rl", learn_lr=0.05,
                     learn_temp=0.5, learn_topk=20, learn_greedy=True),
    "rl_tab": dict(rest_policy="learn", demand_mode="ema", tau_phi3=100,
                   learn_kind="tab", learn_mode="rl", learn_lr=0.2,
                   learn_eps=0.10, learn_anchor=16),
    "attn": dict(rest_policy="learn", demand_mode="ema", tau_phi3=100,
                 learn_kind="attn", learn_mode="rl", learn_lr=0.05,
                 learn_temp=0.5, learn_topk=20),
    "attn_g": dict(rest_policy="learn", demand_mode="ema", tau_phi3=100,
                   learn_kind="attn", learn_mode="rl", learn_lr=0.05,
                   learn_temp=0.5, learn_topk=20, learn_greedy=True),
    "attn_bc": dict(rest_policy="learn", demand_mode="ema", tau_phi3=100,
                    learn_kind="attn", learn_mode="bc", learn_lr=0.05,
                    learn_temp=0.5, learn_topk=20),
}


def _base(map_name="W1", **drop):
    """BASE 副本，剔除 map_name 与所有被上层显式覆盖的键。"""
    keys = set(drop) | {"map_name"}
    return {k: v for k, v in BASE.items() if k not in keys}


def _mk(exp, method, seed, n, pol=None, map_name="W1", **extra):
    j = _base(**extra)
    if pol:
        j.update(POL[pol])
    j.update(extra)
    j.update(exp=exp, method=method, seed=seed, n_agents=n,
             map_name=map_name)
    return j


def _run_job(job):
    """子进程：一次完整 run，返回标量行。"""
    j = dict(job)
    map_name = j.pop("map_name", "W1")
    exp = j.pop("exp", "")
    method = j.pop("method", "")
    seed = j.pop("seed", 0)
    oracle_lam = j.pop("oracle_lam", None)
    wh = get_map(map_name)
    cfg = SimConfig(seed=seed, oracle_lam=oracle_lam, **j)
    t0 = time.perf_counter()
    try:
        r = MAPDSim(wh, cfg).run()
    except Exception as e:                      # noqa: BLE001
        return dict(exp=exp, method=method, map=map_name, seed=seed,
                    error=repr(e))
    wall = time.perf_counter() - t0
    row = dict(exp=exp, method=method, map=map_name, seed=seed, error="")
    for k, v in r.items():
        if isinstance(v, (list, tuple, dict)):
            continue
        row[k] = v
    row["demand_cum"] = "|".join("%.6f" % x for x in r.get("demand_cum", ()))
    for k in ("n_agents", "T", "warmup", "zipf_alpha", "lmbda", "tau_phi3",
              "meta_mode", "drift_mode", "drift_param", "rest_set",
              "dwell_safety", "dwell_probe", "preempt", "planner_mode",
              "tp_dummy", "rest_policy", "demand_mode", "rl_mode",
              "rhcr_h", "rhcr_w", "capacity", "rest_deplete", "svc_radius",
              "flow_cost", "rest_beta", "move_thresh", "warmup",
              "pop_flow", "learn_kind", "learn_mode", "learn_greedy",
              "assign_moving", "flow_def", "dwell_fallback", "comm_lag",
              "obs_noise", "dyn_obs", "het_lat", "demand_feedback",
              "flow_seed", "tau_fast", "meta_ratio", "meta_scale",
              "learn_hidden", "haul_probe"):
        row[k] = j.get(k, SimConfig(**{}).__dict__.get(k, ""))
    row["wall_sec"] = wall
    return row


def _oracle_key(row):
    """同一 seesaw 条件（决定任务流的那些参数）下的 oracle 参考组。"""
    return tuple(row.get(q) for q in
                 ("map", "n_agents", "zipf_alpha", "lmbda", "drift_mode",
                  "drift_param", "T", "planner_mode"))


def _job_key(j):
    return tuple(j.get(q, BASE.get(q)) for q in
                 ("map_name", "n_agents", "zipf_alpha", "lmbda", "drift_mode",
                  "drift_param", "T", "planner_mode"))


def _pool(jobs, workers, tag, t0):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 50 == 0:
                print("  [%5d/%5d %6.0fs]" % (i, len(jobs), time.time() - t0),
                      flush=True)
    return out


def run_exp(exp, jobs, workers=12):
    """跑一组 experiment：先跑非 oracle 任务，再用 nearest 的事后分布补跑 oracle。"""
    os.makedirs(OUT, exist_ok=True)
    plain = [j for j in jobs if j.get("demand_mode") != "oracle"]
    orc = [j for j in jobs if j.get("demand_mode") == "oracle"]
    t0 = time.time()
    rows = _pool(plain, workers, exp, t0)
    if orc:
        cum = {}
        for r in rows:
            if r.get("error") or not r.get("demand_cum"):
                continue
            vals = [float(x) for x in r["demand_cum"].split("|")]
            acc = cum.setdefault(_oracle_key(r), [0.0] * len(vals))
            for i, v in enumerate(vals):
                acc[i] += v
        lam = {}
        for k, acc in cum.items():
            tot = sum(acc)
            lam[k] = [x / tot for x in acc] if tot > 0 else None
        todo = []
        for j in orc:
            j2 = dict(j)
            j2["oracle_lam"] = lam.get(_job_key(j))
            todo.append(j2)
        rows += _pool(todo, workers, exp + "/oracle", t0)
    # ★ 出错的 run **绝不能**静默丢弃：此前 N_bench 缺 1/72、O_scale 缺 18/180
    #   都是被这一行吃掉，前者差点让一个错误结论（"ψ 把不显著推成显著"）进论文。
    #   现在成功的照常写 raw_<exp>.csv，失败的另写 raw_<exp>_errors.csv 并大声报警。
    bad = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    path = os.path.join(OUT, "raw_%s.csv" % exp)
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    if bad:
        epath = os.path.join(OUT, "raw_%s_errors.csv" % exp)
        fields = sorted({k for r in bad for k in r})
        with open(epath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in bad:
                w.writerow(r)
        print("!! %s: %d / %d runs FAILED -> %s"
              % (exp, len(bad), len(bad) + len(rows), epath), flush=True)
        for r in bad[:10]:
            print("   %s %s |A|=%s seed=%s : %s"
                  % (r.get("map"), r.get("method"), r.get("n_agents"),
                     r.get("seed"), str(r.get("error"))[:150]), flush=True)
    print("%-12s runs=%4d  %.0fs -> %s" % (exp, len(rows),
                                           time.time() - t0, path), flush=True)
    return rows


def A_main(seeds):
    jobs = []
    # 4 为饱和区（idle 1%），6~40 覆盖从"无从调度"到"覆盖充分"的完整区间
    for n in (4, 6, 8, 10, 12, 16, 24, 32, 40):
        for pol in ("nearest", "random", "static", "memir0", "memir",
                    "oracle", "oracle_f"):
            for s in seeds:
                jobs.append(_mk("A_main", pol, s, n, pol=pol))
    return jobs


def B_liveness(seeds):
    jobs = []
    variants = [
        ("aco_none", dict(dwell_safety=False, preempt=0)),
        ("aco_ter", dict(dwell_safety=True, preempt=0)),
        ("aco_preempt", dict(dwell_safety=False, preempt=2)),
        ("aco_ter_preempt", dict(dwell_safety=True, preempt=2)),
    ]
    for n in (16, 24, 32):
        for nm, vk in variants:
            for s in seeds:
                jobs.append(_mk("B_liveness", nm, s, n, pol="memir", **vk))
    # park 参考：泊位自由度受限（严格 well-formed），与 any+TER 对照
    for n in (8, 16, 24, 32):
        for pol in ("nearest", "static", "memir"):
            for s in seeds:
                jobs.append(_mk("B_liveness", "park_" + pol, s, n, pol=pol,
                                rest_set="park", dwell_safety=False))
    # ψ 守护 × 让行 的大 fleet 交互
    for n in (32, 40):
        for nm, vk in (("memir0_pre", dict(flow_cost=0.0, preempt=2)),
                       ("memir_pre", dict(flow_cost=30.0, preempt=2))):
            for s in seeds:
                jobs.append(_mk("B_liveness", nm, s, n, pol="memir", **vk))
    return jobs


def C_planner(seeds):
    """规划器变体对照：Token Passing 的两种标准形态（Ma et al. 2017）。

    tp        仅把任务分配到空闲 agent，任务路径止于 delivery
    tp_dummy  原始 TP：一次规划 start→pickup→delivery→non-task endpoint 的完整
              dummy path，末端永久占位，以此保证 well-formed

    RHCR（Li et al. 2021）变体已实现但不作为基线：其对「空闲 agent 驻留何处」
    没有规范化的语义，强行套用会在 |A|≥8 时出现驻留占位死锁或残余冲突，
    这本身是一个独立的研究问题，故从本文实验矩阵中移除。
    """
    jobs = []
    for mode, extra in (("tp", dict()),
                        ("tp_dummy", dict(tp_dummy=True))):
        for pol in ("nearest", "static", "memir0", "memir"):
            for n in (8, 16, 32):
                for s in seeds:
                    kw = dict(extra)
                    kw["planner_mode"] = "tp"
                    jobs.append(_mk("C_planner", "%s+%s" % (mode, pol), s, n,
                                    pol=pol, **kw))
    return jobs


def D_tau(seeds):
    jobs = []
    for n in (8, 16):
        for tau in (20, 50, 100, 200, 500, 2000):
            for s in seeds:
                jobs.append(_mk("D_tau", "memir_t%d" % tau, s, n, pol="memir",
                                tau_phi3=tau))
        for s in seeds:
            jobs.append(_mk("D_tau", "memir_meta", s, n, pol="memir",
                            meta_mode=True, tau_fast=100, meta_ratio=8,
                            meta_scale=0.30))
            jobs.append(_mk("D_tau", "static", s, n, pol="static"))
            jobs.append(_mk("D_tau", "memir0", s, n, pol="memir0"))
    return jobs


def E_hetero(seeds):
    """需求异质性 α × 到达强度 λ：memory 何时有用（外生效应的调制度）。"""
    jobs = []
    for alpha in (0.0, 1.0, 2.0, 3.0):
        for lam in (0.04, 0.08, 0.12):
            for n in (8, 16):
                for pol in ("static", "memir0", "memir"):
                    for s in seeds:
                        jobs.append(_mk("E_hetero", pol, s, n, pol=pol,
                                        zipf_alpha=alpha, lmbda=lam))
    return jobs


def F_drift(seeds):
    jobs = []
    for dm, dp in (("static", 2000), ("periodic", 1500), ("shift", 1500)):
        for meth, extra in (
                ("memir_t100", dict(tau_phi3=100)),
                ("memir_t2000", dict(tau_phi3=2000)),
                ("memir_meta", dict(meta_mode=True, tau_fast=100,
                                    meta_ratio=8, meta_scale=0.30)),
                ("nearest", dict(rest_policy="nearest", demand_mode="none")),
                ("oracle", dict(rest_policy="oracle", demand_mode="oracle"))):
            for n in (8, 16):
                for s in seeds:
                    j = _mk("F_drift", meth, s, n, pol="memir",
                            drift_mode=dm, drift_param=dp)
                    j.update(extra)
                    jobs.append(j)
    return jobs


def G_maps(seeds):
    jobs = []
    for mname in ("W2", "W3"):
        for n in (8, 16):
            for pol in ("nearest", "static", "memir0", "memir"):
                for s in seeds:
                    jobs.append(_mk("G_maps", pol, s, n, pol=pol,
                                    map_name=mname))
    return jobs


def H_psi(seeds):
    """驻留干扰避让强度 ψ 的敏感性 + 内部最优（Table / Fig）。"""
    jobs = []
    for n in (6, 8, 16, 32):
        for psi in (0.0, 10.0, 20.0, 30.0, 40.0, 80.0, 160.0):
            for s in seeds:
                jobs.append(_mk("H_psi", "memir_psi%g" % psi, s, n,
                                pol="memir", flow_cost=psi))
        for s in seeds:
            jobs.append(_mk("H_psi", "static", s, n, pol="static"))
    return jobs


def I_flow(seeds):
    """flow 场来源对照：离线拓扑先验 vs 在线经验车流（外生 vs 内生载体）。

    理论预期（N3「载体必须外生」）：emp 场是自己行为的结果，存在自我否定回路，
    因而至多逼近、不能超越 struct 先验。本实验是对该命题的直接检验。
    """
    jobs = []
    for n in (16, 24, 32):
        for mode in ("struct", "emp", "both"):
            for psi in (10.0, 30.0, 80.0):
                for s in seeds:
                    jobs.append(_mk("I_flow", "%s_psi%g" % (mode, psi), s, n,
                                    pol="memir", flow_cost=psi,
                                    flow_mode=mode))
        for s in seeds:
            jobs.append(_mk("I_flow", "static", s, n, pol="static"))
    return jobs


def J_base(seeds):
    """近年 / 对照社区的非学习基线（2026-09-26 新增）。

    回答三个审稿人一定会问的问题：
      Q1 「预定位这件事本身值不值得做？」        -> stay
      Q2 「MEMIR 的增益是不是只是把 agent 分散开？」-> pmedian（greedy 设施选址，
          用**完全相同**的 p̂ + β + ψ，是公平对照）
      Q3 「按需求比例随机吸引（AMoD 社区的标准基线）够不够？」-> popsample
    另加 popsample_f 检验 ψ 项是否对采样式策略同样有效（ψ 的普适性消融）。
    """
    jobs = []
    for n in (8, 12, 16, 24, 32):
        for pol in ("nearest", "static", "pmedian0", "pmedian",
                    "popsample", "popsample_f", "memir0", "memir"):
            for s in seeds:
                jobs.append(_mk("J_base", pol, s, n, pol=pol))
    # stay：只在两个规模上跑。它在 MAPD 里**不可完全实现** —— agent 完成任务后
    # 位于 delivery endpoint，必须让出，于是"能不动就不动"退化成 nearest。
    # 这一退化本身是结论（见 §Table J 注），故只保留最小证据，不铺满扫描。
    for n in (8, 16):
        for s in seeds:
            jobs.append(_mk("J_base", "stay", s, n, pol="stay"))
    return jobs


def K_learn(seeds):
    """学习类 repositioning 基线：RL / AM 直接输出驻留位置。

    lin  线性候选打分器 + REINFORCE with baseline
    tab  上下文 bandit（锚点离散化动作空间）
    attn Attention Model 式指针网络（Kool et al. 2019 的 pointer/glimpse 结构）

    公平性：三者都拿到与 MEMIR 完全相同的 p̂ / 距离场 / flow，并额外拿到
    MEMIR 没有的邻域竞争信息（见 learn.py）。attn_bc 是蒸馏版：以解析式
    1-中位数为教师，检验"学习能否复现解析解"。

    训练协议：在线学习（warmup 期间积累经验，此后继续自适应），
    并用 warmup=2500 的变体给学习者更多训练数据（learn_warm）。
    """
    jobs = []
    for n in (8, 16, 24):
        for pol in ("static", "memir0", "memir", "rl_lin", "rl_lin_g",
                    "rl_tab", "attn", "attn_g", "attn_bc"):
            for s in seeds:
                jobs.append(_mk("K_learn", pol, s, n, pol=pol))
    # 给学习者更多训练数据（测量窗口相应缩短），检验"训练不足"是否为瓶颈
    for n in (8, 16):
        for pol in ("rl_lin", "rl_tab", "attn", "attn_bc"):
            for s in seeds:
                jobs.append(_mk("K_learn", pol + "_warm", s, n, pol=pol,
                                warmup=2500, T=5000))
    return jobs


def L_gen(seeds):
    """学习类基线的地图泛化（2026-09-26 第三轮新增）。

    K_learn 只在 W1 上跑过，学习类刚好在 |A|=8 追平 MEMIR —— 这里检验
    「追平」是否为 W1 拓扑的偶然。W2（类 KIVA 大仓，通道宽）与 W3（小而密，
    通道稀缺）构成两个方向的压力：W3 的拥堵外部性更强，若学习类真的学到了
    驻留外部性，它应当在 W3 上比 W2 掉得少。
    """
    jobs = []
    for mname in ("W2", "W3"):
        for n in (8, 16):
            for pol in ("static", "memir0", "memir", "rl_lin", "rl_lin_g",
                        "rl_tab", "attn", "attn_g", "attn_bc"):
                for s in seeds:
                    jobs.append(_mk("L_gen", pol, s, n, pol=pol,
                                    map_name=mname))
    return jobs


def M_probe(seeds):
    """TER-Safety 守护强度 dwell_probe 的规模效应（2026-09-26 第三轮新增）。

    dwell_probe 限制「可接受驻留目标」的候选个数：越小越保守（宁可停在近处
    也不冒险），越大越接近无约束的 1-中位数。此前所有实验固定 probe=8，
    因此大 fleet 上的增益可能受守护力强约束。本实验回答：probe 变大能否
    进一步解锁大 fleet 的增益，还是会让 |A|≥32 的驻留外部性重新失控。
    """
    jobs = []
    for n in (16, 24, 32, 40):
        for probe in (4, 8, 16, 32, 64):
            for pol in ("memir0", "memir"):
                for s in seeds:
                    jobs.append(_mk("M_probe", "%s_p%d" % (pol, probe), s, n,
                                    pol=pol, dwell_probe=probe))
        for s in seeds:
            jobs.append(_mk("M_probe", "static", s, n, pol="static"))
    return jobs


def N_bench(seeds):
    """MovingAI 标准 benchmark 地图（2026-09-26 第三轮新增）。

    拓扑原样取自 movingai.com/benchmarks/mapf（Stern et al., SoCS 2019）：
        B1 warehouse-10-20-10-2-1   161×63   5,699 states
        B2 warehouse-10-20-10-2-2   170×84   9,776 states
    端口语义由 movingai.py 的固定规则 R1–R6 叠加（所有地图、所有策略共用同一
    规则与同一份候选驻留集）。回答：MEMIR 的结论是不是只在本文的自制地图上成立。

    注意：标准图的单 tick 规划开销远大于 W1（同样的 4 000 步 × 10 seeds × 3 个
    规模不可接受），因此这里把视界缩到 T=2500 / warmup=800 并在论文里明确说明；
    策略之间的比较都在同一视界下配对，结论不受影响。
    """
    jobs = []
    for mname in ("B1", "B2"):
        for n in (16, 32):
            for pol in ("static", "memir0", "memir"):
                for s in seeds:
                    jobs.append(_mk("N_bench", pol, s, n, pol=pol,
                                    map_name=mname, T=2500, warmup=800))
    return jobs


def O_scale(seeds):
    """★ 尺度实验（2026-09-26 第四轮）：「增益是 idle rate 的函数，还是地图尺寸的函数？」

    动机：现有实验只有（小图 × 高密度）与（大图 × 低密度）两个角落，
    缺（大图 × 中高密度）—— 而真实 KIVA 仓正是后者，也是驻留外部性
    （本文核心机制）最该出现的地方。POGEMA 在 Warehouse 33×46 上跑 32–192
    agents（密度 2–13%），我们此前最大的标准图实例只有 32 agents / 0.56%。

    设计：不做"更大的表"，做一个**可证伪的预测**（H-scale）——
        把所有 (地图, |A|, λ) 格子画在 (idle rate, Δ service time) 平面上，
        若不同尺寸的散点落在同一条曲线上，则增益只是 idle rate 的函数；
        若曲线按尺寸分层，则"可调度窗口"的位置依赖空间尺度（同样是新结论）。

    λ 的选法：按 λ = c·|A| 取 c ∈ {0.0025, 0.005}（探针实测：c=0.0025 →
    idle ≈ 65–80%；c=0.005 → idle ≈ 55–68%；c≥0.01 会进入饱和区，
    队列无界增长、service time 失去意义，故不取）。

    视界：合成图 T=2000/warmup=600；标准图 T=1500/warmup=500（单 tick 开销大，
    探针实测 B3/|A|=64 在 T=400 只要 18 s，外推 T=1500 ≈ 70 s，可接受）。
    """
    jobs = []
    # (map, fleet 档, T, warmup)
    GRID = (("W4", (32, 64, 128), 2000, 600),
            ("W5", (64, 128, 256), 2000, 600),
            ("B1", (64, 128), 1500, 500),
            ("B3", (64, 128), 1500, 500))
    for mname, fleets, T, warm in GRID:
        for n in fleets:
            for c in (0.0025, 0.005):
                lam = round(c * n, 4)
                for pol in ("static", "memir0", "memir"):
                    for s in seeds:
                        jobs.append(_mk("O_scale", pol, s, n, pol=pol,
                                        map_name=mname, T=T, warmup=warm,
                                        lmbda=lam))
    return jobs


EXPS = {
    "A_main": A_main,
    "B_liveness": B_liveness,
    "C_planner": C_planner,
    "D_tau": D_tau,
    "E_hetero": E_hetero,
    "F_drift": F_drift,
    "G_maps": G_maps,
    "H_psi": H_psi,
    "I_flow": I_flow,
    "J_base": J_base,
    "K_learn": K_learn,
    "L_gen": L_gen,
    "M_probe": M_probe,
    "N_bench": N_bench,
    "O_scale": O_scale,
}


def main():
    which = sys.argv[1:] or list(EXPS)
    seeds = list(range(10))
    workers = int(os.environ.get("WORKERS", "10"))
    for name in which:
        if name not in EXPS:
            print("skip unknown", name)
            continue
        if name == "O_scale":
            # 尺度实验的每个格子本身很贵（大图 × 大 fleet），3 seeds 起步；
            # 若结果有意思再补到 6 seeds。
            sd = seeds[:3]
        elif name in ("C_planner", "G_maps", "E_hetero", "F_drift", "H_psi",
                      "N_bench"):
            sd = seeds[:6]
        else:
            sd = seeds
        print("=== %s (seeds=%d) ===" % (name, len(sd)), flush=True)
        run_exp(name, EXPS[name](sd), workers=workers)


if __name__ == "__main__":
    main()
