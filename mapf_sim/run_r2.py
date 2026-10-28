"""Reviewer 2 返修专项实验（2026-10-27）。零第三方依赖。

    Q_guard   TER-Safety 兜底策略：min（启发式）vs park（可证明安全）× probe
    Q_flow    flow(c) 的定义敏感性：single / rand×R / any / multi
    Q_meta    元适应遗忘对 (tau_fast, ratio, kappa) 的稳健性 + 固定 τ 全景
    Q_stress  现实性压力：通信延迟 / 归属噪声 / 动态障碍 / 异质节拍 / 需求反馈

产出 results_final/raw_<exp>.csv。跑法：
    python run_r2.py Q_guard Q_flow Q_meta Q_stress
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_suite import (BASE, POL, _mk, run_exp)       # noqa: E402

EXPS = {}


def reg(name):
    def deco(fn):
        EXPS[name] = fn
        return fn
    return deco


@reg("Q_guard")
def Q_guard(seeds):
    """TER-Safety 的两难：探测窗口有限时，如何既不放任又不冻死 agent。

    min  = 既有行为：窗口内取违规最小者（可能 >0 -> 只是启发式过滤器）
    park = 新增兜底：窗口内无 0 违规者时退到最近的 non-task endpoint，
           在 A1（well-formed）下该格**必然**安全 -> 与 probe 大小无关地保住
           well-formedness（Lemma 3 的可执行形式）。代价是要测量的问题。
    """
    jobs = []
    for n in (16, 24, 32, 40):
        for s in seeds:
            jobs.append(_mk("Q_guard", "static", s, n, pol="static"))
            # 既有：min 兜底，probe 8 / 64
            for probe in (8, 64):
                jobs.append(_mk("Q_guard", "min_p%d" % probe, s, n,
                                pol="memir", dwell_fallback="min",
                                dwell_probe=probe))
            # 可证明安全的兜底，probe 从 2 到 64（32 倍范围）
            for probe in (2, 8, 32, 64):
                jobs.append(_mk("Q_guard", "park_p%d" % probe, s, n,
                                pol="memir", dwell_fallback="park",
                                dwell_probe=probe))
    return jobs


@reg("Q_guard2")
def Q_guard2(seeds):
    """Q_guard 的完整版：追加 free 兜底（在**整个候选集**上做完备扫描）。

    min  窗口内取违规最小者（纯启发式）
    park 退到最近的 non-task endpoint（单向安全）
    free 先扫完 non-task endpoint，再按距离扫其余候选格（完备 ⇒ 双向安全）

    用于把 Lemma 4 的逆命题补齐：free 模式下"扫描失败"等价于"候选集中
    不存在 v=0 的格"，此时任何必须返回一个候选格的规则都保证不了 v=0。
    """
    jobs = []
    for n in (16, 24, 32, 40):
        for s in seeds:
            jobs.append(_mk("Q_guard2", "static", s, n, pol="static"))
            for probe in (8, 64):
                jobs.append(_mk("Q_guard2", "min_p%d" % probe, s, n,
                                pol="memir", dwell_fallback="min",
                                dwell_probe=probe))
            for mode in ("park", "free"):
                for probe in (2, 8, 32, 64):
                    jobs.append(_mk("Q_guard2", "%s_p%d" % (mode, probe), s, n,
                                    pol="memir", dwell_fallback=mode,
                                    dwell_probe=probe))
    return jobs


@reg("K_learn24")
def K_learn24(seeds):
    """补齐 K_learn 在 |A|=24 上缺失的 warm 变体。

    K_learn 的 learn_warm 只在 |A|∈{8,16} 上跑过，而 Table (learned) 的
    |A|=24 列却填了 warm 行的数值（与非 warm 行完全相同）——真实数据缺失。
    这里按完全相同的协议补跑 3 个学习者 × 10 seeds，用于更正那一列。
    """
    jobs = []
    for n in (24,):
        for pol in ("rl_lin", "rl_tab", "attn", "attn_bc"):
            for s in seeds:
                jobs.append(_mk("K_learn24", pol + "_warm", s, n, pol=pol,
                                warmup=2500, T=5000))
    return jobs


@reg("Q_learn4")
def Q_learn4(seeds):
    """补齐 Q_learn 缺失的 warm-up 变体（REINFORCE / bandit）。

    Q_learn 只给 attn 系拧了 warmup=2500，而 Table (budget) 的 warm-up 行
    把 REINFORCE linear 也写成 +3.4/+3.4——没有对应 run。这里按同一协议补跑。
    """
    jobs = []
    for n in (8, 16):
        for s in seeds:
            for pol in ("rl_lin", "rl_tab"):
                jobs.append(_mk("Q_learn4", "%s_w" % pol, s, n, pol=pol,
                                warmup=2500))
    return jobs


@reg("Q_flow")
def Q_flow(seeds):
    """flow(c) 的定义敏感性（Reviewer 2 Q2）。

    四种定义共享同一个 ψ 与同一份候选集，差别只在"哪条最短路代表搬运流"：
        single  固定邻居序取**一条**最短路（论文既有定义）
        rand    同上，但等长候选里随机取（3 个 RNG 种子，检验 tie-break 方差）
        any     落在**任一**最短路上即计 1（集合版）
        multi   概率化路径流：按路径多重性 σ_s(c)·σ_w(c)/σ_total 加权，
                即 single/rand 在 tie-break 上的**期望**，与取法无关
    """
    jobs = []
    for n in (16, 24, 32):
        for s in seeds:
            jobs.append(_mk("Q_flow", "static", s, n, pol="static"))
            jobs.append(_mk("Q_flow", "single", s, n, pol="memir",
                            flow_def="single"))
            for r in (1, 2, 3):
                jobs.append(_mk("Q_flow", "rand%d" % r, s, n, pol="memir",
                                flow_def="rand", flow_seed=r))
            jobs.append(_mk("Q_flow", "any", s, n, pol="memir",
                            flow_def="any"))
            jobs.append(_mk("Q_flow", "multi", s, n, pol="memir",
                            flow_def="multi"))
    # multi 定义下 ψ 是否仍有内部最优（防止"换了定义就要重新拟合"）
    # 另开 haul_probe 的一小组，用于落盘"实际载货交通"，算各定义的相关性
    for psi in (0.0, 10.0, 30.0, 80.0):
        for n in (16, 32):
            for s in seeds:
                jobs.append(_mk("Q_flow", "multi_psi%g" % psi, s, n,
                                pol="memir", flow_def="multi",
                                flow_cost=psi))
    for s in seeds[:3]:
        jobs.append(_mk("Q_flow", "probe", s, 24, pol="memir",
                        haul_probe=1))
    return jobs


@reg("Q_meta")
def Q_meta(seeds):
    """元适应遗忘的稳健性（Reviewer 2 Q3）。

    (a) 固定 τ 全景：τ ∈ {25,...,3200}（128 倍范围）× 三种漂移 × 两个规模，
        用来回答"自适应到底输给了什么"——是输给**每个环境各自的最优 τ**
        （不可部署，因为要预知环境），还是输给**随便挑一个 τ**。
    (b) 元适应网格：tau_fast × ratio(=τ_slow/τ_fast) × kappa(=s_scale) 全扫，
        报告最好/最差/四分位，证明它不是靠某个幸运点。
    """
    jobs = []
    # (a) 固定 τ 全景
    for dm, dp in (("static", 2000), ("periodic", 1500), ("shift", 1500)):
        for n in (8, 16):
            for tau in (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0,
                        3200.0):
                for s in seeds:
                    j = _mk("Q_meta", "fix_t%g" % tau, s, n, pol="memir",
                            drift_mode=dm, drift_param=dp, tau_phi3=tau)
                    jobs.append(j)
    # (b) 元适应网格：只跑漂移环境（静态环境下自适应本就应当退化回慢记忆）
    for dm, dp in (("periodic", 1500), ("shift", 1500)):
        for n in (8, 16):
            for tf in (50.0, 100.0, 200.0):
                for ratio in (4.0, 8.0, 16.0):
                    for kap in (0.15, 0.30, 0.60):
                        for s in seeds:
                            jobs.append(_mk(
                                "Q_meta",
                                "meta_f%g_r%g_k%g" % (tf, ratio, kap), s, n,
                                pol="memir", drift_mode=dm, drift_param=dp,
                                meta_mode=True, tau_fast=tf, meta_ratio=ratio,
                                meta_scale=kap))
    return jobs


@reg("Q_stress")
def Q_stress(seeds):
    """超出理想假设的现实性压力（Reviewer 2 Q6）。

    每个压力都只在**信息侧/环境侧**动手，不破坏离散时空规划的不变量：
        comm_lag=L          决策层读到 L 拍之前的到达场
        obs_noise=p         到达被误记到随机区域的概率
        dyn_obs=k           动态障碍物（人/叉车），轨迹全程预约
        het_lat=w           半数 agent 发车前多等 w 拍（异质节拍）
        demand_feedback=ρ   到达区域以概率 ρ 抽自当前空闲 agent 的位置
    """
    # 每个压力都**成对**跑 static 与 memir：扰动本身会改变绝对水平，
    # 要测的是"MEMIR 相对基线的优势"是否还在，而不是绝对服务时间。
    stress = [("base", {}),
              ("lag50", dict(comm_lag=50)),
              ("lag200", dict(comm_lag=200)),
              ("noise0.15", dict(obs_noise=0.15)),
              ("noise0.35", dict(obs_noise=0.35)),
              ("obs2", dict(dyn_obs=2)),
              ("obs6", dict(dyn_obs=6)),
              ("het2", dict(het_lat=2)),
              ("het5", dict(het_lat=5)),
              ("fb0.3", dict(demand_feedback=0.3)),
              ("fb0.6", dict(demand_feedback=0.6))]
    jobs = []
    for n in (16, 32):
        for tag, extra in stress:
            for pol in ("static", "memir"):
                for s in seeds:
                    jobs.append(_mk("Q_stress", "%s_%s" % (pol, tag), s, n,
                                    pol=pol, **extra))
    return jobs


@reg("Q_learn")
def Q_learn(seeds):
    """学习类基线的"预算公平性"压力测试（Reviewer 2 Q5）。

     reviewers 的合理质疑：attn 只有 120 个参数、训练 1000 拍，可能是容量/
     训练量不足而非方法不行。这里把两个旋钮各拧大一档：
        容量    dh: 8 -> 32（参数量 120 -> 1056）
        训练量  warmup 1000 -> 2500（评估窗仍在 warmup 之后，故是"多练少算"）
     并同时记录 learn_upd（梯度更新次数）与 learn_curve（回报基线的收敛轨迹）。
    """
    jobs = []
    for n in (8, 16):
        for s in seeds:
            for pol in ("static", "memir", "rl_lin", "rl_tab", "attn",
                        "attn_bc"):
                jobs.append(_mk("Q_learn", pol, s, n, pol=pol))
            jobs.append(_mk("Q_learn", "attn_h32", s, n, pol="attn",
                            learn_hidden=32))
            for pol in ("static", "memir", "attn"):
                jobs.append(_mk("Q_learn", "%s_w" % pol, s, n, pol=pol,
                                warmup=2500))
            jobs.append(_mk("Q_learn", "attn_h32_w", s, n, pol="attn",
                            learn_hidden=32, warmup=2500))
    return jobs


@reg("Q_learn2")
def Q_learn2(seeds):
    """把训练视野整体拉长 4 倍（T=16000, warmup=12000），评估窗仍为 4000 拍。

    Q_learn 显示"加大容量 / 延长 warmup"都不奏效，一个残留解释是**梯度更新
    次数太少**（主设置下每 run 只有 ~300 次 REINFORCE 更新）。这里把训练段
    拉到 12000 拍，更新次数约 3 倍，检验"再多练练就好了"这一解释。
    """
    jobs = []
    for n in (8, 16):
        for s in seeds:
            for pol in ("static", "memir", "rl_lin", "attn"):
                jobs.append(_mk("Q_learn2", pol, s, n, pol=pol,
                                T=16000, warmup=12000))
            jobs.append(_mk("Q_learn2", "attn_h32", s, n, pol="attn",
                            learn_hidden=32, T=16000, warmup=12000))
    return jobs


def main():
    which = sys.argv[1:] or list(EXPS)
    seeds = list(range(int(os.environ.get("SEEDS", "10"))))
    workers = int(os.environ.get("WORKERS", "10"))
    for name in which:
        if name not in EXPS:
            print("skip unknown", name)
            continue
        print("=== %s ===" % name, flush=True)
        run_exp(name, EXPS[name](seeds), workers=workers)


if __name__ == "__main__":
    main()
