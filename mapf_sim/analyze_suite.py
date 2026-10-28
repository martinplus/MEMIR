"""把 results_final/raw_*.csv 汇总成论文可直接使用的 markdown 表 + figure 数据。

统计量（全部零依赖自实现，见 stats_util）：
    - 配对 bootstrap 95% CI（同一 seed 两两相减后重采样）
    - Wilcoxon 符号秩（正态近似 + 并列修正）
    - Holm 多重比较校正
    - Cohen's d_z 效应量
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats_util import (paired_bootstrap, wilcoxon_signed_rank, holm,  # noqa
                        cohens_dz, pct, mean_ci, paired_t)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_final")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

NUM = ("service_post", "throughput_post", "completed_post", "idle_ratio",
       "resp_dist", "plan_attempt_fail", "plan_fail", "preempt_yields",
       "dwell_checks", "dwell_blocked", "dwell_viol", "decide_cnt",
       "decide_ns", "wall_sec", "conflicts", "rest_moves", "arrivals",
       "completed", "wait_ratio",
       # L1/L2/L3 延迟分解（diag_legs）
       "leg_assign", "leg_approach", "leg_haul", "no_idle")


def load(exp):
    p = os.path.join(OUT, "raw_%s.csv" % exp)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in NUM:
                if k in d and d[k] not in ("", None):
                    try:
                        d[k] = float(d[k])
                    except ValueError:
                        pass
            for k in ("n_agents", "seed", "T", "warmup"):
                if d.get(k) not in ("", None):
                    try:
                        d[k] = int(float(d[k]))
                    except ValueError:
                        pass
            for k in ("zipf_alpha", "lmbda", "tau_phi3", "drift_param"):
                if d.get(k) not in ("", None):
                    try:
                        d[k] = float(d[k])
                    except ValueError:
                        pass
            if d.get("service_post", -1) is None or d.get("service_post", -1) == -1:
                continue
            rows.append(d)
    return rows


def pick(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if str(r.get(k)) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return sorted(out, key=lambda r: r["seed"])


def series(rows, metric="service_post"):
    return [r[metric] for r in rows]


def compare(base_rows, new_rows, metric="service_post", label=""):
    """配对比较 new vs base。返回 dict（负 = new 更好）。"""
    bs = {r["seed"]: r[metric] for r in base_rows}
    ns = {r["seed"]: r[metric] for r in new_rows}
    seeds = sorted(set(bs) & set(ns))
    if not seeds:
        return None
    d = [ns[s] - bs[s] for s in seeds]
    m, lo, hi, pboot = paired_bootstrap(d)
    w, z, pwil = wilcoxon_signed_rank(d)
    tv, tdf, tp, _tm, _tsd = paired_t(d)
    bm = sum(bs[s] for s in seeds) / len(seeds)
    nm = sum(ns[s] for s in seeds) / len(seeds)
    return dict(label=label, n_pairs=len(seeds), base=bm, new=nm,
                diff=m, ci_lo=lo, ci_hi=hi, pct=pct(nm, bm),
                wilcoxon_p=pwil, dz=cohens_dz(d),
                t=tv, t_df=tdf, t_p=tp, diff_sd=_tsd,
                base_sd=_sd([bs[s] for s in seeds]),
                new_sd=_sd([ns[s] for s in seeds]))


def _sd(x):
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    return (sum((v - m) ** 2 for v in x) / (n - 1)) ** 0.5


def fmt_cmp(c, metric_unit=""):
    if c is None:
        return "n/a"
    star = _star(c)
    p = _p_used(c)
    tag = "W" if c.get("n_pairs", 99) >= 6 else "t"
    df = c.get("t_df", 0)
    tail = ("%s p=%.3g" % (tag, p)) if tag == "W" \
        else ("t(%d)=%.2f, p=%.3g" % (df, c.get("t", 0.0), p))
    return ("%+.2f%% (Δ=%+.2f, 95%%CI [%+.2f,%+.2f], %s %s, d_z=%.2f)"
            % (c["pct"], c["diff"], c["ci_lo"], c["ci_hi"],
               tail, star, c["dz"]))


def _avg(rr, k, default=0.0):
    v = [r.get(k, default) for r in rr if r.get(k, -1) is not None]
    v = [x for x in v if x is not None and x != ""]
    v = [float(x) for x in v]
    v = [x for x in v if x >= 0] or [default]
    return sum(v) / len(v)


def table_main(rows):
    L = ["## Table A  主对照（W1, λ=0.08, α=2.0, T=4000, warmup=1000, 10 seeds）",
         "",
         "service time 是「任务到达 → 送达」的平均驻留时间；"
         "L1/L2/L3 是其三段分解：等派单 / 到位取货 / 取货到入站。",
         "",
         "| |A| | policy | service time (mean±sd) | Δ vs static | 统计量 | "
         "L1 | L2(resp) | L3(haul) | idle% | ms/dec |",
         "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    ns = sorted({r["n_agents"] for r in rows})
    fig = []
    pols = ["nearest", "random", "static", "memir0", "memir", "oracle",
            "oracle_f"]
    for n in ns:
        base = pick(rows, n_agents=n, method="static", map="W1")
        if not base:
            continue
        row = {"n": n}
        for pol in pols:
            rr = pick(rows, n_agents=n, method=pol, map="W1")
            if not rr:
                continue
            md = compare(base, rr, "service_post", pol)
            idle = _avg(rr, "idle_ratio") * 100
            msd = sum(r.get("decide_ns", 0) for r in rr) / 1e6 / \
                max(sum(r.get("decide_cnt", 0) for r in rr), 1)
            if pol == "static":
                stat = "—（对照）"
            else:
                stat = fmt_cmp(md)
            L.append("| %d | %s | %.2f ± %.2f | %+.2f%% | %s | %.2f | %.2f | "
                     "%.2f | %.1f | %.3f |"
                     % (n, pol, md["new"], md["new_sd"], md["pct"], stat,
                        _avg(rr, "leg_assign"), _avg(rr, "leg_approach"),
                        _avg(rr, "leg_haul"), idle, msd))
            if pol in ("nearest", "static", "memir0", "memir", "oracle"):
                row[pol] = md["new"]
            if pol in ("memir", "memir0"):
                row.setdefault("L3_" + pol, _avg(rr, "leg_haul"))
        fig.append(row)
        L.append("")
    with open(os.path.join(FIG, "figA_fleet.json"), "w", encoding="utf-8") as f:
        json.dump(fig, f, indent=1)
    return L


def table_liveness(rows):
    L = ["## Table B  liveness 与驻留干扰消融（W1, λ=0.08, α=2.0）", "",
         "aco_* = rest_set=any（含通道，泊位自由度大）；park_* = 仅 non-task endpoint。",
         "",
         "| |A| | variant | service time | Δ vs aco_none | plan-attempt-fail | "
         "preempt yields | dwell blocked | L3(haul) | ms/dec | wall s |",
         "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="aco_none")
        meths = sorted({r["method"] for r in rows if r["n_agents"] == n})
        for meth in meths:
            rr = pick(rows, n_agents=n, method=meth)
            if not rr:
                continue
            ans = compare(base, rr, "service_post", meth) if base else None
            pf = _avg(rr, "plan_attempt_fail")
            yl = _avg(rr, "preempt_yields")
            db = _avg(rr, "dwell_blocked")
            l3 = _avg(rr, "leg_haul")
            msd = sum(r.get("decide_ns", 0) for r in rr) / 1e6 / \
                max(sum(r.get("decide_cnt", 0) for r in rr), 1)
            wl = _avg(rr, "wall_sec")
            if ans is None or meth == "aco_none":
                L.append("| %d | %s | %.2f ± %.2f | — | %.0f | %.0f | %.0f | "
                         "%.2f | %.3f | %.1f |"
                         % (n, meth, sum(series(rr)) / len(rr),
                            _sd(series(rr)), pf, yl, db, l3, msd, wl))
            else:
                L.append("| %d | %s | %.2f ± %.2f | %s | %.0f | %.0f | %.0f | "
                         "%.2f | %.3f | %.1f |"
                         % (n, meth, ans["new"], ans["new_sd"], fmt_cmp(ans),
                            pf, yl, db, l3, msd, wl))
        L.append("")
    return L


def table_planner(rows):
    L = ["## Table C  规划器: tp / tp+dummy / rhcr（Li et al. 2021 窗口滚动）", "",
         "| |A| | planner+policy | service time | Δ vs tp+static | conflicts | "
         "ms/dec |",
         "|---:|---|---:|---|---:|---:|---:|"]
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="tp+static")
        for meth in sorted({r["method"] for r in rows if r["n_agents"] == n}):
            rr = pick(rows, n_agents=n, method=meth)
            if not rr:
                continue
            ans = compare(base, rr, "service_post", meth) if base else None
            cf = _avg(rr, "conflicts")
            msd = sum(r.get("decide_ns", 0) for r in rr) / 1e6 / \
                max(sum(r.get("decide_cnt", 0) for r in rr), 1)
            if ans is None or meth == "tp+static":
                L.append("| %d | %s | %.2f ± %.2f | — | %.0f | %.3f |"
                         % (n, meth, sum(series(rr)) / len(rr),
                            _sd(series(rr)), cf, msd))
            else:
                L.append("| %d | %s | %.2f ± %.2f | %s | %.0f | %.3f |"
                         % (n, meth, ans["new"], ans["new_sd"], fmt_cmp(ans),
                            cf, msd))
        L.append("")
    return L


def table_tau(rows):
    L = ["## Table D  记忆时间常数 τ 敏感性（含元适应 RL）", "",
         "| |A| | method | service time | Δ vs nearest |",
         "|---:|---|---:|---|---:|"]
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="nearest")
        order = ["nearest"] + sorted(
            (m for m in {r["method"] for r in rows if r["n_agents"] == n}
             if m != "nearest"),
            key=lambda m: (m != "memir_meta",
                           float(m.split("_t")[-1]) if "_t" in m else 1e18))
        figrow = {"n": n}
        for meth in order:
            rr = pick(rows, n_agents=n, method=meth)
            if not rr:
                continue
            if base and meth != "nearest":
                ans = compare(base, rr, "service_post", meth)
                L.append("| %d | %s | %.2f ± %.2f | %s |"
                         % (n, meth, ans["new"], ans["new_sd"], fmt_cmp(ans)))
                figrow[meth] = ans["new"]
            else:
                v = series(rr)
                L.append("| %d | %s | %.2f ± %.2f | — |"
                         % (n, meth, sum(v) / len(v), _sd(v)))
                figrow[meth] = sum(v) / len(v)
        L.append("")
        figrow["_n"] = n
        with open(os.path.join(FIG, "figD_tau_%d.json" % n), "w",
                  encoding="utf-8") as f:
            json.dump(figrow, f, indent=1)
    return L


def table_factor(rows, title, factors, metric="service_post",
                 legs=False, base_name="static"):
    """通用因子表。对照基准为 base_name（缺省 static；缺则退回第一个 method）。"""
    extra = "| L1 | L2 | L3 |" if legs else "|"
    L = ["## " + title, "",
         "| " + " | ".join(factors) + " | policy | service time | Δ vs %s | "
         "idle%% |%s" % (base_name, extra),
         "|---" * len(factors) + "|---|---:|---|---:|---:|"
         + ("---|" * 3 if legs else "")]
    combos = sorted({tuple(str(r.get(f)) for f in factors) for r in rows})
    for combo in combos:
        sub = [r for r in rows
               if tuple(str(r.get(f)) for f in factors) == combo]
        base = pick(sub, method=base_name)
        if not base:
            first = sorted({r["method"] for r in sub})[0]
            base = pick(sub, method=first)
        for meth in sorted({r["method"] for r in sub}):
            rr = pick(sub, method=meth)
            if not rr:
                continue
            v = series(rr, metric)
            idle = _avg(rr, "idle_ratio") * 100
            lg = "| %.2f | %.2f | %.2f |" % (_avg(rr, "leg_assign"),
                                             _avg(rr, "leg_approach"),
                                             _avg(rr, "leg_haul")) \
                if legs else "|"
            if base and meth != base[0]["method"]:
                ans = compare(base, rr, metric, meth)
                L.append("| " + " | ".join(combo) + " | %s | %.2f ± %.2f | %s | "
                         "%.1f %s" % (meth, ans["new"], ans["new_sd"],
                                      fmt_cmp(ans), idle, lg))
            else:
                L.append("| " + " | ".join(combo) + " | %s | %.2f ± %.2f | — | "
                         "%.1f %s" % (meth, sum(v) / len(v), _sd(v), idle, lg))
        L.append("")
    return L


def table_psi(rows):
    """ψ（驻留干扰避让强度）敏感性 —— 存在内部最优 ψ*，且 ψ* 随 |A| 微移。"""
    L = ["## Table H  驻留干扰避让强度 ψ 的敏感性（W1, λ=0.08, α=2.0）", "",
         "| |A| | ψ | service time | Δ vs static | L2(resp) | L3(haul) | "
         "resp_dist |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fig = []
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="static")
        row = {"n": n}
        order = sorted({r["method"] for r in rows if r["n_agents"] == n},
                       key=lambda m: (m != "static",
                                      float(m.replace("memir_psi", ""))
                                      if "psi" in m else -1))
        for meth in order:
            rr = pick(rows, n_agents=n, method=meth)
            if not rr:
                continue
            psi = meth.replace("memir_psi", "") if "psi" in meth else "—"
            v = series(rr)
            mv = sum(v) / len(v)
            stat = fmt_cmp(compare(base, rr, "service_post", meth)) \
                if base and meth != "static" else "—（对照）"
            L.append("| %d | %s | %.2f ± %.2f | %s | %.2f | %.2f | %.2f |"
                     % (n, psi, mv, _sd(v), stat, _avg(rr, "leg_approach"),
                        _avg(rr, "leg_haul"), _avg(rr, "resp_dist")))
            row[psi] = mv
        fig.append(row)
        L.append("")
    with open(os.path.join(FIG, "figH_psi.json"), "w", encoding="utf-8") as f:
        json.dump(fig, f, indent=1)
    return L


def table_flow(rows):
    """flow 场来源：外生离线先验 struct vs 内生在线车流 emp vs 两者合并 both。

    检验 N3「载体必须外生」：emp 场是自己行为的结果（Braverman 式自我否定回路），
    预期至多逼近、不能超越 struct。
    """
    L = ["## Table I  flow 场来源对照：外生的离线先验 vs 内生的在线车流", "",
         "struct = 离线拓扑先验 `build_haul_load`（不依赖策略自身行为）；"
         "emp = 在线 EMA 观测的真实载货车流（τ_flow=200）；both = 两者相加。",
         "",
         "| |A| | mode | ψ | service time | Δ vs static | L2 | L3 | wait% | "
         "ms/dec |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    MODES = ("struct", "emp", "both")
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="static")
        L.append("| %d | — | — | %.2f ± %.2f | —（对照） | %.2f | %.2f | "
                 "%.3f | %.3f |"
                 % (n, sum(series(base)) / len(base), _sd(series(base)),
                    _avg(base, "leg_approach"), _avg(base, "leg_haul"),
                    _avg(base, "wait_ratio") * 100, _msdec(base)))
        for mode in MODES:
            for psi in (10, 30, 80):
                meth = "%s_psi%g" % (mode, psi)
                rr = pick(rows, n_agents=n, method=meth)
                if not rr:
                    continue
                c = compare(base, rr, "service_post", meth)
                star = "***" if c["wilcoxon_p"] < 0.001 else \
                    "**" if c["wilcoxon_p"] < 0.01 else \
                    "*" if c["wilcoxon_p"] < 0.05 else "n.s."
                L.append("| %d | %s | %g | %.2f ± %.2f | %+.2f%% %s | %.2f | "
                         "%.2f | %.3f | %.3f |"
                         % (n, mode, psi, c["new"], c["new_sd"], c["pct"],
                            star, _avg(rr, "leg_approach"),
                            _avg(rr, "leg_haul"), _avg(rr, "wait_ratio") * 100,
                            _msdec(rr)))
        L.append("")
    return L


def _p_used(c):
    """实际用于判显著的 p 值。

    ★ Wilcoxon 符号秩在 n<6 时数学上不可能达到 p<0.05（精确检验 n=3 的最小
      双侧 p = 0.25，n=5 为 0.0625；本文的正态近似在 n=3 时最小 ≈ 0.18）。
      因此 n_pairs<6 时改用配对 t 检验，否则整张表会无意义地全部标记为 n.s.。
    """
    if c is None:
        return 1.0
    return c["wilcoxon_p"] if c.get("n_pairs", 99) >= 6 else c.get("t_p", 1.0)


def _star(c):
    if c is None:
        return "n/a"
    p = _p_used(c)
    return "***" if p < 0.001 else \
        "**" if p < 0.01 else \
        "*" if p < 0.05 else "n.s."


def table_base(rows):
    """Table J：近年 / 对照社区的非学习基线。

    三个审稿人必问的问题各自对应一行：
      stay      预定位这件事本身值不值得做（在 MAPD 中会退化为 nearest）
      pmedian   MEMIR 的增益是不是"只要把 agent 分散开"就能拿到
                （用**完全相同**的 p̂ + β + ψ 的贪心设施选址）
      popsample 按需求比例随机吸引（AMoD 社区的标准基线）够不够
    """
    L = ["## Table J  近年 / 对照社区的非学习基线（W1, λ=0.08, α=2.0）", "",
         "pmedian = 贪心 p-median / 设施选址，使用与 MEMIR **完全相同**的 p̂、β、ψ，"
         "故是公平对照；pmedian0 = ψ=0 的剥离版。",
         "popsample = 按局部需求质量比例采样驻留格（AMoD 的需求吸引式重定位）；"
         "popsample_f 额外乘上 exp(−3·flow(c))，用于检验 ψ 是否对采样式策略同样有效。",
         "",
         "| |A| | policy | service time (mean±sd) | Δ vs static | "
         "L2(resp) | L3(haul) | resp_dist | ms/dec |",
         "|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    POLS = ["nearest", "stay", "static", "pmedian0", "pmedian", "popsample",
            "popsample_f", "memir0", "memir"]
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="static")
        if not base:
            continue
        bm = sum(series(base)) / len(base)
        L.append("| %d | static | %.2f ± %.2f | —（对照） | %.2f | %.2f | "
                 "%.2f | %.3f |"
                 % (n, bm, _sd(series(base)), _avg(base, "leg_approach"),
                    _avg(base, "leg_haul"), _avg(base, "resp_dist"),
                    _msdec(base)))
        for pol in POLS:
            if pol == "static":
                continue
            rr = pick(rows, n_agents=n, method=pol)
            if not rr:
                continue
            c = compare(base, rr, "service_post", pol)
            L.append("| %d | %s | %.2f ± %.2f | %+.2f%% %s | %.2f | %.2f | "
                     "%.2f | %.3f |"
                     % (n, pol, c["new"], c["new_sd"], c["pct"], _star(c),
                        _avg(rr, "leg_approach"), _avg(rr, "leg_haul"),
                        _avg(rr, "resp_dist"), _msdec(rr)))
        L.append("")
    return L


def table_learn(rows):
    """Table K：学习类 repositioning 基线（RL / AM 直接输出驻留位置）。

    所有学习类策略都拿到与 MEMIR 完全相同的 p̂ / 距离场 / flow，
    并额外拿到邻域竞争信息（见 learn.py），因此对照对 MEMIR 偏严。
    attn_bc 是蒸馏版：以解析式 1-中位数为教师。
    *_warm 给学习者更长的训练窗口（warmup 2500 / T 5000）。
    """
    L = ["## Table K  学习类基线：RL / AM 直接输出驻留位置", "",
         "lin = 线性候选打分器 + REINFORCE；tab = 上下文 bandit（锚点离散动作）；"
         "attn = Attention Model 式指针网络；attn_bc = 蒸馏解析式 1-中位数。",
         "全部为**在线学习**（warmup 期间积累经验，此后继续自适应）。"
         "公平性见 learn.py：输入特征含 MEMIR 的全部信息且更多。",
         "",
         "| |A| | policy | service time (mean±sd) | Δ vs static | Δ vs memir | "
         "L2 | L3 | ms/dec |",
         "|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    POLS = ["memir0", "memir", "rl_lin", "rl_lin_g", "rl_tab", "attn",
            "attn_g", "attn_bc", "rl_lin_warm", "rl_tab_warm", "attn_warm",
            "attn_bc_warm"]
    for n in sorted({r["n_agents"] for r in rows}):
        base = pick(rows, n_agents=n, method="static")
        mem = pick(rows, n_agents=n, method="memir")
        if not base:
            continue
        bm = sum(series(base)) / len(base)
        L.append("| %d | static | %.2f ± %.2f | —（对照） | — | %.2f | %.2f | "
                 "%.3f |"
                 % (n, bm, _sd(series(base)), _avg(base, "leg_approach"),
                    _avg(base, "leg_haul"), _msdec(base)))
        for pol in POLS:
            rr = pick(rows, n_agents=n, method=pol)
            if not rr:
                continue
            c = compare(base, rr, "service_post", pol)
            cm = compare(mem, rr, "service_post", pol) if mem else None
            vs = ("%+.2f%% %s" % (cm["pct"], _star(cm))) if cm else "—"
            L.append("| %d | %s | %.2f ± %.2f | %+.2f%% %s | %s | %.2f | "
                     "%.2f | %.3f |"
                     % (n, pol, c["new"], c["new_sd"], c["pct"], _star(c), vs,
                        _avg(rr, "leg_approach"), _avg(rr, "leg_haul"),
                        _msdec(rr)))
        L.append("")
    return L


def table_gen(rows):
    """Table L：学习类基线在**未见过的地图**上是否还能追上 MEMIR。

    K_learn 只在 W1 上比较过，且恰好在 |A|=8 追平。若"追平"来自 W1 的拓扑
    特殊性，那么在拓扑截然不同的 W2 / W3 上应当分离。W3 通道稀缺、驻留外部性
    更强 —— 若学习类真的学到了这个外部性，它在 W3 上应比在 W2 上掉得少。
    """
    L = ["## Table L  学习类基线的地图泛化（W2 / W3，型号与 W1 完全相同）", "",
         "Δ 相对同一地图同一 |A| 的 static；最后一列是相对 MEMIR 的差值，"
         "负值表示学习类更差（MEMIR 更好）。",
         "",
         "| map | |A| | policy | service time | Δ vs static | Δ vs memir | "
         "L2 | L3 | ms/dec |",
         "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    POLS = ["memir0", "memir", "rl_lin", "rl_lin_g", "rl_tab", "attn",
            "attn_g", "attn_bc"]
    for m in sorted({r["map"] for r in rows}):
        sub = [r for r in rows if r["map"] == m]
        for n in sorted({r["n_agents"] for r in sub}):
            base = pick(sub, n_agents=n, method="static")
            mem = pick(sub, n_agents=n, method="memir")
            if not base:
                continue
            bm = sum(series(base)) / len(base)
            L.append("| %s | %d | static | %.2f ± %.2f | —（对照） | — | "
                     "%.2f | %.2f | %.3f |"
                     % (m, n, bm, _sd(series(base)),
                        _avg(base, "leg_approach"), _avg(base, "leg_haul"),
                        _msdec(base)))
            for pol in POLS:
                rr = pick(sub, n_agents=n, method=pol)
                if not rr:
                    continue
                c = compare(base, rr, "service_post", pol)
                cm = compare(mem, rr, "service_post", pol) if mem else None
                vs = ("%+.2f%% %s" % (cm["pct"], _star(cm))) if cm else "—"
                L.append("| %s | %d | %s | %.2f ± %.2f | %+.2f%% %s | %s | "
                         "%.2f | %.2f | %.3f |"
                         % (m, n, pol, c["new"], c["new_sd"], c["pct"],
                            _star(c), vs, _avg(rr, "leg_approach"),
                            _avg(rr, "leg_haul"), _msdec(rr)))
            L.append("")
    return L


def table_probe(rows):
    """Table M：TER-Safety 守护强度 dwell_probe 的规模效应。

    probe 限制「可接受驻留目标」的候选个数：小 = 保守（宁近勿险），
    大 = 更接近无约束 1-中位数。此前所有实验固定 probe=8，因此大 fleet 的
    增益可能被守护强度本身锁住。本表回答：放宽 probe 能解锁多少，
    代价是不是又把 |A|≥32 的驻留外部性放回来（看 L3 列）。
    """
    L = ["## Table M  TER-Safety 守护强度 dwell_probe 的规模效应（W1）", "",
         "probe = 允许的驻留候选个数。最后一列给出相对 probe=8（论文默认）的"
         "配对差值，负值表示放宽后更小。",
         "",
         "| |A| | policy | probe | service time | Δ vs static | Δ vs probe=8 | "
         "L2 | L3 | plan_fail |",
         "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in sorted({r["n_agents"] for r in rows}):
        sub = [r for r in rows if r["n_agents"] == n]
        base = pick(sub, method="static")
        if not base:
            continue
        bm = sum(series(base)) / len(base)
        L.append("| %d | static | — | %.2f ± %.2f | —（对照） | — | %.2f | "
                 "%.2f | %.0f |"
                 % (n, bm, _sd(series(base)), _avg(base, "leg_approach"),
                    _avg(base, "leg_haul"), _avg(base, "plan_fail")))
        for pol in ("memir0", "memir"):
            ref = pick(sub, method="%s_p8" % pol)
            for probe in sorted({r.get("dwell_probe") for r in sub
                                 if str(r.get("method", "")).startswith(pol)
                                 and "_p" in str(r.get("method", ""))},
                                key=lambda x: int(float(x))):
                meth = "%s_p%d" % (pol, int(float(probe)))
                rr = pick(sub, method=meth)
                if not rr:
                    continue
                c = compare(base, rr, "service_post", meth)
                cp = compare(ref, rr, "service_post", meth) if ref else None
                vs = ("%+.2f%% %s" % (cp["pct"], _star(cp))) if cp else "—"
                L.append("| %d | %s | %d | %.2f ± %.2f | %+.2f%% %s | %s | "
                         "%.2f | %.2f | %.0f |"
                         % (n, pol, int(float(probe)), c["new"], c["new_sd"],
                            c["pct"], _star(c), vs, _avg(rr, "leg_approach"),
                            _avg(rr, "leg_haul"), _avg(rr, "plan_fail")))
        L.append("")
    return L


def table_bench(rows):
    """Table N：MovingAI 标准 benchmark 地图上的结果（移植性检验）。"""
    L = ["## Table N  MovingAI 标准 MAPF benchmark 地图上的泛化", "",
         "地图来源 movingai.com/benchmarks/mapf（Stern et al., SoCS 2019），"
         "原样使用其拓扑；端口语义由 movingai.py 的固定规则 R1–R6 叠加，"
         "所有被测策略共用同一份候选驻留集。",
         "",
         "| map | |A| | policy | service time | Δ vs static | idle% | "
         "L2 | L3 | ms/dec |",
         "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    POLS = ["nearest", "memir0", "memir"]
    for m in sorted({r["map"] for r in rows}):
        sub = [r for r in rows if r["map"] == m]
        for n in sorted({r["n_agents"] for r in sub}):
            base = pick(sub, n_agents=n, method="static")
            if not base:
                continue
            bm = sum(series(base)) / len(base)
            L.append("| %s | %d | static | %.2f ± %.2f | —（对照） | %.1f | "
                     "%.2f | %.2f | %.3f |"
                     % (m, n, bm, _sd(series(base)),
                        _avg(base, "idle_ratio") * 100,
                        _avg(base, "leg_approach"), _avg(base, "leg_haul"),
                        _msdec(base)))
            for pol in POLS:
                rr = pick(sub, n_agents=n, method=pol)
                if not rr:
                    continue
                c = compare(base, rr, "service_post", pol)
                L.append("| %s | %d | %s | %.2f ± %.2f | %+.2f%% %s | %.1f | "
                         "%.2f | %.2f | %.3f |"
                         % (m, n, pol, c["new"], c["new_sd"], c["pct"],
                            _star(c), _avg(rr, "idle_ratio") * 100,
                            _avg(rr, "leg_approach"), _avg(rr, "leg_haul"),
                            _msdec(rr)))
            L.append("")
    return L


def _msdec(rr):
    return (sum(r.get("decide_ns", 0) for r in rr) / 1e6
            / max(sum(r.get("decide_cnt", 0) for r in rr), 1))


def table_scale(rows):
    """Table O：尺度实验 —— 增益到底是 idle rate 的函数还是地图尺寸的函数。

    每一行是一个 (地图, |A|, λ) 格子：给出 agent 密度、实测 idle rate、
    memir0 / memir 相对 static 的增益。H-scale 的检验方式是把所有行画在
    (idle rate, Δ) 平面上：若不同尺寸的散点落在同一条曲线上 → 增益只是
    idle rate 的函数；若按尺寸分层 → 可调度窗口的位置依赖空间尺度。
    """
    L = ["## Table O  尺度实验：地图尺寸 × fleet × 到达强度", "",
         "λ 按 λ = c·|A| 取 c ∈ {0.0025, 0.005}（探针标定：c=0.0025 → idle ≈ "
         "65–80%，c=0.005 → idle ≈ 55–68%，c ≥ 0.01 进入饱和区不取）。",
         "密度 = |A| / free cells；POGEMA 的 Warehouse 33×46 用 32–192 agents，"
         "密度 2–13%，可作对照。", "",
         "★ 统计口径：Wilcoxon 符号秩在 n=3 时**数学上不可能**达到 p<0.05"
         "（精确检验最小双侧 p=0.25，本文正态近似最小 ≈0.18），故 n<6 的组"
         "改用**配对 t 检验**判显著，标记 \\*/\\*\\*/\\*\\*\\* 同其他表。"
         "既有 18 格已补到 6 seeds；**大 fleet 角（B1/\\|A\\|=256、W6/128、W6/256）"
         "只跑了 static 与 memir**（memir0 是 ψ 消融，在 \\|A\\|≤128 已充分验证），"
         "故 memir0 列记为 —。", "",
         "★ 24 格中 **22 格** memir 优于 static（符号检验 p≈3.8e-5）。"
         "另 2 格为**反转**（B1/\\|A\\|=128 λ=0.32 与 B1/\\|A\\|=256 λ=0.64），"
         "已定位到派单排队段，见 Table P 与 §6.8.3。", "",
         "| map | 尺寸 / free | \\|A\\| | 密度 | λ | static | memir0 | **memir** "
         "| idle(memir) | L2 → L2 | L3 → L3 | ms/dec |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    FREE = {"W1": 711, "W2": 1230, "W3": 213, "W4": 2551, "W5": 5511,
            "W6": 9591, "B1": 5699, "B2": 9776, "B3": 22599, "B4": 30000}
    for m in sorted({r["map"] for r in rows}):
        sub = [r for r in rows if r["map"] == m]
        for n in sorted({int(r["n_agents"]) for r in sub}):
            for lam in sorted({float(r["lmbda"]) for r in sub
                               if int(r["n_agents"]) == n}):
                cell = [r for r in sub if int(r["n_agents"]) == n
                        and abs(float(r["lmbda"]) - lam) < 1e-9]
                base = pick(cell, method="static")
                if not base:
                    continue
                bm = sum(series(base)) / len(base)
                r0 = pick(cell, method="memir0")
                r1 = pick(cell, method="memir")
                c0 = compare(base, r0, "service_post", "memir0") if r0 else None
                c1 = compare(base, r1, "service_post", "memir") if r1 else None
                if c1 is None:
                    continue
                # 大 fleet 角（B1/256、W6/*）为节省开销只跑了 static 与 memir，
                # 没有 memir0 —— 该列留 "—" 而不是让表崩掉。
                col0 = ("%+.2f%% %s" % (c0["pct"], _star(c0))) if c0 else "—"
                free = FREE.get(m, 0)
                L.append(
                    "| %s | %s | %d | %.2f%% | %.3f | %.2f | %s | "
                    "**%+.2f%%** %s | %.1f%% | %.1f → %.1f | %.1f → %.1f | %.2f |"
                    % (m, "%s/%d" % (m, free) if free else m, n,
                       100.0 * n / free if free else float("nan"), lam, bm,
                       col0, c1["pct"], _star(c1),
                       _avg(r1, "idle_ratio") * 100,
                       _avg(base, "leg_approach"),
                       _avg(r1, "leg_approach"),
                       _avg(base, "leg_haul"), _avg(r1, "leg_haul"),
                       _msdec(r1)))
            L.append("")
    return L


def table_pavail(rows):
    """Table P：大 fleet 反转的归因 —— 逐条排除候选解释。

    O_scale 的 24 格里有 2 格 memir 反而劣于 static（B1/|A|=128 λ=0.32 与
    B1/|A|=256 λ=0.64），都属于「极大 fleet + 极高空闲率」。本表给出三段归因
    与三个被证伪的候选解释：
      ① 规划失败级联 —— 证伪：plan_fail = 0（24 格全为 0）
      ② 移库途中不可派单（assign_moving=0）—— 证伪：放宽后增益变化 < 1 pp
      ③ 空闲 agent 停在 task endpoint 上 —— 证伪：rest 候选集 ∩ task endpoint = ∅
    真正的相关量是 plan_attempt_fail（任务路径规划尝试失败次数）：
    它在 B1 的 6 格内与增益**完全单调**。
    """
    import collections
    L = ["## Table P  大 fleet 反转的归因（P_avail，30 runs）", "",
         "反转格 = B1 / \\|A\\|∈{128, 256} / idle ≥ 74%。对照的 am=0 行取自 O_scale，",
         "am=1 行取自 P_avail（`SimConfig.assign_moving=1`：移库途中的空闲 agent 也允许被派单）。",
         "两者用 common random numbers，可配对。", "",
         "| map | \\|A\\| | λ | am=0 增益 | am=1 增益 | 判定 | "
         "am=0 paf(memir) | am=1 paf(memir) |",
         "|---|---:|---:|---:|---:|---|---:|---:|"]
    d0 = collections.defaultdict(lambda: collections.defaultdict(list))
    d1 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        tgt = d1 if str(r.get("assign_moving")) in ("1", "1.0") else d0
        tgt[(r["map"], r["n_agents"], r["lmbda"])][r["method"]].append(r)
    keys = sorted({k for k in d1} | {k for k in d0},
                  key=lambda x: (x[0], int(x[1]), float(x[2])))
    for k in keys:
        cells = []
        for dd in (d0, d1):
            s = dd[k].get("static", [])
            m = dd[k].get("memir", [])
            if not s or not m:
                cells.append(None)
                continue
            bs = sum(float(r.get("service_post") or 0.0) for r in s) / len(s)
            bm = sum(float(r.get("service_post") or 0.0) for r in m) / len(m)
            cells.append(((bm - bs) / bs * 100 if bs else float("nan"),
                          _avg(m, "plan_attempt_fail")))
        if cells[0] is None or cells[1] is None:
            continue
        g0, g1 = cells[0][0], cells[1][0]
        if g0 > 0 and g1 < 0:
            v = "**反转消除**"
        elif abs(g1 - g0) < 2.0:
            v = "基本不变（假设②证伪）"
        elif g1 - g0 < -2.0:
            v = "再增益 %.1f pp" % (g1 - g0)
        else:
            v = "退化 %.1f pp" % (g1 - g0)
        L.append("| %s | %s | %s | %+.2f%% | %+.2f%% | %s | %.0f | %.0f |"
                 % (k[0], k[1], k[2], g0, g1, v,
                    cells[0][1], cells[1][1]))
    L += ["",
          "### 结论", "",
          "- **假设②（移库途中不可派单）被证伪**：放宽规则后 B1/\\|A\\|=256 的增益从 "
          "+24.32% 变为 +25.28%（变化 <1 pp），反转**没有**消除。",
          "  且 static 的响应段降幅（34.29→23.01）大于 memir（34.97→29.66）"
          "—— 放宽可用性对基线的帮助更大。",
          "- **假设①（规划失败级联）被证伪**：24 格 `plan_fail` 全为 0。",
          "- **假设③（停在 task endpoint 上）被证伪**：`rest` 候选集与 "
          "task endpoint 的交集为 ∅（B1：388 个候选，0 个是 pickup/station）。",
          "- **真正的解释**：`plan_attempt_fail`（任务路径规划尝试失败）暴涨 —— "
          "B1/\\|A\\|=256 下 static 2 688 次 vs memir **22 873 次**（8.5×）。"
          "集中的空闲 body 让**任务路径本身**更难规划 ⇒ 任务滞留在 pending ⇒ "
          "leg_assign 从 3.51 涨到 30.74。在 B1 的 6 格内，paf(memir) 与增益**完全单调**："
          "307→−22.03%、871→−14.37%、1 494→−12.73%、5 094→−5.35%、"
          "10 454→+1.56%、22 873→+24.32%。",
          "- **机制含义**：ψ 只吸收**搬运干道**上的外部性；残留在极大 fleet 上的外部性是"
          "**泊位/通道可规划性**层面的，ψ 不覆盖。这是 §7.3 Limitations 与 future work 的落点。"]
    return L


def main():
    exps = sys.argv[1:] or ["A_main", "B_liveness", "C_planner", "D_tau",
                            "E_hetero", "F_drift", "G_maps", "H_psi",
                            "I_flow", "J_base", "K_learn", "L_gen",
                            "M_probe", "N_bench", "O_scale", "P_avail"]
    md = ["# 实验结果汇总（自动生成）", ""]
    for e in exps:
        rows = load(e)
        if not rows:
            md.append("## %s  —— 无数据" % e)
            continue
        md.append("_runs=%d, seeds=%s_" % (len(rows),
                                           sorted({r["seed"] for r in rows})))
        md.append("")
        if e == "A_main":
            md += table_main(rows)
        elif e == "B_liveness":
            md += table_liveness(rows)
        elif e == "C_planner":
            md += table_planner(rows)
        elif e == "D_tau":
            md += table_tau(rows)
        elif e == "E_hetero":
            md += table_factor(rows, "Table E  需求异质性 α × 到达强度 λ",
                               ["zipf_alpha", "lmbda", "n_agents"], legs=True)
        elif e == "F_drift":
            md += table_factor(rows, "Table F  非平稳漂移",
                               ["drift_mode", "drift_param", "n_agents"],
                               legs=True)
        elif e == "G_maps":
            md += table_factor(rows, "Table G  地图泛化",
                               ["map", "n_agents"], legs=True)
        elif e == "H_psi":
            md += table_psi(rows)
        elif e == "I_flow":
            md += table_flow(rows)
        elif e == "J_base":
            md += table_base(rows)
        elif e == "K_learn":
            md += table_learn(rows)
        elif e == "L_gen":
            md += table_gen(rows)
        elif e == "M_probe":
            md += table_probe(rows)
        elif e == "N_bench":
            md += table_bench(rows)
        elif e == "O_scale":
            md += table_scale(rows)
        elif e == "P_avail":
            # am=0 的对照行来自 O_scale，故两份一起喂进去
            md += table_pavail(rows + load("O_scale"))
    p = os.path.join(OUT, "tables.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("->", p)
    print("\n".join(md[:80]))


if __name__ == "__main__":
    main()
