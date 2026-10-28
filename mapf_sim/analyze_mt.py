"""Phase 2（MT-MAPD）结果分析与作图。零第三方依赖：手写 SVG。

读取 results_mt/raw.csv，产出：
  results_mt/summary.csv      各方法 × 条件的汇总
  results_mt/summary.md       Markdown 汇总表
  results_mt/fig_capacity.svg  容量扫描（吞吐 + 边际行程）
  results_mt/fig_construct.svg 序列构造器对比
  results_mt/fig_ph.svg        信息素参数扫描
  results_mt/fig_cond.svg      条件矩阵（α × |A|）
  results_mt/fig_drift.svg     非平稳漂移
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results_mt")

INK = "#2c3446"
MUTED = "#6b7280"
GRID = "#e3e6ea"
C_BASE = "#8a8f98"
C_BLUE = "#185FA5"
C_GREEN = "#0F6E56"
C_ORANGE = "#BA7517"
C_PURPLE = "#534AB7"
C_RED = "#A32D2D"

NUM = ("throughput_post", "completed_post", "service_time", "geom_per_batch",
       "margin_per_task", "steps_per_task", "avg_batch", "wait_ratio",
       "tau2_spread", "alpha")
INT = ("cap", "n_agents", "seed", "conflicts", "plan_fail", "drift_param")


def load(path=None):
    path = path or os.path.join(RES, "raw.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in INT:
                if k in r and r[k] not in ("", None):
                    try:
                        r[k] = int(float(r[k]))
                    except ValueError:
                        pass
            for k in NUM:
                if k in r and r[k] not in ("", None):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def group(rows, keys):
    g = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        if len(k) == 1:
            k = k[0]
        g.setdefault(k, []).append(r)
    return g


def agg(rs, field="throughput_post"):
    v = [r[field] for r in rs]
    m = st.mean(v)
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    return m, sd


# =============================================================================
# SVG 绘图
# =============================================================================
def _svg_header(w, h, title, sub=""):
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'viewBox="0 0 %d %d" font-family="system-ui,-apple-system,Segoe UI,'
         'sans-serif">' % (w, h, w, h)]
    s.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (w, h))
    s.append('<text x="24" y="30" font-size="16" font-weight="600" fill="%s">'
             '%s</text>' % (INK, title))
    if sub:
        s.append('<text x="24" y="50" font-size="11.5" fill="%s">%s</text>'
                 % (MUTED, sub))
    return s


def _nice(v):
    """取一个好看的轴上限。"""
    if v <= 0:
        return 1.0
    import math
    e = math.floor(math.log10(v))
    b = v / (10 ** e)
    for c in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if b <= c:
            return c * (10 ** e)
    return 10.0 * (10 ** e)


def bar_chart(path, title, sub, labels, values, sds, colors, ylabel,
              zero_line=True):
    n = len(labels)
    W = max(560, 90 + n * 68)
    H = 380
    ml, mr, mt, mb = 62, 24, 74, 92
    pw, ph = W - ml - mr, H - mt - mb
    vmax = max([abs(v) + (s or 0) for v, s in zip(values, sds)] + [1e-9])
    vmin = min([v - (s or 0) for v, s in zip(values, sds)] + [0.0])
    top = _nice(vmax)
    bot = -_nice(abs(vmin)) if vmin < 0 else 0.0
    rng = top - bot

    def y(v):
        return mt + ph - (v - bot) / rng * ph

    s = _svg_header(W, H, title, sub)
    # 网格
    steps = 5
    for i in range(steps + 1):
        v = bot + rng * i / steps
        yy = y(v)
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                 'stroke-width="1"/>' % (ml, yy, ml + pw, yy, GRID))
        s.append('<text x="%d" y="%.1f" font-size="10.5" fill="%s" '
                 'text-anchor="end">%.4g</text>' % (ml - 8, yy + 3.5, MUTED, v))
    if zero_line and bot < 0 < top:
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                 'stroke-width="1.4"/>' % (ml, y(0), ml + pw, y(0), INK))
    bw = pw / n * 0.62
    gap = pw / n
    for i, (lb, v, sd, c) in enumerate(zip(labels, values, sds, colors)):
        cx = ml + gap * (i + 0.5)
        x0 = cx - bw / 2
        y0, y1 = y(0 if bot == 0 else min(v, 0)), y(max(v, 0) if v > 0 else v)
        if v < 0:
            y0, y1 = y(v), y(0)
        h = max(abs(y1 - y0), 1.0)
        s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" '
                 'rx="2"/>' % (x0, min(y0, y1), bw, h, c))
        if sd:
            ey = y(v + sd) if v >= 0 else y(v - sd)
            ey2 = y(v - sd) if v >= 0 else y(v + sd)
            s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                     'stroke="%s" stroke-width="1.2"/>'
                     % (cx, ey, cx, ey2, INK))
        ty = min(y0, y1) - 6 if v >= 0 else max(y0, y1) + 13
        s.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" '
                 'text-anchor="middle">%+.2f</text>' % (cx, ty, INK, v))
        s.append('<text x="%.1f" y="%.1f" font-size="10" fill="%s" '
                 'text-anchor="end" transform="rotate(-38 %.1f %.1f)">%s</text>'
                 % (cx, mt + ph + 16, INK, cx, mt + ph + 16, lb))
    s.append('<text x="%d" y="%d" font-size="11" fill="%s">%s</text>'
             % (ml, mt - 14, MUTED, ylabel))
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    return path


def line_chart(path, title, sub, xs, series, xlabel, ylabel, colors):
    """series: [(name, [y...]), ...]"""
    W, H = 620, 380
    ml, mr, mt, mb = 64, 130, 74, 62
    pw, ph = W - ml - mr, H - mt - mb
    allv = [v for _, ys in series for v in ys if v is not None]
    vmax = _nice(max(allv + [1e-9]))
    vmin = min(allv + [0.0])
    if vmin < 0:
        vmin = -_nice(abs(vmin))
    rng = vmax - vmin

    def y(v):
        return mt + ph - (v - vmin) / rng * ph

    def x(i):
        return ml + (pw * i / max(len(xs) - 1, 1))

    s = _svg_header(W, H, title, sub)
    for i in range(6):
        v = vmin + rng * i / 5
        yy = y(v)
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>'
                 % (ml, yy, ml + pw, yy, GRID))
        s.append('<text x="%d" y="%.1f" font-size="10.5" fill="%s" '
                 'text-anchor="end">%.4g</text>' % (ml - 8, yy + 3.5, MUTED, v))
    if vmin < 0 < vmax:
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                 'stroke-width="1.4"/>' % (ml, y(0), ml + pw, y(0), INK))
    for i, xv in enumerate(xs):
        s.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (x(i), mt + ph + 18, INK, xv))
    for k, (name, ys) in enumerate(series):
        c = colors[k % len(colors)]
        pts = " ".join("%.1f,%.1f" % (x(i), y(v)) for i, v in enumerate(ys)
                       if v is not None)
        s.append('<polyline points="%s" fill="none" stroke="%s" '
                 'stroke-width="2.2"/>' % (pts, c))
        for i, v in enumerate(ys):
            if v is None:
                continue
            s.append('<circle cx="%.1f" cy="%.1f" r="3.2" fill="%s"/>'
                     % (x(i), y(v), c))
        ly = mt + 16 + k * 19
        s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                 'stroke-width="2.2"/>' % (ml + pw + 14, ly, ml + pw + 34, ly, c))
        s.append('<text x="%d" y="%d" font-size="10.5" fill="%s">%s</text>'
                 % (ml + pw + 40, ly + 3.5, INK, name))
    s.append('<text x="%d" y="%d" font-size="11" fill="%s">%s</text>'
             % (ml, mt - 14, MUTED, ylabel))
    s.append('<text x="%.1f" y="%d" font-size="11" fill="%s" '
             'text-anchor="middle">%s</text>'
             % (ml + pw / 2, H - 12, MUTED, xlabel))
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    return path


# =============================================================================
# 汇总
# =============================================================================
def build_summary(rows):
    """按 (group, method) 聚合；增益相对同条件下的 nn（容量相同）。"""
    out = []
    for grp, rs in sorted(group(rows, ["group"]).items()):
        caps = sorted({r["cap"] for r in rs})
        for cap in caps:
            sub = [r for r in rs if r["cap"] == cap]
            by = group(sub, ["method"])
            # 基线优先级：nn2opt > nn > 第一个
            bname = "nn2opt" if "nn2opt" in by else ("nn" if "nn" in by else None)
            bmean = st.mean([r["throughput_post"] for r in by[bname]]) \
                if bname else None
            for m in sorted(by.keys()):
                v = by[m]
                tp, tsd = agg(v, "throughput_post")
                gm, gsd = agg(v, "geom_per_batch")
                mg, _ = agg(v, "margin_per_task")
                spt, _ = agg(v, "steps_per_task")
                out.append(dict(
                    group=grp, cap=cap, method=m,
                    n_runs=len(v),
                    throughput=round(tp, 1), sd=round(tsd, 1),
                    gain_pct=round((tp / bmean - 1) * 100, 2) if bname and m != bname else 0.0,
                    baseline=bname or "",
                    geom_per_batch=round(gm, 2),
                    geom_gain_pct=round((gm / st.mean([r["geom_per_batch"] for r in by[bname]]) - 1) * 100, 2)
                    if bname and m != bname else 0.0,
                    margin_per_task=round(mg, 2),
                    steps_per_task=round(spt, 2),
                    service_time=round(agg(v, "service_time")[0], 1),
                    avg_batch=round(agg(v, "avg_batch")[0], 2),
                    conflicts=sum(r["conflicts"] for r in v),
                    plan_fail=sum(r["plan_fail"] for r in v),
                    tau_spread=round(agg(v, "tau2_spread")[0], 4),
                ))
    return out


def write_csv(sumry, path=None):
    path = path or os.path.join(RES, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sumry[0].keys()))
        w.writeheader()
        w.writerows(sumry)
    return path


def write_md(sumry, path=None):
    path = path or os.path.join(RES, "summary.md")
    lines = ["# Phase 2（MT-MAPD）实验汇总", ""]
    for grp, items in sorted(group(sumry, ["group"]).items()):
        base = ""
        for it in items:
            if it["baseline"]:
                base = it["baseline"]
                break
        lines.append("## %s（基线 = %s）" % (grp, base or "-"))
        lines.append("")
        lines.append("| cap | 方法 | 吞吐 | sd | 增益% | 序列几何长 | 几何增益% | "
                     "边际行程/任务 | 步/任务 | service | 冲突 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for it in items:
            lines.append("| %s | %s | %.1f | %.1f | %+.2f%% | %.2f | %+.2f%% | "
                         "%.2f | %.2f | %.1f | %d |"
                         % (it["cap"], it["method"], it["throughput"], it["sd"],
                            it["gain_pct"], it["geom_per_batch"],
                            it["geom_gain_pct"], it["margin_per_task"],
                            it["steps_per_task"], it["service_time"],
                            it["conflicts"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    rows = load()
    print("载入 %d 行" % len(rows))
    sumry = build_summary(rows)
    print("汇总 %d 条" % len(sumry))
    write_csv(sumry)
    write_md(sumry)

    by = group(rows, ["group"])

    # ---- 图1：容量扫描 ----
    e1 = by.get("E1_capacity", [])
    caps = sorted({r["cap"] for r in e1})
    tp = [agg([r for r in e1 if r["cap"] == c])[0] for c in caps]
    mg = [agg([r for r in e1 if r["cap"] == c], "margin_per_task")[0] for c in caps]
    spt = [agg([r for r in e1 if r["cap"] == c], "steps_per_task")[0] for c in caps]
    line_chart(os.path.join(RES, "fig_capacity.svg"),
               "E1 容量扫描：MT-MAPD 的吞吐增益与边际行程规模经济",
               "closed-loop 饱和，|A|=40，Zipf α=1.0，任务池深度 8|A|，6 seeds",
               [str(c) for c in caps],
               [("吞吐 (tasks/1k steps)", tp),
                ("每任务步数", spt),
                ("追加任务的边际行程", mg)],
               "agent 容量 c（一次携带的任务数）", "指标值",
               [C_BLUE, C_ORANGE, C_GREEN])

    # ---- 图2：构造器对比 ----
    e2 = by.get("E2_constructor", [])
    labels, vals, sds, cols = [], [], [], []
    order = ["random", "nn", "cluster", "cluster2opt", "nn2opt", "oracle"]
    for cap in sorted({r["cap"] for r in e2}):
        sub = [r for r in e2 if r["cap"] == cap]
        bm = agg([r for r in sub if r["method"] == "nn_c%d" % cap])[0]
        for m in order:
            key = "%s_c%d" % (m, cap)
            v = [r for r in sub if r["method"] == key]
            if not v:
                continue
            mu, sd = agg(v)
            labels.append("c%d %s" % (cap, m))
            vals.append((mu / bm - 1) * 100)
            sds.append(sd / bm * 100)
            cols.append(C_BASE if m in ("nn", "cluster") else
                        (C_GREEN if "2opt" in m or m == "oracle" else C_RED))
    bar_chart(os.path.join(RES, "fig_construct.svg"),
              "E2 序列构造器对比（相对链式最近邻 NN 的吞吐增益）",
              "2-opt 局部搜索几乎完全拿到穷举最优的收益；聚簇式集合策略反而更差",
              labels, vals, sds, cols, "吞吐增益 %")

    # ---- 图3：信息素参数扫描 ----
    e3 = by.get("E3_pheromone", [])
    bm2 = agg([r for r in e3 if r["method"] == "ph2opt_c4_t500.0_g10_x0.0"], )[0] \
        if False else None
    # 基线取同条件下的 nn2opt（E2 cap=4）
    e2c4 = [r for r in e2 if r["cap"] == 4]
    base2opt = agg([r for r in e2c4 if r["method"] == "nn2opt_c4"])[0]
    labels, vals, sds, cols = [], [], [], []
    for m in sorted({r["method"] for r in e3}):
        v = [r for r in e3 if r["method"] == m]
        mu, sd = agg(v)
        labels.append(m.replace("ph2opt_c4_", "2opt ").replace("ph_c4_", "")
                      .replace("t500.0_", "").replace("_", " "))
        vals.append((mu / base2opt - 1) * 100)
        sds.append(sd / base2opt * 100)
        cols.append(C_BLUE if m.startswith("ph2opt") else C_ORANGE)
    bar_chart(os.path.join(RES, "fig_ph.svg"),
              "E3 序列级信息素参数扫描（相对 nn+2opt 的吞吐增益）",
              "τ_φ=记忆时间常数，γ=信息素重要性，ξ=局部更新（多样性）强度；全部落在噪声内",
              labels, vals, sds, cols, "吞吐增益 %")

    # ---- 图4：条件矩阵 ----
    e4 = by.get("E4_conditions", [])
    alphas = sorted({r["alpha"] for r in e4})
    series, names = [], []
    for n in sorted({r["n_agents"] for r in e4}):
        ys = []
        for a in alphas:
            sub = [r for r in e4 if r["n_agents"] == n and r["alpha"] == a]
            bm = agg([r for r in sub if r["method"] == "nn2opt_a%g_n%d" % (a, n)])[0]
            ph = agg([r for r in sub if r["method"] == "ph2opt_a%g_n%d" % (a, n)])[0]
            ys.append((ph / bm - 1) * 100 if bm else None)
        series.append(ys)
        names.append("|A|=%d" % n)
    line_chart(os.path.join(RES, "fig_cond.svg"),
               "E4 条件矩阵：信息素相对 nn+2opt 的增益 vs 空间不均匀度与负载",
               "增益在所有条件下均落在 ±2% 噪声带内",
               ["α=%.1f" % a for a in alphas],
               list(zip(names, series)),
               "任务空间不均匀度 Zipf α", "信息素增益 %",
               [C_BLUE, C_ORANGE, C_GREEN])

    # ---- 图5：漂移 ----
    e5 = by.get("E5_drift", [])
    labels, vals, sds, cols = [], [], [], []
    for dm, dp in [("static", 0), ("periodic", 400), ("periodic", 1200),
                   ("shift", 400), ("shift", 1200)]:
        sub = [r for r in e5 if r["drift_mode"] == dm and r["drift_param"] == dp]
        if not sub:
            continue
        bm = agg([r for r in sub if r["method"] == "nn2opt_%s%d" % (dm, dp)])[0]
        labels.append("%s%d\n基线" % (dm, dp))
        vals.append(0.0)
        sds.append(0.0)
        cols.append(C_BASE)
        for tp in [100.0, 500.0, 2000.0]:
            v = [r for r in sub
                 if r["method"] == "ph2opt_%s%d_t%d" % (dm, dp, tp)]
            if not v:
                continue
            mu, sd = agg(v)
            labels.append("τ_φ=%d" % tp)
            vals.append((mu / bm - 1) * 100)
            sds.append(sd / bm * 100)
            cols.append(C_BLUE)
    bar_chart(os.path.join(RES, "fig_drift.svg"),
              "E5 非平稳任务分布下信息素的增益（相对 nn+2opt）",
              "periodic=可预测周期轮换，shift=不可预测突变；数字为漂移周期（步）",
              labels, vals, sds, cols, "吞吐增益 %")

    print("图表已生成于", RES)


if __name__ == "__main__":
    main()
