"""Phase 3 结果分析：汇总表 + 零依赖手写 SVG 图表。"""
import csv
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results_open")

# 浅色主题（与 IDE light theme 一致）
BG = "#ffffff"
FG = "#1f2a44"
MUTED = "#6b7a99"
GRID = "#e3e8f0"
ACC = "#2563eb"
ACC2 = "#dc2626"
ACC3 = "#059669"
ACC4 = "#d97706"
PAL = [ACC, ACC2, ACC3, ACC4, "#7c3aed", "#0891b2", "#be185d", "#65a30d"]


def load():
    with open(os.path.join(OUT, "raw.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def agg(rows, field="service_post"):
    v = [num(r[field]) for r in rows]
    if not v:
        return 0.0, 0.0, 0
    m = st.mean(v)
    s = st.pstdev(v) if len(v) > 1 else 0.0
    return m, s, len(v)


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


# ------------------------------------------------------------------ SVG
def svg_open(w, h, title, xlab, ylab):
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
         'font-family="system-ui,-apple-system,Segoe UI,sans-serif">' % (w, h)]
    s.append('<rect width="%d" height="%d" fill="%s"/>' % (w, h, BG))
    s.append('<text x="%d" y="26" font-size="15" font-weight="600" fill="%s">%s</text>'
             % (w // 2, FG, esc(title)))
    s.append('<text x="%d" y="%d" font-size="11" fill="%s" text-anchor="middle">%s</text>'
             % (w // 2, h - 8, MUTED, esc(xlab)))
    s.append('<text x="14" y="%d" font-size="11" fill="%s" '
             'transform="rotate(-90 14 %d)">%s</text>'
             % (h // 2, MUTED, h // 2, esc(ylab)))
    return s


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def svg_close(s):
    s.append("</svg>")
    return "".join(s)


def axes(s, x0, y0, x1, y1, xmin, xmax, ymin, ymax, xticks, yticks,
         xfmt=lambda v: "%g" % v, yfmt=lambda v: "%g" % v, logx=False):
    def px(v):
        if logx:
            v = math.log10(max(v, 1e-9))
            a, b = math.log10(max(xmin, 1e-9)), math.log10(max(xmax, 1e-9))
        else:
            a, b = xmin, xmax
        if b == a:
            return x0
        return x0 + (v - a) / (b - a) * (x1 - x0)

    def py(v):
        if ymax == ymin:
            return y0
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    for v in yticks:
        y = py(v)
        s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
                 'stroke-width="1"/>' % (x0, y, x1, y, GRID))
        s.append('<text x="%g" y="%g" font-size="10" fill="%s" '
                 'text-anchor="end">%s</text>' % (x0 - 6, y + 3.5, MUTED, yfmt(v)))
    for v in xticks:
        x = px(v)
        s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
                 'stroke-width="1"/>' % (x, y1, x, y0 + (y1 - y0), GRID))
        s.append('<text x="%g" y="%g" font-size="10" fill="%s" '
                 'text-anchor="middle">%s</text>' % (x, y1 + 14, MUTED, xfmt(v)))
    s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1.5"/>'
             % (x0, y1, x1, y1, FG))
    s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1.5"/>'
             % (x0, y0, x0, y1, FG))
    return px, py


def legend(s, x, y, items):
    for i, (lab, col) in enumerate(items):
        yy = y + i * 17
        s.append('<rect x="%g" y="%g" width="12" height="3" rx="1.5" fill="%s"/>'
                 % (x, yy - 3, col))
        s.append('<text x="%g" y="%g" font-size="11" fill="%s">%s</text>'
                 % (x + 18, yy + 1, FG, esc(lab)))


def save(name, content):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    print("  wrote", name)
    return p


# ------------------------------------------------------------------ 图
def fig_e1(rows):
    r = sel(rows, group="E1_baseline")
    order = ["nearest", "random", "static", "aco_window", "aco_t100",
             "aco_t500", "aco_t2000", "aco_rl_credit", "aco_rl_meta", "oracle"]
    names = {"nearest": "nearest (无记忆)", "random": "random (阴性对照)",
             "static": "static (分散对照)", "aco_window": "ACO 无跨窗记忆",
             "aco_t100": "ACO τ=100", "aco_t500": "ACO τ=500",
             "aco_t2000": "ACO τ=2000", "aco_rl_credit": "ACO+RL 信用分配",
             "aco_rl_meta": "ACO+RL 元适应", "oracle": "ORACLE (genie)"}
    data = []
    for m in order:
        sub = sel(r, method=m)
        if sub:
            mu, sd, n = agg(sub)
            data.append((m, mu, sd))
    base = dict((m, v) for m, v, _ in data)["nearest"]
    W, H = 720, 420
    s = svg_open(W, H, "E1 基线对比：open-loop 小舰队 (|A|=8, λ=0.08, α=2.0)",
                 "", "service time（越小越好）")
    x0, y0, x1, y1 = 190, 60, W - 30, H - 60
    vals = [v for _, v, _ in data]
    ymin, ymax = 0, max(vals) * 1.12
    px, py = axes(s, x0, y0, x1, y1, 0, len(data), ymin, ymax, [],
                  [0, 10, 20, 30, 40, 50, 60])
    for i, (m, v, sd) in enumerate(data):
        cx = x0 + (i + 0.5) * (x1 - x0) / len(data)
        col = ACC if m.startswith("aco") else (MUTED if m == "nearest" else ACC3)
        if m == "oracle":
            col = ACC2
        s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
                 'stroke-width="2"/>' % (cx, py(v - sd), cx, py(v + sd), col))
        s.append('<rect x="%g" y="%g" width="34" height="%g" rx="2" fill="%s"/>'
                 % (cx - 17, py(v), y1 - py(v), col))
        s.append('<text x="%g" y="%g" font-size="10" fill="%s" '
                 'text-anchor="middle">%.1f</text>' % (cx, py(v) - 6, FG, v))
        d = (v / base - 1) * 100
        s.append('<text x="%g" y="%g" font-size="9.5" fill="%s" '
                 'text-anchor="middle">%+.1f%%</text>'
                 % (cx, py(v) - 19, ACC2 if d < 0 else MUTED, d))
        lab = names.get(m, m)
        parts = lab.split(" ", 1)
        s.append('<text x="%g" y="%g" font-size="9.5" fill="%s" '
                 'text-anchor="end">%s</text>' % (x0 - 8, py(v) + 3.5, FG, parts[0]))
        if len(parts) > 1:
            s.append('<text x="%g" y="%g" font-size="8.5" fill="%s" '
                     'text-anchor="end">%s</text>'
                     % (x0 - 8, py(v) + 14, MUTED, parts[1]))
    return save("fig_e1_baseline.svg", svg_close(s))


def fig_e2(rows):
    r = sel(rows, group="E2_fleet")
    ns = [4, 6, 8, 12, 16, 24, 32]
    series = [("nearest", "nearest", MUTED), ("static", "static 分散", ACC4),
              ("aco", "ACO τ=100", ACC), ("meta", "ACO+RL 元适应", ACC3),
              ("oracle", "ORACLE", ACC2)]
    W, H = 720, 430
    s = svg_open(W, H, "E2 记忆价值随舰队规模衰减（P1）",
                 "number of agents |A|（λ=0.08 固定，饱和度随之下降）",
                 "service time")
    x0, y0, x1, y1 = 80, 60, W - 30, H - 60
    allv = []
    for pre, _, _ in series:
        for n in ns:
            sub = sel(r, method="%s_n%d" % (pre, n))
            if sub:
                allv.append(agg(sub)[0])
    ymin, ymax = min(allv) * 0.92, max(allv) * 1.05
    px, py = axes(s, x0, y0, x1, y1, min(ns), max(ns), ymin, ymax, ns,
                  [round(v) for v in
                   [ymin + (ymax - ymin) * k / 4 for k in range(5)]])
    for (pre, lab, col) in series:
        pts = []
        for n in ns:
            sub = sel(r, method="%s_n%d" % (pre, n))
            if sub:
                pts.append((n, agg(sub)[0]))
        if len(pts) < 2:
            continue
        d = " ".join("%g,%g" % (px(a), py(b)) for a, b in pts)
        s.append('<polyline points="%s" fill="none" stroke="%s" '
                 'stroke-width="2.2"/>' % (d, col))
        for a, b in pts:
            s.append('<circle cx="%g" cy="%g" r="3.2" fill="%s"/>'
                     % (px(a), py(b), col))
    # 增益间隙标注（aco vs nearest）
    for n in ns:
        a = sel(r, method="nearest_n%d" % n)
        b = sel(r, method="aco_n%d" % n)
        if a and b:
            va, vb = agg(a)[0], agg(b)[0]
            s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
                     'stroke-width="1" stroke-dasharray="2,2" opacity="0.55"/>'
                     % (px(n), py(va), px(n), py(vb), ACC))
            s.append('<text x="%g" y="%g" font-size="9" fill="%s" '
                     'text-anchor="middle">%+.0f%%</text>'
                     % (px(n), py(vb) - 7, ACC, (vb / va - 1) * 100))
    legend(s, x0 + 16, y0 + 14, [(lab, col) for _, lab, col in series])
    return save("fig_e2_fleet.svg", svg_close(s))


def fig_e3(rows):
    r = sel(rows, group="E3_alpha")
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0]
    series = [("nearest", "nearest", MUTED), ("static", "static", ACC4),
              ("aco", "ACO τ=100", ACC), ("oracle", "ORACLE", ACC2)]
    W, H = 700, 420
    s = svg_open(W, H, "E3 记忆价值随空间不均匀度上升（P2/P3）",
                 "Zipf α（0 = 任务在空间上均匀分布）",
                 "ACO 相对 nearest 的增益 %")
    x0, y0, x1, y1 = 80, 60, W - 30, H - 60
    gains = []
    for a in alphas:
        nn = sel(r, method="nearest_a%.1f" % a)
        ac = sel(r, method="aco_a%.1f" % a)
        if nn and ac:
            va, vb = agg(nn)[0], agg(ac)[0]
            gains.append((a, (vb / va - 1) * 100))
    ys = [g for _, g in gains] + [0]
    ymin, ymax = min(ys) * 1.25 - 2, 2
    px, py = axes(s, x0, y0, x1, y1, 0, 3.0, ymin, ymax, alphas,
                  [round(ymin + (ymax - ymin) * k / 5, 1) for k in range(6)])
    s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
             'stroke-dasharray="3,3" stroke-width="1"/>'
             % (x0, py(0), x1, py(0), MUTED))
    d = " ".join("%g,%g" % (px(a), py(g)) for a, g in gains)
    s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.4"/>'
             % (d, ACC))
    for a, g in gains:
        s.append('<circle cx="%g" cy="%g" r="3.6" fill="%s"/>'
                 % (px(a), py(g), ACC))
        s.append('<text x="%g" y="%g" font-size="10" fill="%s" '
                 'text-anchor="middle">%+.1f%%</text>'
                 % (px(a), py(g) - 9, ACC, g))
    return save("fig_e3_alpha.svg", svg_close(s))


def fig_e4(rows):
    W, H = 720, 430
    s = svg_open(W, H, "E4 时间常数扫描：倒 U 型与最优 τ 的迁移",
                 "信息素时间常数 τ_φ（对数刻度）", "service time")
    x0, y0, x1, y1 = 80, 60, W - 30, H - 60
    taus = [20, 50, 100, 300, 1000, 3000]
    cols = {0.02: ACC, 0.04: ACC3, 0.08: ACC2}
    allv = []
    for lam in (0.02, 0.04, 0.08):
        r = sel(rows, group="E4_tau_l%.2f" % lam)
        for tp in taus:
            sub = sel(r, method="aco_t%d" % tp)
            if sub:
                allv.append(agg(sub)[0])
    ymin, ymax = min(allv) * 0.95, max(allv) * 1.05
    px, py = axes(s, x0, y0, x1, y1, min(taus), max(taus), ymin, ymax,
                  taus, [round(ymin + (ymax - ymin) * k / 4, 1) for k in range(5)],
                  logx=True)
    for lam in (0.02, 0.04, 0.08):
        r = sel(rows, group="E4_tau_l%.2f" % lam)
        pts = []
        for tp in taus:
            sub = sel(r, method="aco_t%d" % tp)
            if sub:
                pts.append((tp, agg(sub)[0]))
        if len(pts) < 2:
            continue
        d = " ".join("%g,%g" % (px(a), py(b)) for a, b in pts)
        s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.2"/>'
                 % (d, cols[lam]))
        for a, b in pts:
            s.append('<circle cx="%g" cy="%g" r="3" fill="%s"/>' % (px(a), py(b), cols[lam]))
        best = min(pts, key=lambda x: x[1])
        s.append('<circle cx="%g" cy="%g" r="6" fill="none" stroke="%s" '
                 'stroke-width="1.6" stroke-dasharray="3,2"/>'
                 % (px(best[0]), py(best[1]), cols[lam]))
    legend(s, x0 + 16, y0 + 12,
           [("λ=0.02 (τ*=300)", ACC), ("λ=0.04 (τ*=100)", ACC3),
            ("λ=0.08 (τ*=100)", ACC2)])
    return save("fig_e4_tau.svg", svg_close(s))


def fig_e5(rows):
    r = sel(rows, group="E5_drift")
    conds = [("static0", "静止"), ("shift1500", "突变/1500"),
             ("shift750", "突变/750"), ("shift400", "突变/400"),
             ("periodic1500", "周期/1500"), ("periodic750", "周期/750")]
    series = [("aco_t100", "ACO τ=100", ACC), ("aco_t500", "ACO τ=500", ACC4),
              ("aco_t2000", "ACO τ=2000", MUTED),
              ("meta", "ACO+RL 元适应 τ_fast=100", ACC2)]
    W, H = 760, 440
    s = svg_open(W, H, "E5 环境漂移下：固定 τ 失效，元适应保持鲁棒",
                 "", "相对 nearest 的增益 %（越负越好）")
    x0, y0, x1, y1 = 90, 60, W - 30, H - 70
    allg = []
    vals = {}
    for key, _ in conds:
        nn = sel(r, method="nearest_" + key)
        if not nn:
            continue
        b = agg(nn)[0]
        for pre, _, _ in series:
            sub = sel(r, method="%s_%s" % (pre, key))
            if sub:
                g = (agg(sub)[0] / b - 1) * 100
                vals[(pre, key)] = g
                allg.append(g)
    ymin, ymax = min(allg) * 1.15, max(allg) * 0.9
    px, py = axes(s, x0, y0, x1, y1, 0, len(conds), ymin, ymax, [],
                  [round(ymin + (ymax - ymin) * k / 5) for k in range(6)])
    nser = len(series)
    bw = (x1 - x0) / len(conds)
    for i, (key, lab) in enumerate(conds):
        for j, (pre, _, col) in enumerate(series):
            g = vals.get((pre, key))
            if g is None:
                continue
            cx = x0 + i * bw + bw * (j + 0.5) / nser
            s.append('<rect x="%g" y="%g" width="%g" height="%g" fill="%s" '
                     'opacity="0.9"/>'
                     % (cx - bw / nser * 0.4, py(g), bw / nser * 0.8,
                        max(y1 - py(g), 1), col))
        s.append('<text x="%g" y="%g" font-size="10" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (x0 + i * bw + bw / 2, y1 + 16, FG, lab))
    legend(s, x0 + 10, y0 + 10, [(lab, col) for _, lab, col in series])
    return save("fig_e5_drift.svg", svg_close(s))


# ------------------------------------------------------------------ 表
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    rows = load()
    print("rows:", len(rows))
    figs = [fig_e1(rows), fig_e2(rows), fig_e3(rows), fig_e4(rows), fig_e5(rows)]

    # ---- summary.csv ----
    groups = {}
    for r in rows:
        groups.setdefault((r["group"], r["method"]), []).append(r)
    with open(os.path.join(OUT, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "n", "service_mean", "service_sd",
                    "throughput", "mean_pending", "idle_ratio",
                    "demand_spread", "conflicts", "plan_fail"])
        for (g, m), rs in sorted(groups.items()):
            mu, sd, n = agg(rs)
            w.writerow([g, m, n, round(mu, 3), round(sd, 3),
                        round(agg(rs, "throughput_post")[0], 4),
                        round(agg(rs, "mean_pending")[0], 3),
                        round(agg(rs, "idle_ratio")[0], 4),
                        round(agg(rs, "demand_spread")[0], 4),
                        sum(int(num(x["conflicts"])) for x in rs),
                        sum(int(num(x["plan_fail"])) for x in rs)])
    print("wrote summary.csv")

    # ---- summary.md ----
    L = ["# Phase 3 结果汇总（open-loop 非饱和 MAPD）", ""]
    for g in sorted(set(r["group"] for r in rows)):
        sub = sel(rows, group=g)
        base = None
        for m in sorted(set(r["method"] for r in sub)):
            if m.startswith("nearest"):
                base = agg(sel(sub, method=m))[0]
                break
        L.append("## %s" % g)
        L.append("")
        L.append(md_table(["method", "service", "sd", "vs nearest"],
                          [[m, "%.2f" % agg(sel(sub, method=m))[0],
                            "%.2f" % agg(sel(sub, method=m))[1],
                            ("%+.2f%%" % ((agg(sel(sub, method=m))[0] / base - 1) * 100))
                            if base else "-"]
                           for m in sorted(set(r["method"] for r in sub))]))
        L.append("")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote summary.md")
    for p in figs:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
