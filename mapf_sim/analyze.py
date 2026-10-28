"""结果分析与作图。零第三方依赖：统计用 statistics，图表直接手写 SVG。

读取 results/main_results.csv，产出：
  - results/summary.csv        各条件 × 方法的汇总统计
  - results/summary.md         Markdown 汇总表
  - results/fig_main.svg       主结果：方法对比
  - results/fig_tau.svg        tau_phi 扫描
  - results/fig_alpha.svg      增益 vs 空间不均匀度
"""
import os
import csv
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

METHOD_ORDER = ["greedy", "hungarian", "ph_window", "ph_ema", "oracle"]
METHOD_LABEL = {
    "greedy": "TP-greedy",
    "hungarian": "Hungarian",
    "ph_window": "PH-window (no memory)",
    "ph_ema": "PH-EMA (proposed)",
    "oracle": "Oracle (true field)",
}
COLORS = {
    "greedy": "#888780",
    "hungarian": "#185FA5",
    "ph_window": "#BA7517",
    "ph_ema": "#0F6E56",
    "oracle": "#534AB7",
}


def load(path=None):
    path = path or os.path.join(RES, "main_results.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("n_agents", "seed", "tau_phi", "completed", "conflicts",
                      "plan_fail"):
                if k in r and r[k] not in ("", None):
                    try:
                        r[k] = int(float(r[k]))
                    except ValueError:
                        pass
            for k in ("alpha", "throughput", "throughput_all", "service_time",
                      "detour", "wait_ratio"):
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
            k = k[0]          # 单键时返回标量，便于按值直接索引
        g.setdefault(k, []).append(r)
    return g


def summarize(rows):
    """按 (n_agents, alpha) 汇总每个方法。基线 = hungarian。"""
    out = []
    for (n, a), rs in sorted(group(rows, ["n_agents", "alpha"]).items()):
        by = group(rs, ["method"])
        base = by.get("hungarian", [])
        bmean = st.mean([r["throughput"] for r in base]) if base else float("nan")
        bsd = st.pstdev([r["throughput"] for r in base]) if len(base) > 1 else 0.0
        for m in METHOD_ORDER:
            if m not in by:
                continue
            v = by[m]
            tp = [r["throughput"] for r in v]
            mean = st.mean(tp)
            sd = st.pstdev(tp) if len(tp) > 1 else 0.0
            # 同一 method 下按 tau_phi 再分组（ph_ema 有多档）
            taus = sorted({r["tau_phi"] for r in v if r["tau_phi"]})
            out.append(dict(
                n_agents=n, alpha=a, method=m, label=METHOD_LABEL.get(m, m),
                n_runs=len(v), throughput=round(mean, 1), sd=round(sd, 1),
                gain_pct=round((mean / bmean - 1) * 100, 2) if bmean else 0.0,
                service_time=round(st.mean([r["service_time"] for r in v]), 1),
                detour=round(st.mean([r["detour"] for r in v]), 3),
                wait_ratio=round(st.mean([r["wait_ratio"] for r in v]), 4),
                conflicts=sum(r["conflicts"] for r in v),
                plan_fail=sum(r["plan_fail"] for r in v),
                tau_phi=";".join(str(t) for t in taus),
            ))
    return out


def write_summary_csv(sumry, path=None):
    path = path or os.path.join(RES, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sumry[0].keys()))
        w.writeheader()
        w.writerows(sumry)
    return path


def write_summary_md(sumry, path=None):
    path = path or os.path.join(RES, "summary.md")
    lines = ["# 实验汇总（基线 = Hungarian）", ""]
    for (n, a), items in sorted(group(sumry, ["n_agents", "alpha"]).items()):
        lines.append("## |A|=%d, Zipf α=%.1f" % (n, a))
        lines.append("")
        lines.append("| 方法 | 吞吐 (tasks/1k steps) | sd | 增益 % | service time | detour | conflicts |")
        lines.append("|---|---|---|---|---|---|---|")
        for it in items:
            lines.append("| %s | %.1f | %.1f | %+.2f%% | %.1f | %.3f | %d |"
                         % (it["label"], it["throughput"], it["sd"],
                            it["gain_pct"], it["service_time"], it["detour"],
                            it["conflicts"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------- SVG 作图（纯手写） ----------------
def _svg_header(w, h):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
            'width="100%%" font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
            % (w, h))


def _axes(x0, y0, x1, y1):
    return ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#B4B2A9" '
            'stroke-width="1"/>' % (x0, y0, x1, y1))


def fig_main(sumry, path=None):
    """主结果：每个 (负载, α) 条件下各方法的吞吐柱状图。"""
    path = path or os.path.join(RES, "fig_main.svg")
    conds = sorted({(s["n_agents"], s["alpha"]) for s in sumry})
    W = 200 + 220 * len(conds)
    H = 380
    x0, y0 = 70, 300          # 原点
    plot_h = 230
    s = [_svg_header(W, H)]
    s.append('<text x="20" y="26" font-size="15" font-weight="500" fill="#2C2C2A">'
             'Throughput by method (baseline = Hungarian)</text>')
    s.append(_axes(x0, y0, x0, y0 - plot_h))
    s.append(_axes(x0, y0, x0 + 200 * len(conds), y0))

    # 统一 y 轴
    vmax = max(x["throughput"] for x in sumry) * 1.12
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yv = y0 - frac * plot_h
        s.append('<line x1="%d" y1="%.0f" x2="%d" y2="%.0f" stroke="#F1EFE8"/>'
                 % (x0, yv, x0 + 200 * len(conds), yv))
        s.append('<text x="%d" y="%.0f" font-size="11" fill="#888780" '
                 'text-anchor="end" dominant-baseline="central">%.0f</text>'
                 % (x0 - 8, yv, vmax * frac))

    for ci, (n, a) in enumerate(conds):
        items = [x for x in sumry if x["n_agents"] == n and x["alpha"] == a]
        items = sorted(items, key=lambda x: METHOD_ORDER.index(x["method"]))
        bx = x0 + 30 + ci * 200
        bw = 22
        for i, it in enumerate(items):
            hgt = it["throughput"] / vmax * plot_h
            cx = bx + i * (bw + 6)
            s.append('<rect x="%d" y="%.1f" width="%d" height="%.1f" rx="3" fill="%s"/>'
                     % (cx, y0 - hgt, bw, hgt, COLORS.get(it["method"], "#888780")))
            s.append('<text x="%d" y="%.1f" font-size="11" fill="#5F5E5A" '
                     'text-anchor="middle">%s</text>'
                     % (cx + bw / 2, y0 + 14,
                        {"greedy": "TP", "hungarian": "HUNG",
                         "ph_window": "PHw", "ph_ema": "PH",
                         "oracle": "ORC"}.get(it["method"], "?")))
        s.append('<text x="%d" y="%d" font-size="12" font-weight="500" '
                 'fill="#2C2C2A" text-anchor="middle">|A|=%d</text>'
                 % (bx + 60, y0 + 34, n))
        s.append('<text x="%d" y="%d" font-size="11" fill="#888780" '
                 'text-anchor="middle">alpha=%.1f</text>' % (bx + 60, y0 + 50, a))

    # 图例
    lx = x0 + 200 * len(conds) - 150
    for i, m in enumerate(METHOD_ORDER):
        s.append('<rect x="%d" y="%d" width="12" height="12" rx="2" fill="%s"/>'
                 % (lx, 60 + i * 20, COLORS[m]))
        s.append('<text x="%d" y="%d" font-size="11" fill="#5F5E5A" '
                 'dominant-baseline="central">%s</text>'
                 % (lx + 18, 66 + i * 20, METHOD_LABEL[m]))
    s.append('<text x="%d" y="%d" font-size="11" fill="#888780">'
             'throughput (tasks / 1000 timesteps), warmup excluded</text>'
             % (x0, y0 + 74))
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(s))
    return path


def fig_tau(rows, path=None):
    """τ_φ 扫描：信息素增益 vs 时间常数。"""
    path = path or os.path.join(RES, "fig_tau.svg")
    W, H = 720, 380
    x0, y0 = 80, 300
    plot_w, plot_h = 560, 220
    s = [_svg_header(W, H)]
    s.append('<text x="20" y="26" font-size="15" font-weight="500" fill="#2C2C2A">'
             'Pheromone gain vs memory time constant tau_phi</text>')
    s.append(_axes(x0, y0, x0 + plot_w, y0))
    s.append(_axes(x0, y0, x0, y0 - plot_h))

    taus = sorted({r["tau_phi"] for r in rows
                   if r["method"] == "ph_ema" and r["tau_phi"]})
    base = {}
    for (n, a), rs in group(rows, ["n_agents", "alpha"]).items():
        hs = [r["throughput"] for r in rs if r["method"] == "hungarian"]
        if hs:
            base[(n, a)] = st.mean(hs)

    series = {}
    for (n, a), rs in sorted(group(rows, ["n_agents", "alpha"]).items()):
        pts = []
        for tp in taus:
            v = [r["throughput"] for r in rs
                 if r["method"] == "ph_ema" and r["tau_phi"] == tp]
            if v and (n, a) in base:
                pts.append((tp, (st.mean(v) / base[(n, a)] - 1) * 100))
        if pts:
            series[(n, a)] = pts

    allv = [p[1] for v in series.values() for p in v] or [0]
    ymin, ymax = min(allv + [0]) - 0.6, max(allv + [0]) + 0.6

    def px(tp):
        i = taus.index(tp)
        return x0 + plot_w * (i + 0.5) / len(taus)

    def py(v):
        return y0 - plot_h * (v - ymin) / max(ymax - ymin, 1e-9)

    for frac in (0, 0.5, 1.0):
        v = ymin + frac * (ymax - ymin)
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#F1EFE8"/>'
                 % (x0, py(v), x0 + plot_w, py(v)))
        s.append('<text x="%d" y="%.1f" font-size="11" fill="#888780" '
                 'text-anchor="end" dominant-baseline="central">%+.1f%%</text>'
                 % (x0 - 8, py(v), v))
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#D3D1C7" '
             'stroke-dasharray="4 3"/>' % (x0, py(0), x0 + plot_w, py(0)))

    pal = ["#0F6E56", "#185FA5", "#BA7517", "#534AB7", "#D85A30", "#993556",
           "#3B6D11", "#0C447C", "#854F0B"]
    for ci, ((n, a), pts) in enumerate(sorted(series.items())):
        col = pal[ci % len(pal)]
        d = " ".join("%s%.0f,%.1f" % ("M" if i == 0 else "L", px(t), py(v))
                     for i, (t, v) in enumerate(pts))
        s.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.8"/>' % (d, col))
        for (t, v) in pts:
            s.append('<circle cx="%.0f" cy="%.1f" r="3" fill="%s"/>' % (px(t), py(v), col))
        s.append('<text x="%d" y="%d" font-size="11" fill="%s">|A|=%d a=%.1f</text>'
                 % (x0 + plot_w + 6, 80 + ci * 18, col, n, a))

    for i, tp in enumerate(taus):
        s.append('<text x="%.0f" y="%d" font-size="11" fill="#5F5E5A" '
                 'text-anchor="middle">%d</text>' % (px(tp), y0 + 18, tp))
    s.append('<text x="%d" y="%d" font-size="11" fill="#888780" '
             'text-anchor="middle">tau_phi (timesteps)</text>'
             % (x0 + plot_w / 2, y0 + 40))
    s.append('<text x="%d" y="%d" font-size="11" fill="#888780">'
             'gain vs Hungarian baseline; dashed = parity</text>' % (x0, y0 + 62))
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(s))
    return path


def fig_alpha(sumry, path=None):
    """增益 vs 空间不均匀度 α。"""
    path = path or os.path.join(RES, "fig_alpha.svg")
    W, H = 620, 340
    x0, y0 = 80, 250
    plot_w, plot_h = 480, 180
    s = [_svg_header(W, H)]
    s.append('<text x="20" y="26" font-size="15" font-weight="500" fill="#2C2C2A">'
             'Pheromone gain vs spatial skew (Zipf alpha)</text>')
    s.append(_axes(x0, y0, x0 + plot_w, y0))
    s.append(_axes(x0, y0, x0, y0 - plot_h))

    alphas = sorted({x["alpha"] for x in sumry})
    rows_ph = [x for x in sumry if x["method"] == "ph_ema"]
    pts = []
    for a in alphas:
        v = [x["gain_pct"] for x in rows_ph if x["alpha"] == a]
        if v:
            pts.append((a, st.mean(v)))
    allv = [p[1] for p in pts] or [0]
    ymin, ymax = min(allv + [0]) - 0.5, max(allv + [0]) + 0.5

    def px(a):
        return x0 + plot_w * (alphas.index(a) + 0.5) / len(alphas)

    def py(v):
        return y0 - plot_h * (v - ymin) / max(ymax - ymin, 1e-9)

    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#D3D1C7" '
             'stroke-dasharray="4 3"/>' % (x0, py(0), x0 + plot_w, py(0)))
    d = " ".join("%s%.0f,%.1f" % ("M" if i == 0 else "L", px(a), py(v))
                 for i, (a, v) in enumerate(pts))
    s.append('<path d="%s" fill="none" stroke="#0F6E56" stroke-width="2"/>' % d)
    for (a, v) in pts:
        s.append('<circle cx="%.0f" cy="%.1f" r="4" fill="#0F6E56"/>' % (px(a), py(v)))
        s.append('<text x="%.0f" y="%.1f" font-size="11" fill="#0F6E56" '
                 'text-anchor="middle">%+.2f%%</text>' % (px(a), py(v) - 14, v))
    for a in alphas:
        s.append('<text x="%.0f" y="%d" font-size="11" fill="#5F5E5A" '
                 'text-anchor="middle">%.1f</text>' % (px(a), y0 + 18, a))
    s.append('<text x="%d" y="%d" font-size="11" fill="#888780" '
             'text-anchor="middle">Zipf alpha (0 = uniform task distribution)</text>'
             % (x0 + plot_w / 2, y0 + 40))
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(s))
    return path


def main():
    rows = load()
    print("loaded %d rows" % len(rows))
    sumry = summarize(rows)
    for p in (write_summary_csv(sumry), write_summary_md(sumry),
              fig_main(sumry), fig_tau(rows), fig_alpha(sumry)):
        print("wrote", os.path.abspath(p))
    print()
    for (n, a), items in sorted(group(sumry, ["n_agents", "alpha"]).items()):
        line = "|A|=%2d a=%.1f : " % (n, a)
        line += "  ".join("%s=%+.2f%%" % (i["method"], i["gain_pct"]) for i in items)
        print(line)


if __name__ == "__main__":
    main()
