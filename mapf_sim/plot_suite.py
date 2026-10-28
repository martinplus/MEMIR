"""从 results_final/raw_*.csv 生成论文插图（纯手写 SVG，零依赖）。

图清单：
    fig1_fleet.svg      fleet 规模 × service time（主图，含 U 型 / 适用区间）
    fig2_liveness.svg   liveness 守护消融（plan-attempt-fail 与服务时间对照）
    fig3_tau.svg        记忆时间常数 τ 与元适应
    fig4_hetero.svg     需求异质性 α × 到达强度 λ
    fig5_drift.svg      非平稳漂移
    fig6_maps.svg       地图泛化 + 可扩展性
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_suite import load, pick, series, _sd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_final", "figs")
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "nearest": "#94a3b8",
    "random": "#a78bfa",
    "static": "#f59e0b",
    "memir": "#dc2626",
    "oracle": "#0ea5e9",
    "memir_meta": "#166534",
}


def _svg(w, h, title, xlab, ylab):
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
         f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
         f'<text x="{w/2}" y="26" font-size="15" font-weight="600" '
         f'fill="#1f2a44" text-anchor="middle">{title}</text>',
         f'<text x="{w/2}" y="{h-16}" font-size="11.5" fill="#475569" '
         f'text-anchor="middle">{xlab}</text>',
         f'<text x="20" y="{h/2}" font-size="11.5" fill="#475569" '
         f'transform="rotate(-90 20 {h/2})" text-anchor="middle">{ylab}</text>']
    return p


def _frame(p, box, ys, ylabels, xs, xlabels):
    L, R, T, B = box
    ymin, ymax = ys
    W = None

    def py(v):
        v = min(max(v, ymin), ymax)
        return None
    return None


def line_plot(path, title, xlab, ylab, xlabels, series_dict, ymin, ymax,
              w=760, h=430, dash=("static", "oracle")):
    L, R, T, B = 70, 150, 52, 62
    n = len(xlabels)

    def px(i):
        return L + (w - L - R) * i / max(n - 1, 1)

    def py(v):
        v = min(max(v, ymin), ymax)
        return h - B - (h - B - T) * (v - ymin) / (ymax - ymin)

    p = _svg(w, h, title, xlab, ylab)
    for yy in _yticks(ymin, ymax):
        yv = py(yy)
        p.append(f'<line x1="{L}" y1="{yv:.1f}" x2="{w-R}" y2="{yv:.1f}" '
                 f'stroke="#eef2f7" stroke-width="1"/>')
        p.append(f'<text x="{L-8}" y="{yv+4:.1f}" font-size="11" fill="#6b7a99" '
                 f'text-anchor="end">{yy:g}</text>')
    p.append(f'<line x1="{L}" y1="{py(ymin):.1f}" x2="{w-R}" y2="{py(ymin):.1f}" '
             f'stroke="#cbd5e1"/>')
    for i, lab in enumerate(xlabels):
        p.append(f'<text x="{px(i):.1f}" y="{h-B+20}" font-size="11" '
                 f'fill="#6b7a99" text-anchor="middle">{lab}</text>')
    for name, vals in series_dict.items():
        pts = " ".join("%.1f,%.1f" % (px(i), py(v))
                       for i, v in enumerate(vals) if v is not None)
        col = PALETTE.get(name, "#334155")
        dsh = ' stroke-dasharray="6 4"' if name in dash else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                 f'stroke-width="2.6"{dsh}/>')
        for i, v in enumerate(vals):
            if v is None:
                continue
            p.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.6" '
                     f'fill="{col}"/>')
    ly = T + 4
    for name in series_dict:
        col = PALETTE.get(name, "#334155")
        dsh = ' stroke-dasharray="6 4"' if name in dash else ""
        p.append(f'<line x1="{w-R+16}" y1="{ly}" x2="{w-R+42}" y2="{ly}" '
                 f'stroke="{col}" stroke-width="2.6"{dsh}/>')
        p.append(f'<text x="{w-R+48}" y="{ly+4}" font-size="11" '
                 f'fill="#334155">{name}</text>')
        ly += 19
    p.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    print("wrote", path)


def _yticks(ymin, ymax, step=None):
    if step is None:
        span = ymax - ymin
        step = 5 if span <= 40 else 10 if span <= 100 else 25
    out = []
    v = int(ymin // step) * step
    while v <= ymax:
        if v >= ymin:
            out.append(v)
        v += step
    return out


def fig1_fleet(rows):
    ns = sorted({r["n_agents"] for r in rows})
    series_dict = {}
    for pol in ("nearest", "random", "static", "memir", "oracle"):
        vals = []
        for n in ns:
            rr = pick(rows, n_agents=n, method=pol, map="W1")
            vals.append(sum(series(rr)) / len(rr) if rr else None)
        series_dict[pol] = vals
    line_plot(os.path.join(OUT, "fig1_fleet.svg"),
              "空闲智能体重定位：service time 随 fleet 规模的变化",
              "fleet size  |A|（λ=0.08/timestep 固定，throughput 全程≈78/1000 steps）",
              "service time（timestep，越低越好）",
              [str(n) for n in ns], series_dict, 30, 165)


def fig3_tau(rows):
    for n in sorted({r["n_agents"] for r in rows}):
        meths = [m for m in {r["method"] for r in rows if r["n_agents"] == n}]
        ordered = [m for m in meths if m.startswith("memir_t")]
        ordered.sort(key=lambda m: float(m.split("_t")[1]))
        labels, vals = [], []
        for m in ordered + ["memir_meta"]:
            rr = pick(rows, n_agents=n, method=m)
            if rr:
                labels.append(m.replace("memir_", ""))
                vals.append(sum(series(rr)) / len(rr))
        rr = pick(rows, n_agents=n, method="nearest")
        if rr:
            labels.insert(0, "nearest")
            vals.insert(0, sum(series(rr)) / len(rr))
        if not vals:
            continue
        vs = vals[1:] if labels[0] == "nearest" else vals
        line_plot(os.path.join(OUT, "fig3_tau_%d.svg" % n),
                  "记忆时间常数 τ 的非单调最优（|A|=%d）" % n,
                  "记忆时间常数 τ（越大 = 忘得越慢）",
                  "service time（timestep，越低越好）",
                  labels, {"memir": vs}, min(vs) - 3, max(vs) + 3,
                  dash=())


def fig4_hetero(rows):
    alphas = sorted({r["zipf_alpha"] for r in rows})
    for lam in sorted({r["lmbda"] for r in rows}):
        for n in sorted({r["n_agents"] for r in rows}):
            series_dict = {}
            for pol in ("nearest", "static", "memir"):
                vals = []
                for a in alphas:
                    rr = [r for r in rows if r["n_agents"] == n
                          and r["method"] == pol
                          and abs(r["zipf_alpha"] - a) < 1e-9
                          and abs(r["lmbda"] - lam) < 1e-9]
                    vals.append(None if not rr else
                               sum(x["service_post"] for x in rr) / len(rr))
                if any(v is not None for v in vals):
                    series_dict[pol] = vals
            if not series_dict:
                continue
            flat = [v for vs in series_dict.values() for v in vs if v is not None]
            line_plot(os.path.join(OUT, "fig4_hetero_l%g_n%d.svg" % (lam, n)),
                      "需求异质性 Zipf α（λ=%g, |A|=%d）" % (lam, n),
                      "Zipf 指数 α（0 = 均匀，越大越集中）",
                      "service time（timestep，越低越好）",
                      [str(a) for a in alphas], series_dict,
                      max(min(flat) - 6, 0), max(flat) + 6)


EXPS = {"A_main": fig1_fleet, "D_tau": fig3_tau, "E_hetero": fig4_hetero}


def main():
    for name, fn in EXPS.items():
        rows = load(name)
        if not rows:
            print("no data", name)
            continue
        fn(rows)


if __name__ == "__main__":
    main()
