"""从已有 E2 数据画「fleet size × service time」曲线 —— 方法型论文的主图候选。

要点（无需新实验）：
  (a) nearest 随舰队规模迅速饱和（8→32 只从 56.9 降到 54.9），永远追不上 ACO 的 38.4
  (b) iso-service 车队缩减：ACO 用 6 个 agent 达到 49.8，而 nearest 用 32 个仍只有 54.9
  (c) 交叉点：约 |A| ≈ 20 之后 static 反超 ACO（← 待 diag_rest6 判定是机制还是 artifact）
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "results_open", "summary.csv")
OUT = os.path.join(ROOT, "results_open", "fig_fleet_curves.svg")

NS = (4, 6, 8, 12, 16, 24, 32)
SERIES = [
    ("nearest", "就近泊位（默认行为）", "#94a3b8"),
    ("static", "固定泊位均匀分布", "#f59e0b"),
    ("aco", "本文方法（带遗忘的外生记忆）", "#dc2626"),
]


def load():
    d = {}
    with open(RAW, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] != "E2_fleet":
                continue
            m = row["method"]
            for key in ("nearest", "static", "aco"):
                if m.startswith(key + "_n"):
                    d[(key, int(m.split("_n")[1]))] = float(row["service_mean"])
    return d


def main():
    d = load()
    W, H = 800, 470
    L, R, T, B = 88, 40, 60, 72
    xs = list(range(len(NS)))
    ymin, ymax = 30.0, 180.0

    def px(i):
        return L + (W - L - R) * i / (len(NS) - 1)

    def py(v):
        v = min(max(v, ymin), ymax)
        return H - B - (H - B - T) * (v - ymin) / (ymax - ymin)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2}" y="26" font-size="15" font-weight="600" fill="#1f2a44" '
         f'text-anchor="middle">车队规模买不到记忆提供的东西：service time 随 |A| 变化</text>']

    # grid + y ticks
    for v in range(30, 181, 25):
        y = py(v)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" '
                 f'stroke="#eef2f7" stroke-width="1"/>')
        p.append(f'<text x="{L-10}" y="{y+4:.1f}" font-size="11" fill="#6b7a99" '
                 f'text-anchor="end">{v}</text>')
    p.append(f'<line x1="{L}" y1="{py(ymin):.1f}" x2="{W-R}" y2="{py(ymin):.1f}" '
             f'stroke="#cbd5e1" stroke-width="1"/>')
    p.append(f'<text x="20" y="{H/2}" font-size="11.5" fill="#475569" '
             f'transform="rotate(-90 20 {H/2})" text-anchor="middle">'
             f'service time（timestep，越低越好）</text>')
    p.append(f'<text x="{W/2}" y="{H-38}" font-size="11.5" fill="#475569" '
             f'text-anchor="middle">fleet size |A|（λ 固定 = 0.08/timestep，故 throughput ≈ 78 全程不变）</text>')

    # x labels
    for i, n in enumerate(NS):
        p.append(f'<text x="{px(i):.1f}" y="{H-B+20}" font-size="11" fill="#6b7a99" '
                 f'text-anchor="middle">{n}</text>')

    # curves
    for key, label, color in SERIES:
        pts = " ".join(f"{px(i):.1f},{py(d[(key, n)]):.1f}" for i, n in enumerate(NS))
        dash = ' stroke-dasharray="6 4"' if key == "static" else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2.6"{dash}/>')
        for i, n in enumerate(NS):
            p.append(f'<circle cx="{px(i):.1f}" cy="{py(d[(key, n)]):.1f}" r="4" '
                     f'fill="{color}"/>')

    # reference band: nearest best achievable
    nb = min(d[("nearest", n)] for n in NS)
    p.append(f'<line x1="{L}" y1="{py(nb):.1f}" x2="{W-R}" y2="{py(nb):.1f}" '
             f'stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 4"/>')
    p.append(f'<text x="{W-R-4}" y="{py(nb)-7:.1f}" font-size="10.5" fill="#64748b" '
             f'text-anchor="end">nearest 在任意规模下的最好水平 {nb:.1f}（即使 |A|=32）</text>')

    # annotation (a): ACO@8 below nearest floor
    x8, y8 = px(2), py(d[("aco", 8)])
    p.append(f'<text x="{x8+10:.1f}" y="{y8-8:.1f}" font-size="10.5" fill="#b91c1c">'
             f'本文方法 |A|=8 → {d[("aco",8)]:.1f}，比 nearest 的最好水平还低 '
             f'{100*(d[("aco",8)]-nb)/nb:.0f}%</text>')

    # annotation (b): iso-service
    x6, y6 = px(1), py(d[("aco", 6)])
    x32, y32 = px(6), py(d[("nearest", 32)])
    p.append(f'<line x1="{x6:.1f}" y1="{y6:.1f}" x2="{x32:.1f}" y2="{y32:.1f}" '
             f'stroke="#16a34a" stroke-width="1.2" stroke-dasharray="4 4"/>')
    p.append(f'<text x="{(x6+x32)/2:.1f}" y="{(y6+y32)/2-8:.1f}" font-size="10.5" '
             f'fill="#15803d" text-anchor="middle">同等 service time ≈ 50：6 个 vs 32 个 agent</text>')

    # annotation (c): crossover
    xc = px(5)
    p.append(f'<line x1="{xc:.1f}" y1="{py(d[("aco",24)]):.1f}" x2="{xc:.1f}" '
             f'y2="{py(d[("static",24)]):.1f}" stroke="#64748b" stroke-width="1"/>')
    p.append(f'<text x="{xc+8:.1f}" y="{py(150):.1f}" font-size="10.5" fill="#64748b">'
             f'|A| ≥ 24 后本文方法反被超越</text>')
    p.append(f'<text x="{xc+8:.1f}" y="{py(150)+15:.1f}" font-size="10.5" fill="#64748b">'
             f'已诊断为 liveness 失效（泊位</text>')
    p.append(f'<text x="{xc+8:.1f}" y="{py(150)+30:.1f}" font-size="10.5" fill="#64748b">'
             f'永久占位→规划失败级联），可修</text>')

    # legend
    lx, ly = L + 6, T + 6
    for key, label, color in SERIES:
        dash = ' stroke-dasharray="6 4"' if key == "static" else ""
        p.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" '
                 f'stroke-width="2.6"{dash}/>')
        p.append(f'<text x="{lx+34}" y="{ly+4}" font-size="11" fill="#334155">{label}</text>')
        ly += 19

    p.append("</svg>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    print("wrote", OUT)
    for key, _, _ in SERIES:
        print(key, [round(d[(key, n)], 2) for n in NS])


if __name__ == "__main__":
    main()
