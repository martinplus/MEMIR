"""论文正式插图（Fig.1 – Fig.5）。手写 SVG，零第三方依赖。

    plot_paper.py                生成全部
    plot_paper.py fig1 fig2      只生成指定图

产出目录 results_final/figs/
    fig1_fleet.svg    双面板：全量程（含 |A|=4 饱和悬崖）+ |A|>=6 放大
    fig2_legs.svg     ★ L1/L2/L3 堆叠柱 —— 全文最重要的一张图
    fig3_psi.svg      ψ 的内部最优与 ψ* 迁移
    fig4_hetero.svg   Zipf α × λ 增益热力图
    fig5_map.svg      地图俯视：flow 热力 + 三种策略的驻留分布
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_suite import load, pick, _avg, compare, _sd   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_final", "figs")
os.makedirs(OUT, exist_ok=True)

FONT = ('"Segoe UI","PingFang SC","Microsoft YaHei","Helvetica Neue",'
        'system-ui,sans-serif')
INK = "#1e293b"
SUB = "#64748b"
GRID = "#e8edf4"
AXIS = "#cbd5e1"

C = {
    "nearest": "#94a3b8",
    "random": "#a78bfa",
    "static": "#475569",
    "memir0": "#2563eb",
    "memir": "#dc2626",
    "oracle": "#0891b2",
    # ---- 2026-09-26 新增基线 ----
    "pmedian": "#059669",        # greedy 设施选址（运筹学对照）
    "popsample": "#d97706",      # 需求比例采样（AMoD 对照）
    "attn": "#7c3aed",           # 注意力指针网络
    "rl_lin": "#db2777",         # REINFORCE 线性打分器
    "rl_tab": "#65a30d",         # 上下文 bandit
}
DASH = {"memir0": "7 4", "oracle": "9 4 2 4",
        "pmedian": "7 4", "popsample": "2 3",
        "rl_tab": "7 4", "rl_lin": "2 3"}
WIDTH = {"memir": 3.0, "static": 2.6, "memir0": 2.4, "oracle": 2.0,
         "nearest": 1.9, "random": 1.9,
         "pmedian": 2.4, "popsample": 2.0,
         "attn": 2.4, "rl_lin": 2.2, "rl_tab": 2.0}
LEG = {"nearest": "nearest (TP 原生 / 无预期)", "random": "random",
       "static": "static 均匀分层（对照）", "memir0": "MEMIR ψ=0",
       "memir": "MEMIR ψ=30", "oracle": "oracle 事后真值",
       "pmedian": "p-median 贪心设施选址（同 p̂/β/ψ）",
       "pmedian0": "p-median ψ=0",
       "popsample": "popsample 需求比例采样（AMoD）",
       "popsample_f": "popsample + ψ",
       "attn": "attn 注意力指针网络（AM）",
       "attn_bc": "attn 蒸馏解析解",
       "attn_bc_warm": "attn 蒸馏（长训练窗口）",
       "rl_lin": "rl_lin REINFORCE 线性打分器",
       "rl_tab": "rl_tab 上下文 bandit（锚点）",
       "attn_warm": "attn（长训练窗口）",
       "rl_lin_warm": "rl_lin（长训练窗口）"}


# ============================ 通用 SVG 基元 ============================
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def head(P, w, h, title):
    P.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
             "font-family='%s'>" % (w, h, FONT))
    P.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (w, h))
    P.append('<text x="%.1f" y="26" font-size="15" font-weight="600" '
             'fill="%s" text-anchor="middle">%s</text>'
             % (w / 2.0, INK, esc(title)))


def txt(P, x, y, s, size=11, fill=SUB, anchor="middle", weight="400",
        rot=None):
    if rot:
        P.append('<text x="%.1f" y="%.1f" font-size="%s" fill="%s" '
                 'text-anchor="%s" font-weight="%s" transform="rotate(%s '
                 '%.1f %.1f)">%s</text>'
                 % (x, y, size, fill, anchor, weight, rot, x, y, esc(s)))
    else:
        P.append('<text x="%.1f" y="%.1f" font-size="%s" fill="%s" '
                 'text-anchor="%s" font-weight="%s">%s</text>'
                 % (x, y, size, fill, anchor, weight, esc(s)))


def box(P, x, y, w, h, fill="none", stroke=AXIS, rw=1.0, op=None):
    o = ' opacity="%s"' % op if op else ""
    P.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
             'fill="%s" stroke="%s" stroke-width="%.1f"%s/>'
             % (x, y, w, h, fill, stroke, rw, o))


def poly(P, pts, stroke, w=2.0, dash=None, fill="none"):
    d = ' stroke-dasharray="%s"' % dash if dash else ""
    P.append('<polyline points="%s" fill="%s" stroke="%s" '
             'stroke-width="%.1f"%s stroke-linejoin="round"/>'
             % (" ".join("%.1f,%.1f" % p for p in pts), fill, stroke, w, d))


def dot(P, x, y, r, fill, stroke="none"):
    P.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" '
             'stroke="%s" stroke-width="1"/>' % (x, y, r, fill, stroke))


def save(P, path):
    P.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(P))
    print("wrote", path)


def ticks(lo, hi, want=5):
    raw = (hi - lo) / max(want, 1)
    for m in (1, 2, 2.5, 5, 10):
        step = m * (10 ** _log10(raw))
        if (hi - lo) / step <= want + 1.5:
            break
    out, v = [], _floor(lo, step)
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            out.append(round(v, 6))
        v += step
    return out


def _log10(x):
    import math
    return math.floor(math.log10(max(x, 1e-12)))


def _floor(v, step):
    import math
    return math.floor(v / step) * step


def _tw(s, size):
    """粗略估算文本像素宽度：CJK ≈ 1.0em，拉丁 ≈ 0.55em。"""
    n_cjk = sum(1 for ch in s if ord(ch) > 0x2E80)
    return size * (n_cjk + 0.55 * (len(s) - n_cjk))


def legend(P, x, y, names, cols=1, gap=17.0, lw=20.0):
    for i, nm in enumerate(names):
        cy = y + i * gap
        poly(P, [(x, cy), (x + lw, cy)], C.get(nm, INK),
             WIDTH.get(nm, 2.0), DASH.get(nm))
        txt(P, x + lw + 7, cy + 4, LEG.get(nm, nm), 11, INK, "start")


def hlegend(P, x, y, names, maxw, gap=26.0, lw=20.0):
    """水平图例：按估算宽度自动换行，保证不越出画布。"""
    cx, cy = x, y
    for nm in names:
        label = LEG.get(nm, nm)
        itemw = _tw(label, 11) + lw + 7 + gap
        if cx + itemw - gap > x + maxw and cx > x:
            cx, cy = x, cy + 19
        poly(P, [(cx, cy), (cx + lw, cy)], C.get(nm, INK),
             WIDTH.get(nm, 2.0), DASH.get(nm))
        txt(P, cx + lw + 7, cy + 4, label, 11, INK, "start")
        cx += itemw


# ============================ 通用折线面板 ============================
def panel_line(P, bx, xlabels, data, ylim, title, xlab, ylab, show_leg=True,
               mark=None, note=None):
    """bx = (x0, y0, w, h) 为绘图矩形。"""
    x0, y0, w, h = bx
    ymin, ymax = ylim
    n = len(xlabels)

    def px(i):
        return x0 + (w * i / max(n - 1, 1))

    def py(v):
        v = min(max(v, ymin), ymax)
        return y0 + h - h * (v - ymin) / (ymax - ymin)

    txt(P, x0 + w / 2, y0 - 8, title, 12, INK, "middle", "600")
    for v in ticks(ymin, ymax):
        yy = py(v)
        P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                 'stroke="%s" stroke-width="1"/>' % (x0, yy, x0 + w, yy, GRID))
        txt(P, x0 - 7, yy + 4, "%g" % v, 10.5, SUB, "end")
    for i, lb in enumerate(xlabels):
        txt(P, px(i), y0 + h + 17, lb, 10.5, SUB)
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, y0 + h, x0 + w, y0 + h, AXIS))
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, y0, x0, y0 + h, AXIS))
    for nm, vals in data.items():
        pts = [(px(i), py(v)) for i, v in enumerate(vals) if v is not None]
        poly(P, pts, C.get(nm, INK), WIDTH.get(nm, 2.0), DASH.get(nm))
        for p in pts:
            dot(P, p[0], p[1], 3.2, C.get(nm, INK))
    if mark:
        mi, label, dy = mark
        P.append('<circle cx="%.1f" cy="%.1f" r="7" fill="none" '
                 'stroke="%s" stroke-width="1.6" opacity="0.75"/>'
                 % (px(mi), py(data["memir"][mi]), C["memir"]))
        txt(P, px(mi), py(data["memir"][mi]) + dy, label, 10.5, C["memir"],
            "middle", "600")
    if note:
        txt(P, x0 + w - 4, y0 + 14, note, 10.5, SUB, "end")
    txt(P, x0 + w / 2, y0 + h + 36, xlab, 11, SUB)
    txt(P, x0 - 42, y0 + h / 2, ylab, 11, SUB, "middle", "400", "-90")
    if show_leg:
        legend(P, x0 + w + 12, y0 + 6, list(data))


# ============================ Fig.1 ============================
def fig1():
    rows = load("A_main")
    ns = sorted({r["n_agents"] for r in rows})
    pols = ["nearest", "random", "static", "memir0", "memir", "oracle"]
    full = {}
    zoom = {}
    zn = [n for n in ns if n >= 6]
    for p in pols:
        full[p] = []
        zoom[p] = []
        for n in ns:
            rr = pick(rows, n_agents=n, method=p, map="W1")
            v = sum(r["service_post"] for r in rr) / len(rr) if rr else None
            full[p].append(v)
        for n in zn:
            rr = pick(rows, n_agents=n, method=p, map="W1")
            zoom[p].append(sum(r["service_post"] for r in rr) / len(rr)
                           if rr else None)

    P, W, H = [], 1000, 500
    head(P, W, H, "Fig.1  MEMIR 相对各基线的 service time：随 fleet 规模的"
                  "完整扫描（W1, λ=0.08, α=2.0, 10 seeds）")
    panel_line(P, (78, 62, 470, 300),
               [str(n) for n in ns], full, (0, 165),
               "(a) 全量程：|A|=4 处于饱和区（idle≈1%），所有策略退化到 ≈152",
               "", "", show_leg=False)
    panel_line(P, (620, 62, 250, 300),
               [str(n) for n in zn], zoom, (30, 56),
               "(b) 放大：|A|≥6 的可调度区间", "", "", show_leg=False,
               mark=(zn.index(12), "峰值 −26.2%", -14),
               note="ψ=0 在 |A|≥32 反超为负收益")
    txt(P, 78 + 470 / 2, 424, "fleet size  |A|（open-loop 泊松到达，"
        "throughput 全程恒定 ≈78 任务/1000 timesteps）", 11, SUB)
    txt(P, 40, 212, "service time（timestep，越低越好）", 11, SUB, "middle",
        "400", "-90")
    hlegend(P, 78, 452, pols, 860)
    txt(P, W / 2, 486,
        "全部策略共用同一 open-loop 任务流（common random numbers）；"
        "throughput 恒定，故 service time 的差异完全来自「空闲时站在哪」。",
        11.5, INK, "middle")
    save(P, os.path.join(OUT, "fig1_fleet.svg"))


# ============================ Fig.2  ★ 核心图 ============================
L1C, L2C, L3C = "#e2e8f0", "#3b82f6", "#ef4444"


def fig2():
    rows = load("A_main")
    ns = [12, 16, 24, 32, 40]
    pols = ["static", "memir0", "memir"]
    dat = {}
    for n in ns:
        for p in pols:
            rr = pick(rows, n_agents=n, method=p, map="W1")
            dat[(n, p)] = dict(L1=_avg(rr, "leg_assign"),
                               L2=_avg(rr, "leg_approach"),
                               L3=_avg(rr, "leg_haul"))

    P, W, H = [], 900, 470
    head(P, W, H, "Fig.2  干扰在哪里诞生的，修复就在哪里：service time 的"
                  "三段时延分解（L1 等派单 / L2 到位取货 / L3 取货到入站）")
    x0, y0, w, h = 86, 78, 700, 300
    ymax = 58.0
    groups = len(ns)
    bw = w / groups
    barw = bw * 0.20
    offs = {"static": -bw * 0.24, "memir0": 0.0, "memir": bw * 0.24}

    def py(v):
        return y0 + h - h * v / ymax

    for v in range(0, 59, 10):
        yy = py(v)
        P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                 'stroke="%s"/>' % (x0, yy, x0 + w, yy, GRID))
        txt(P, x0 - 8, yy + 4, str(v), 10.5, SUB, "end")
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, py(0), x0 + w, py(0), AXIS))
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, y0, x0, y0 + h, AXIS))

    for gi, n in enumerate(ns):
        cx = x0 + bw * (gi + 0.5)
        for p in pols:
            d = dat[(n, p)]
            bx = cx + offs[p] - barw / 2
            cur = 0.0
            for key, col in (("L1", L1C), ("L2", L2C), ("L3", L3C)):
                dv = d[key]
                ytop = py(cur + dv)
                ybot = py(cur)
                if key == "L3":
                    box(P, bx, ytop, barw, ybot - ytop, col, "#ffffff", 0.8)
                    # ψ 修掉的那一包用虚线描边强调
                    P.append('<rect x="%.1f" y="%.1f" width="%.1f" '
                             'height="%.1f" fill="none" stroke="%s" '
                             'stroke-width="1.4" stroke-dasharray="3 2"/>'
                             % (bx, ytop, barw, ybot - ytop, "#991b1b"))
                else:
                    box(P, bx, ytop, barw, ybot - ytop, col, "#ffffff", 0.8)
                if dv > 2.4:
                    txt(P, bx + barw / 2, (ytop + ybot) / 2 + 3.5,
                        "%.1f" % dv, 9.5,
                        "#1e293b" if key != "L1" else "#64748b", "middle",
                        "600")
                cur += dv
            txt(P, bx + barw / 2, ybot + 14, p.replace("memir0", "ψ=0")
                .replace("memir", "ψ=30"), 9.5, SUB)
            txt(P, bx + barw / 2, py(cur) - 6, "%.1f" % cur, 10.5, INK,
                "middle", "600")
        txt(P, cx, y0 + h + 32, "|A|=%d" % n, 12, INK, "middle", "600")
        # 组内注释：L3 的膨胀与回吐
        ge = dat[(n, "memir0")]["L3"] - dat[(n, "static")]["L3"]
        gfix = dat[(n, "memir0")]["L3"] - dat[(n, "memir")]["L3"]
        txt(P, cx, y0 + h + 48, "L3  +%.1f → ψ 修回 −%.1f" % (ge, gfix),
            9.5, "#b91c1c", "middle")

    # 图例
    lx, ly = x0 + 8, y0 + 8
    for nm, col in (("L1 等派单（release→assign）", L1C),
                    ("L2 到位取货（assign→pickup）—— 记忆的收益来源", L2C),
                    ("L3 取货到入站（pickup→delivery）—— 驻留干扰的代价",
                     L3C)):
        box(P, lx, ly, 13, 13, col, "#94a3b8", 0.8)
        txt(P, lx + 19, ly + 11, nm, 10.5, INK, "start")
        ly += 19

    txt(P, x0 + w / 2, y0 + h + 68,
        "fleet size  |A|（同一 open-loop 任务流，common random numbers，"
        "10 seeds 平均；残差 Σ = service time，误差 < 0.01）", 11, SUB)
    txt(P, 34, y0 + h / 2, "service time（timestep）", 11, SUB, "middle",
        "400", "-90")
    txt(P, W / 2, 388,
        "ψ=0 把 L2 从 17.4 压到 5.1（|A|=12），但 L3 随 fleet 单调膨胀 "
        "29.8 → 34.1：簇聚的空闲 agent 正好压在 pickup→station 的搬运干道上。",
        11.5, INK, "middle", "600")
    txt(P, W / 2, 408,
        "ψ=30 把 L3 压回基线（≈29.8–30.4），几乎不动 L2 —— 增益因此得以净实现，"
        "且在全部 fleet 规模上同时优于 static 与 oracle。", 11.5, INK, "middle")
    save(P, os.path.join(OUT, "fig2_legs.svg"))


# ============================ Fig.3 ============================
def fig3():
    rows = load("H_psi")
    P, W, H = [], 880, 470
    head(P, W, H, "Fig.3  驻留干扰避让强度 ψ 存在内部最优，且最优值随 fleet "
                  "规模迁移（W1, λ=0.08, α=2.0, 6 seeds）")
    x0, y0, w, h = 82, 74, 640, 260
    psis = [0, 10, 20, 30, 40, 80, 160]
    ns = sorted({r["n_agents"] for r in rows})
    cols = {"6": "#7c3aed", "8": "#2563eb", "16": "#dc2626", "32": "#ea580c"}
    dash = {"32": "8 4"}
    base = {}
    for n in ns:
        rr = pick(rows, n_agents=n, method="static")
        base[n] = sum(r["service_post"] for r in rr) / len(rr) if rr else None
    vals = {}
    for n in ns:
        vs = []
        for psi in psis:
            rr = pick(rows, n_agents=n, method="memir_psi%g" % psi)
            vs.append(sum(r["service_post"] for r in rr) / len(rr)
                      if rr else None)
        vals[n] = vs
    flat = [min(v for v in vs if v is not None) for vs in vals.values()]
    flat += [v for v in base.values() if v is not None]
    ymin, ymax = max(min(flat) - 4, 0), max(flat) + 4

    def px(i):
        return x0 + w * i / (len(psis) - 1)

    def py(v):
        return y0 + h - h * (v - ymin) / (ymax - ymin)

    for v in ticks(ymin, ymax, 6):
        yy = py(v)
        P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
                 % (x0, yy, x0 + w, yy, GRID))
        txt(P, x0 - 7, yy + 4, "%g" % v, 10.5, SUB, "end")
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, py(ymin), x0 + w, py(ymin), AXIS))
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, y0, x0, y0 + h, AXIS))
    for i, p in enumerate(psis):
        txt(P, px(i), y0 + h + 17, str(p), 10.5, SUB)

    for n in ns:
        pts = [(px(i), py(v)) for i, v in enumerate(vals[n]) if v is not None]
        poly(P, pts, cols[str(n)], 2.6, dash.get(str(n)))
        for q in pts:
            dot(P, q[0], q[1], 3.2, cols[str(n)])
        bi = min(range(len(psis)), key=lambda i: vals[n][i])
        bx, by = px(bi), py(vals[n][bi])
        P.append('<circle cx="%.1f" cy="%.1f" r="7.5" fill="none" '
                 'stroke="%s" stroke-width="1.6" opacity="0.8"/>'
                 % (bx, by, cols[str(n)]))
        txt(P, bx, by - 13, "ψ*=%d" % psis[bi], 10, cols[str(n)], "middle",
            "600")
        # static 参考线
        P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                 'stroke="%s" stroke-width="1.2" stroke-dasharray="2 3" '
                 'opacity="0.85"/>' % (x0, py(base[n]), x0 + w, py(base[n]),
                                       cols[str(n)]))

    hl = []
    for n in ns:
        hl.append(("|A|=%d" % n, cols[str(n)], dash.get(str(n)),
                   "static %.1f" % base[n]))
    # 静态图例：最多两个一行，保证不越出画布
    cx2, cy2 = x0, y0 + h + 58
    for i, (lab, col, dsh, sub) in enumerate(hl):
        if i == 2:
            cx2, cy2 = x0, cy2 + 21
        poly(P, [(cx2, cy2), (cx2 + 22, cy2)], col, 2.6, dsh)
        txt(P, cx2 + 28, cy2 + 4, lab, 10.5, INK, "start")
        txt(P, cx2 + 78, cy2 + 4, "虚线 = 该规模 static 基线 (%s)" % sub,
            10.5, SUB, "start")
        cx2 += 300
    txt(P, x0 + w / 2, y0 + h + 120,
        "驻留干扰避让强度 ψ（加性惩罚系数；虚线为该规模的 static 基线）",
        11, SUB)
    txt(P, 34, y0 + h / 2, "service time（timestep，越低越好）", 11, SUB,
        "middle", "400", "-90")
    txt(P, W / 2, 440,
        "ψ=0（纯 1-中位数）在大 fleet 上比 static 更差；ψ≥80 又开始撕裂 L2 —— "
        "没有跨规模通用的 ψ，本文 ψ=30 取自 |A|=16 附近的稳健区段。",
        11.5, INK, "middle")
    save(P, os.path.join(OUT, "fig3_psi.svg"))


# ============================ Fig.4 ============================
def _heat_color(v, vmax):
    """v<0 越好（蓝）；v>0 越差（橙）。"""
    if v < 0:
        t = min(-v / vmax, 1.0)
        return "rgb(%d,%d,%d)" % (int(238 - 175 * t), int(244 - 140 * t),
                                  int(255 - 60 * t))
    t = min(v / vmax, 1.0)
    return "rgb(%d,%d,%d)" % (255, int(244 - 100 * t), int(229 - 160 * t))


def fig4():
    rows = load("E_hetero")
    alphas = sorted({r["zipf_alpha"] for r in rows})
    lam = sorted({r["lmbda"] for r in rows})
    ns = sorted({r["n_agents"] for r in rows})
    P, W, H = [], 900, 420
    head(P, W, H, "Fig.4  记忆何时有用：需求异质性 Zipf α 与到达强度 λ 对 MEMIR "
                  "增益的联合调制（Δ% vs static，越蓝越好）")
    panels = ns[:2]
    pw, ph = 300, 190
    gap = 60
    for pi, n in enumerate(panels):
        bx = 84 + pi * (pw + gap)
        by = 74
        grid = {}
        vmax = 1.0
        for a in alphas:
            for lm in lam:
                b_rows = pick([r for r in rows if r["n_agents"] == n
                               and abs(r["zipf_alpha"] - a) < 1e-9
                               and abs(r["lmbda"] - lm) < 1e-9],
                              method="static")
                m_rows = pick([r for r in rows if r["n_agents"] == n
                               and abs(r["zipf_alpha"] - a) < 1e-9
                               and abs(r["lmbda"] - lm) < 1e-9],
                              method="memir")
                if not b_rows or not m_rows:
                    grid[(a, lm)] = None
                    continue
                c = compare(b_rows, m_rows, "service_post", "memir")
                grid[(a, lm)] = c["pct"]
                vmax = max(vmax, abs(c["pct"]))
        cw = pw / len(lam)
        chh = ph / len(alphas)
        txt(P, bx + pw / 2, by - 8, "|A|=%d" % n, 12, INK, "middle", "600")
        for i, a in enumerate(alphas):
            txt(P, bx - 8, by + chh * (i + 0.5) + 4, "α=%g" % a, 10.5, SUB,
                "end")
            for j, lm in enumerate(lam):
                v = grid[(a, lm)]
                if v is None:
                    continue
                box(P, bx + cw * j, by + chh * i, cw - 1.5, chh - 1.5,
                    _heat_color(v, vmax), "#ffffff", 1.0)
                txt(P, bx + cw * (j + 0.5), by + chh * (i + 0.5) + 4,
                    "%+.1f" % v, 11,
                    "#0f172a" if abs(v) < vmax * 0.55 else "#ffffff",
                    "middle", "600")
        for j, lm in enumerate(lam):
            txt(P, bx + cw * (j + 0.5), by + ph + 16, "λ=%g" % lm, 10.5, SUB)
        txt(P, bx + pw / 2, by + ph + 34, "到达强度 λ", 11, SUB)
    # 色标
    lx, ly, lw, lh = 84 + 2 * (pw + gap) - gap + 10, 92, 26, 150
    steps = 40
    for k in range(steps):
        t = k / (steps - 1.0)
        v = vmax - 2 * vmax * t          # +vmax -> -vmax
        P.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                 'fill="%s"/>' % (lx, ly + lh * t, lw, lh / steps + 1,
                                  _heat_color(v, vmax)))
    box(P, lx, ly, lw, lh, "none", AXIS, 1.0)
    txt(P, lx + lw + 6, ly + 10, "+%g 更差" % vmax, 10, SUB, "start")
    txt(P, lx + lw + 6, ly + lh - 2, "−%g 更好" % vmax, 10, SUB, "start")
    txt(P, lx + lw / 2, ly - 6, "Δ%", 10.5, SUB)
    txt(P, 30, 74 + ph / 2, "需求异质性 Zipf α", 11, SUB, "middle", "400",
        "-90")
    txt(P, W / 2, 372,
        "α 越大（需求越集中）增益越大且单调；α=0 时空间结构消失，"
        "记忆在高空闲率下反而有害（+2.7% ~ +10.1%，见 § limitation）。",
        11.5, INK, "middle")
    save(P, os.path.join(OUT, "fig4_hetero.svg"))


# ============================ Fig.5 ============================
def _render_grid(P, bx, by, cell, J, series, title, mode="dwell", vmax=None):
    """mode: dwell=红色驻留密度 / heat=黄橙经验交通 / prior=棕黄结构先验。"""
    W_, H_ = J["layout"]["W"], J["layout"]["H"]
    stop = J["layout"]["station_top"]
    grid = J["layout"]["grid"]
    for y in range(H_):
        for x in range(W_):
            X, Y = bx + x * cell, by + y * cell
            if grid[y][x]:
                box(P, X, Y, cell, cell, "#334155", "none", 0)
            elif y >= stop:
                box(P, X, Y, cell, cell, "#e7f6ef", "none", 0)
            else:
                box(P, X, Y, cell, cell, "#f7f9fc", "#eef2f7", 0.5)
    vals = [v for _, _, v in series]
    mx = vmax if vmax else (max(vals) if vals else 1.0)
    for x, y, v in series:
        t = (v / mx) ** 0.45 if mx else 0.0
        if mode == "heat":
            fill = "rgb(%d,%d,%d)" % (int(255 - 40 * t), int(247 - 130 * t),
                                      int(224 - 150 * t))
            op = 0.25 + 0.75 * t
        elif mode == "prior":
            fill = "rgb(%d,%d,%d)" % (int(255 - 25 * t), int(246 - 120 * t),
                                      int(232 - 165 * t))
            op = 0.25 + 0.75 * t
        else:
            fill = "#dc2626"
            op = 0.10 + 0.90 * t
        P.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                 'fill="%s" opacity="%.3f"/>'
                 % (bx + x * cell, by + y * cell, cell, cell, fill, op))
    txt(P, bx + W_ * cell / 2, by - 7, title, 11.5, INK, "middle", "600")


def fig5():
    p = os.path.join(OUT, "fig5_dwell_W1_24.json")
    if not os.path.exists(p):
        print("missing", p)
        return
    J = json.load(open(p, encoding="utf-8"))
    W_, H_ = J["layout"]["W"], J["layout"]["H"]
    cell = 9
    P = []
    pw, ph = W_ * cell, H_ * cell
    gapx, gapy = 40, 54
    w = 26 + 2 * pw + gapx + 20
    h = 60 + 2 * ph + gapy + 150
    head(P, w, h, "Fig.5  簇聚的空闲 agent 正好压在载货干道上：%s，|A|=%d，"
         "warmup 后累积 6 seeds" % (J["map"], J["n"]))

    panels = [
        (0, 0, "haul", None, "① 经验载货交通 haul(c) —— 真正会被堵塞的车流"),
        (0, 1, "dwell", "static", "② static：铺开=避开干道"),
        (1, 0, "dwell", "memir0", "③ MEMIR ψ=0：1-中位数把质量吸到热点"),
        (1, 1, "dwell", "memir", "④ MEMIR ψ=30：保留空间结构，让开干道"),
    ]
    dmax = max(max(v for _, _, v in J["dwell"][k])
               for k in ("static", "memir0", "memir"))
    for (r, c, mode, key, title) in panels:
        bx = 26 + c * (pw + gapx)
        by = 62 + r * (ph + gapy)
        if mode == "haul":
            _render_grid(P, bx, by, cell, J, J["haul"], title, mode="heat")
        else:
            _render_grid(P, bx, by, cell, J, J["dwell"][key], title,
                         mode="dwell", vmax=dmax)
            st = J["stat"][key]
            b = J["stat"]["static"]
            txt(P, bx + pw / 2, by + ph + 16,
                "重叠 = %.2f  ｜ wait = %.3f%%  ｜ detour = %.4f"
                % (st["overlap"], st["wait"], st["detour"]),
                10.5, INK, "middle", "600")
            txt(P, bx + pw / 2, by + ph + 31,
                "L2 = %.2f  ｜  L3 = %.2f （Δ vs static %+.2f）"
                % (st["leg_approach"], st["leg_haul"],
                   st["leg_haul"] - b["leg_haul"]),
                10.5, "#b91c1c" if st["leg_haul"] - b["leg_haul"] > 0.3
                else "#15803d", "middle", "600")

    # 底部 footdialogue：机制链
    fy = 62 + 2 * ph + gapy + 6
    txt(P, 26, fy + 4,
        "机制链：空闲质量 × 载货交通的重叠 ⇒ 让行 wait 与绕行 detour ⇒ L3 膨胀。",
        12, INK, "start", "600")
    chain = [
        ("重叠 ↑", "ψ=0 把驻留质量堆在干道上：重叠 1.36 = static(0.41) 的 3.3 倍、"
                   "nearest(0.09) 的 15 倍"),
        ("wait ↑", "wait 0.102% → 0.458%（4.5 倍）；detour 1.0116 → 1.0287"),
        ("L3 ↑", "L3 29.57 → 32.77，把 L2 省下的 5.1 timestep 吃掉 3.2"),
        ("ψ 修复", "ψ=30：重叠 1.36 → 0.75（−45%）、wait → 0.237%、"
                   "L3 → 29.75，代价仅 L2 +0.39"),
    ]
    cx = 26
    for i, (tag, desc) in enumerate(chain):
        col = "#dc2626" if i < 3 else "#15803d"
        P.append('<rect x="%.1f" y="%.1f" width="%.1f" height="20" rx="4" '
                 'fill="%s" opacity="0.10"/>' % (cx, fy + 20, 62, col))
        txt(P, cx + 31, fy + 34, tag, 11, col, "middle", "600")
        txt(P, cx + 72, fy + 34, desc, 11, INK, "start")
        fy += 26
    txt(P, 26, h - 16,
        "深色 = 货架（障碍）；浅绿 = delivery endpoint 行；红色深浅 = 空闲 "
        "agent 的驻留频次（三幅驻留图共用同一色标，故可直接比较浓淡）。",
        10.5, SUB, "start")
    save(P, os.path.join(OUT, "fig5_map.svg"))


# ============================ Fig.6 ============================
def _delta_table(exp, pols):
    """返回 {pol: [Δ% vs static, ...]} 与 x 轴标签（fleet 规模）。"""
    rows = load(exp)
    if not rows:
        return [], {}
    ns = sorted({r["n_agents"] for r in rows})
    out = {}
    for p in pols:
        vals = []
        for n in ns:
            base = pick(rows, n_agents=n, method="static")
            rr = pick(rows, n_agents=n, method=p)
            if not base or not rr:
                vals.append(None)
                continue
            c = compare(base, rr, "service_post", p)
            vals.append(c["pct"])
        out[p] = vals
    return ns, out


def fig6():
    """新增基线的总对照：(a) 近年非学习基线；(b) 学习类基线；(c) 决策开销。"""
    nsJ, dJ = _delta_table("J_base",
                           ["nearest", "popsample", "pmedian", "memir0",
                            "memir"])
    nsK, dK = _delta_table("K_learn",
                           ["memir", "attn", "rl_lin", "rl_tab"])
    P, W, H = [], 1060, 560
    head(P, W, H, "Fig.6  新增基线总对照：近年非学习方法 / 学习类方法 "
                  "（Δ service time vs static，越负越好；W1, λ=0.08, α=2.0, "
                  "10 seeds）")

    panel_line(P, (78, 66, 380, 320), [str(n) for n in nsJ], dJ, (-30, 55),
               "(a) 近年 / 对照社区的非学习基线", "", "", show_leg=False)
    txt(P, 78 + 190, 66 + 320 + 54, "|A|（fleet 规模）", 11, SUB)
    txt(P, 36, 66 + 160, "Δ vs static (%)", 11, SUB, "middle", "400", "-90")
    legend(P, 78 + 380 + 14, 66 + 8, ["nearest", "popsample", "pmedian",
                                      "memir0", "memir"])
    txt(P, 78, 66 + 320 + 76,
        "p-median 用与 MEMIR 完全相同的 p̂/β/ψ，却仍差 0.2–2.0 pp —— "
        "增益不来自优化器，来自「外生需求场 + 流载先验」。", 10.5, SUB, "start")

    panel_line(P, (600, 66, 330, 320), [str(n) for n in nsK], dK, (-25, 40),
               "(b) 学习类基线（直接输出驻留位置）", "", "", show_leg=False)
    txt(P, 600 + 165, 66 + 320 + 54, "|A|（fleet 规模）", 11, SUB)
    txt(P, 558, 66 + 160, "Δ vs static (%)", 11, SUB, "middle", "400", "-90")
    legend(P, 600 + 330 + 14, 66 + 8, ["memir", "attn", "rl_lin", "rl_tab"])
    txt(P, 600, 66 + 320 + 76,
        "小 fleet + 长训练窗口下学习类可与 MEMIR 统计不可区分；"
        "差距随 fleet 单调扩大（+1.5 n.s. → +2.5 → +5.2 pp）。",
        10.5, SUB, "start")

    # (c) 决策开销条
    jr = load("J_base")
    kr = load("K_learn")
    items = []
    for exp, rows, pols, ns in (("J_base", jr, ["static", "popsample",
                                                "pmedian", "memir"], 24),
                                ("K_learn", kr, ["attn", "rl_lin",
                                                 "memir"], 24)):
        for pol in pols:
            rr = pick(rows, n_agents=ns, method=pol)
            if not rr:
                continue
            ms = sum(r.get("decide_ns", 0) for r in rr) / 1e6 / \
                max(sum(r.get("decide_cnt", 0) for r in rr), 1)
            items.append((pol, ms))
    seen = set()
    uniq = []
    for nm, ms in items:
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append((nm, ms))
    uniq.sort(key=lambda kv: kv[1])
    bx, by, bw = 78, 500, 900
    txt(P, bx, by - 6, "(c) 单次决策开销 @ |A|=24（ms/decision）", 12, INK,
        "start", "600")
    mx = max(v for _, v in uniq) or 1.0
    y = by + 10
    for nm, v in uniq:
        col = C.get(nm, INK)
        txt(P, bx, y + 12, LEG.get(nm, nm), 10.5, INK, "start")
        box(P, bx + 250, y + 2, 6.0, 12, fill=col, stroke="none")
        P.append('<rect x="%.1f" y="%.1f" width="%.1f" height="12" '
                 'fill="%s" opacity="0.85"/>'
                 % (bx + 250, y + 2, 520.0 * v / mx, col))
        txt(P, bx + 250 + 520.0 * v / mx + 6, y + 12, "%.2f ms" % v, 10.5,
            INK, "start")
        y += 20
    txt(P, bx, y + 16,
        "MEMIR 比 p-median 快 4–7 倍、比学习类快 2–7 倍，且零训练、"
        "不需维护「他人已承诺驻守集合」这一全局状态。", 10.5, SUB, "start")
    save(P, os.path.join(OUT, "fig6_baselines.svg"))


# ============================ Fig.7 ============================
LEARN_POLS = ["rl_lin", "rl_lin_g", "rl_tab", "attn", "attn_g", "attn_bc",
              "rl_lin_warm", "rl_tab_warm", "attn_warm", "attn_bc_warm"]
for _k, _col in (("m16", "#dc2626"), ("m24", "#ea580c"), ("m32", "#0891b2"),
                 ("m40", "#7c3aed")):
    C.setdefault(_k, _col)
    WIDTH.setdefault(_k, 2.2)
    LEG.setdefault(_k, "MEMIR |A|=%s" % _k[1:])


def _gbars(P, x0, y0, w, h, groups, names, ylim, title, note=None):
    """分组柱状图（Δ% 面板）。负值向下延伸 = 更好。"""
    ymin, ymax = ylim

    def py(v):
        v = min(max(v, ymin), ymax)
        return y0 + h - h * (v - ymin) / (ymax - ymin)

    txt(P, x0 + w / 2, y0 - 8, title, 12, INK, "middle", "600")
    for v in ticks(ymin, ymax):
        yy = py(v)
        P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                 'stroke="%s" stroke-width="1"/>' % (x0, yy, x0 + w, yy, GRID))
        txt(P, x0 - 7, yy + 4, "%g" % v, 10.5, SUB, "end")
    gw = w / max(len(groups), 1)
    bw = min(24.0, gw * 0.62 / max(len(names), 1))
    for gi, (glabel, vals) in enumerate(groups):
        cx = x0 + gw * (gi + 0.5)
        total = bw * len(names)
        for ni, nm in enumerate(names):
            v = vals.get(nm)
            if v is None:
                continue
            bx = cx - total / 2 + ni * bw
            yz, yv = py(0.0), py(v)
            top, bot = min(yz, yv), max(yz, yv)
            P.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="%s" opacity="0.88"/>'
                     % (bx, top, bw * 0.86, max(bot - top, 0.6),
                        C.get(nm, INK)))
            txt(P, bx + bw * 0.43, bot + 12 if v < 0 else top - 5,
                "%.1f" % v, 9.5, INK, "middle", "600")
        txt(P, cx, y0 + h + 17, glabel, 10.5, SUB)
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, py(0.0), x0 + w, py(0.0), INK))
    P.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
             % (x0, y0, x0, y0 + h, AXIS))
    if note:
        txt(P, x0, y0 + h + 40, note, 10.5, SUB, "start")


def fig7():
    """泛化与边界：标准 benchmark / 学习类跨地图 / 守护强度不敏感。"""
    rowsJ, rowsK = load("J_base"), load("K_learn")
    rowsL, rowsM = load("L_gen"), load("M_probe")
    rowsN = load("N_bench")

    def dvs(rows, n, meth, base="static", m=None):
        b = pick(rows, n_agents=n, method=base, **({"map": m} if m else {}))
        r = pick(rows, n_agents=n, method=meth, **({"map": m} if m else {}))
        if not b or not r:
            return None
        return compare(b, r, "service_post", meth)["pct"]

    P, W, H = [], 1120, 700
    head(P, W, H, "Fig.7  泛化与边界：(a) MovingAI 标准 benchmark 地图    "
                  "(b) 学习类基线的地图泛化    (c) TER-Safety 守护强度不敏感")

    # ---- (a) 标准 benchmark ----
    groups = []
    for label, rows, m, n in (("W1 16", rowsJ, None, 16), ("W1 32", rowsJ,
                                                           None, 32),
                              ("B1 16", rowsN, "B1", 16),
                              ("B1 32", rowsN, "B1", 32),
                              ("B2 16", rowsN, "B2", 16),
                              ("B2 32", rowsN, "B2", 32)):
        groups.append((label, {"memir0": dvs(rows, n, "memir0", m=m),
                               "memir": dvs(rows, n, "memir", m=m)}))
    _gbars(P, 78, 66, 470, 300, groups, ["memir0", "memir"], (-30, 8),
           "(a) Δ service time vs static（越向下越好）：社区标准地图复现结论",
           "B1 = warehouse-10-20-10-2-1（161×63）；B2 = warehouse-10-20-10-2-2"
           "（170×84）。标准图上增益 −19.6% ~ −25.8%，不小于 W1。")
    hlegend(P, 78, 380, ["memir0", "memir"], 470)

    # ---- (b) 学习类跨地图：最优学习类相对 MEMIR 的差距 ----
    gaps = []
    for label, rows, m, n in (("W1 8", rowsK, None, 8), ("W1 16", rowsK,
                                                         None, 16),
                              ("W2 8", rowsL, "W2", 8),
                              ("W2 16", rowsL, "W2", 16),
                              ("W3 8", rowsL, "W3", 8),
                              ("W3 16", rowsL, "W3", 16)):
        kw = {"map": m} if m else {}
        mem = pick(rows, n_agents=n, method="memir", **kw)
        best, bname = None, ""
        for pol in LEARN_POLS:
            rr = pick(rows, n_agents=n, method=pol, **kw)
            if not rr or not mem:
                continue
            c = compare(mem, rr, "service_post", pol)
            if best is None or c["pct"] < best:
                best, bname = c["pct"], pol
        gaps.append((label, best, bname))
    gb = [(lb, {"attn": v}) for lb, v, _ in gaps]
    _gbars(P, 640, 66, 420, 300, gb, ["attn"], (0, 8),
           "(b) 最优学习类 − MEMIR（正值 = 学习类更差）",
           "离开 W1 后没有任何配置能追平；W3/|A|=16 处 ψ 价值 ≈19 pp，"
           "学习类全线崩溃（+3.2% ~ +23.2%）。")
    for i, (lb, v, nm) in enumerate(gaps):
        cx = 640 + 420 * (i + 0.5) / len(gaps)
        txt(P, cx, 66 + 300 + 30, nm if v is not None else "—", 9.5, SUB)

    # ---- (c) probe 不敏感 ----
    probes = [4, 8, 16, 32, 64]
    data = {}
    for n in (16, 24, 32, 40):
        key = "m%d" % n
        vals = []
        for pr in probes:
            rr = pick(rowsM, n_agents=n, method="memir_p%d" % pr)
            base = pick(rowsM, n_agents=n, method="static")
            if not rr or not base:
                vals.append(None)
                continue
            vals.append(compare(base, rr, "service_post", key)["pct"])
        data[key] = vals
    panel_line(P, (78, 470, 1000, 150), [str(p) for p in probes], data,
               (-16, 4), "(c) TER-Safety 守护强度 dwell_probe：16 倍范围内"
               "全部 n.s.（Δ vs static，越负越好）", "", "", show_leg=False)
    txt(P, 78 + 500, 470 + 150 + 34, "dwell_probe（允许试探的候选个数）",
        11, SUB)
    txt(P, 40, 470 + 75, "Δ vs static (%)", 11, SUB, "middle", "400", "-90")
    hlegend(P, 78, 470 + 150 + 48, ["m16", "m24", "m32", "m40"], 1000)
    txt(P, 78, 470 + 150 + 90,
        "40 组配对比较全部 n.s.，|Δ| < 0.8 pp，全表 plan_fail = 0 —— "
        "守护强度不是瓶颈，默认 8 已远超 liveness 所需的安全裕度。",
        10.5, SUB, "start")
    save(P, os.path.join(OUT, "fig7_generalization.svg"))


FREE_CELLS = {"W1": 711, "W2": 1230, "W3": 213, "W4": 2551, "W5": 5511,
              "W6": 9591, "B1": 5699, "B2": 9776, "B3": 22599, "B4": 30000}
MAP_COLOR = {"W4": "#0891b2", "W5": "#2563eb", "W6": "#7c3aed",
             "B1": "#dc2626", "B3": "#059669"}
MAP_MARK = {"W4": "circle", "W5": "square", "W6": "diamond",
            "B1": "circle", "B3": "square"}


def _scatter_panel(P, bx, pts, xlim, ylim, title, xlab, ylab, note=None):
    """散点面板。pts = [(x, y, color, marker, label)]。"""
    x0, y0, w, h = bx
    xmin, xmax = xlim
    ymin, ymax = ylim
    box(P, x0, y0, w, h)
    txt(P, x0 + w / 2.0, y0 - 12, title, 12, INK, "middle", "600")

    def px(v):
        return x0 + w * (v - xmin) / (xmax - xmin)

    def py(v):
        return y0 + h - h * (v - ymin) / (ymax - ymin)

    for tv in ticks(xmin, xmax, 5):
        if tv < xmin - 1e-9 or tv > xmax + 1e-9:
            continue
        X = px(tv)
        poly(P, [(X, y0), (X, y0 + h)], GRID, 1.0)
        txt(P, X, y0 + h + 16, "%g" % tv, 10, SUB)
    for tv in ticks(ymin, ymax, 5):
        if tv < ymin - 1e-9 or tv > ymax + 1e-9:
            continue
        Y = py(tv)
        poly(P, [(x0, Y), (x0 + w, Y)], GRID, 1.0)
        txt(P, x0 - 8, Y + 4, "%g" % tv, 10, SUB, "end")
    if ymin < 0 < ymax:
        poly(P, [(x0, py(0)), (x0 + w, py(0))], AXIS, 1.4)
    txt(P, x0 + w / 2.0, y0 + h + 34, xlab, 11, SUB)
    txt(P, x0 - 46, y0 + h / 2.0, ylab, 11, SUB, "middle", "400", "-90")

    for (vx, vy, col, mk, lb) in pts:
        X, Y = px(vx), py(vy)
        if mk == "square":
            P.append('<rect x="%.1f" y="%.1f" width="9" height="9" '
                     'fill="%s" stroke="#ffffff" stroke-width="1"/>'
                     % (X - 4.5, Y - 4.5, col))
        elif mk == "diamond":
            P.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f '
                     '%.1f,%.1f" fill="%s" stroke="#ffffff" stroke-width="1"/>'
                     % (X, Y - 5.5, X + 5.5, Y, X, Y + 5.5, X - 5.5, Y, col))
        else:
            dot(P, X, Y, 4.6, col, "#ffffff")
    if note:
        txt(P, x0 + 6, y0 + 16, note, 10, SUB, "start")


def fig8():
    """★ H-scale：增益是 idle rate 的函数，还是地图尺寸的函数？

    (a) 把 O_scale 全部 (地图, |A|, λ) 格子画在 (idle rate, Δ service time) 平面，
        按地图着色。若不同尺寸的散点落在同一条曲线上 → 坍缩（与尺度无关）；
        若按尺寸分层 → 可调度窗口的位置依赖空间尺度。
    (b) 在 idle 相近的一档内，画增益 vs free cells（对数轴）：
        水平 ⇒ 增益与尺度无关；倾斜 ⇒ 存在尺度效应。
    """
    rows = load("O_scale")
    if not rows:
        print("skip fig8: no O_scale data")
        return
    cells = {}
    for r in rows:
        k = (r["map"], int(r["n_agents"]), float(r["lmbda"]))
        cells.setdefault(k, {}).setdefault(r["method"], {})[int(r["seed"])] = r

    recs = []
    for k in sorted(cells, key=lambda z: (z[0], z[1], z[2])):
        cell = cells[k]
        if "static" not in cell or "memir" not in cell:
            continue
        base, r1 = cell["static"], cell["memir"]
        seeds = sorted(set(base) & set(r1))
        if not seeds:
            continue
        try:
            svc = sum(float(base[s]["service_post"]) for s in seeds) / len(seeds)
            svc1 = sum(float(r1[s]["service_post"]) for s in seeds) / len(seeds)
            idle = sum(float(r1[s]["idle_ratio"]) for s in seeds) / len(seeds)
        except (TypeError, ValueError, KeyError):
            continue
        if svc <= 0:
            continue
        recs.append(dict(map=k[0], n=k[1], lam=k[2], idle=idle * 100.0,
                         gain=100.0 * (svc1 - svc) / svc))
    if not recs:
        print("skip fig8: no usable cells")
        return

    P, W, H = [], 1120, 620
    head(P, W, H, "Fig.8  ★ 尺度检验 H-scale：增益是 idle rate 的函数，"
                  "还是地图尺寸的函数？（O_scale）")

    # ---- (a) 坍缩图 ----
    pts = [(c["idle"], c["gain"], MAP_COLOR.get(c["map"], INK),
            MAP_MARK.get(c["map"], "circle"),
            "%s |A|=%d" % (c["map"], c["n"])) for c in recs]
    _scatter_panel(P, (78, 66, 620, 400), pts, (0, 90), (-32, 4),
                   "(a) 全部格子画在 (idle rate, Δ service time) 平面上"
                   "（按地图着色，越向下越好）",
                   "idle rate of MEMIR (%)", "Δ service time vs static (%)",
                   "灰色 = 饱和区：idle 低于约 15% 时ΔL2/L2≈0，增益消失")

    # 饱和区阴影
    x0, y0, w, h = 78, 66, 620, 400
    xsh = x0 + w * 15.0 / 90.0
    box(P, x0, y0, xsh - x0, h, fill="#f1f5f9", stroke="none", op="0.9")
    for (vx, vy, col, mk, lb) in pts:
        X = x0 + w * vx / 90.0
        Y = y0 + h - h * (vy + 32) / 36.0
        if mk == "square":
            P.append('<rect x="%.1f" y="%.1f" width="9" height="9" '
                     'fill="%s" stroke="#ffffff" stroke-width="1"/>'
                     % (X - 4.5, Y - 4.5, col))
        elif mk == "diamond":
            P.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f '
                     '%.1f,%.1f" fill="%s" stroke="#ffffff" stroke-width="1"/>'
                     % (X, Y - 5.5, X + 5.5, Y, X, Y + 5.5, X - 5.5, Y, col))
        else:
            dot(P, X, Y, 4.6, col, "#ffffff")
    txt(P, x0 + 8, y0 + h - 8, "饱和区", 9.5, SUB, "start")

    lx, ly = 78, 66 + 400 + 52
    cx = lx
    for m in ("W4", "W5", "W6", "B1", "B3"):
        col = MAP_COLOR.get(m, INK)
        mk = MAP_MARK.get(m, "circle")
        if mk == "square":
            P.append('<rect x="%.1f" y="%.1f" width="9" height="9" fill="%s"/>'
                     % (cx - 4.5, ly - 4.5, col))
        elif mk == "diamond":
            P.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f '
                     '%.1f,%.1f" fill="%s"/>'
                     % (cx, ly - 5.5, cx + 5.5, ly, cx, ly + 5.5,
                        cx - 5.5, ly, col))
        else:
            dot(P, cx, ly, 4.6, col, "#ffffff")
        txt(P, cx + 9, ly + 4, "%s（%d free）" % (m, FREE_CELLS.get(m, 0)),
            10.5, INK, "start")
        cx += 128

    # ---- (b) 匹配 idle 档内：增益 vs 地图规模 ----
    band = [c for c in recs if 35.0 <= c["idle"] <= 50.0]
    x1, y1_, w2, h2 = 760, 66, 300, 400
    box(P, x1, y1_, w2, h2)
    txt(P, x1 + w2 / 2.0, y1_ - 12,
        "(b) idle ∈ [35%, 50%] 档内：Δ vs 地图规模", 12, INK, "middle", "600")
    if band:
        import math
        xs = [math.log10(FREE_CELLS.get(c["map"], 1)) for c in band]
        ys = [c["gain"] for c in band]
        lo, hi = min(xs) - 0.15, max(xs) + 0.15
        ylo, yhi = min(-30.0, min(ys) - 3), max(0.0, max(ys) + 3)
        for tv in (3.3, 3.5, 3.7, 3.9, 4.1, 4.3):
            if tv < lo or tv > hi:
                continue
            X = x1 + w2 * (tv - lo) / (hi - lo)
            poly(P, [(X, y1_), (X, y1_ + h2)], GRID, 1.0)
            txt(P, X, y1_ + h2 + 16, "%dk" % int(round(10 ** tv / 1000.0)),
                10, SUB)
        for tv in ticks(ylo, yhi, 5):
            Y = y1_ + h2 - h2 * (tv - ylo) / (yhi - ylo)
            poly(P, [(x1, Y), (x1 + w2, Y)], GRID, 1.0)
            txt(P, x1 - 8, Y + 4, "%g" % tv, 10, SUB, "end")
        poly(P, [(x1, y1_ + h2 - h2 * (0 - ylo) / (yhi - ylo)),
                 (x1 + w2, y1_ + h2 - h2 * (0 - ylo) / (yhi - ylo))],
             AXIS, 1.4)
        for c in band:
            X = x1 + w2 * (math.log10(FREE_CELLS.get(c["map"], 1)) - lo) / (hi - lo)
            Y = y1_ + h2 - h2 * (c["gain"] - ylo) / (yhi - ylo)
            dot(P, X, Y, 5.0, MAP_COLOR.get(c["map"], INK), "#ffffff")
            txt(P, X, Y - 9, "%s/%d" % (c["map"], c["n"]), 9, SUB)
        # 线性回归斜率
        n = len(xs)
        if n >= 2:
            mx = sum(xs) / n
            my = sum(ys) / n
            den = sum((v - mx) ** 2 for v in xs)
            if den > 1e-9:
                sl = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den
                txt(P, x1 + w2 / 2.0, y1_ + h2 + 34,
                    "free cells（对数轴）", 11, SUB, "middle")
                txt(P, x1 + 8, y1_ + 18,
                    "斜率 = %.1f pp / 每十倍面积" % sl, 10.5, INK, "start")
    else:
        txt(P, x1 + w2 / 2.0, y1_ + h2 / 2.0, "该 idle 档内暂无格子", 11, SUB)

    txt(P, 40, y1_ + h2 / 2.0, "Δ service time vs static (%)", 11, SUB,
        "middle", "400", "-90")

    txt(P, 78, 560,
        "读法：(a) 若不同尺寸的散点落在同一条曲线上，则增益只是 idle rate 的函数"
        "（与尺度无关）——这是 H-scale 的“坍缩”分支；"
        "所有格子同向（负值）且符号检验 p≈3.8e-6。", 10.5, SUB, "start")
    save(P, os.path.join(OUT, "fig8_scale_collapse.svg"))


FIG = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
       "fig6": fig6, "fig7": fig7, "fig8": fig8}


def main():
    which = sys.argv[1:] or list(FIG)
    for k in which:
        FIG[k]()


if __name__ == "__main__":
    main()
