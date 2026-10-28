"""Phase 3 补充图：倒 U 型（空闲率）与「ORACLE 在漂移下失效」。

这两张图承载本课题最核心的机制结论，因此单独成模块以免与基础图混淆。
依赖 analyze_open 的绘图原语。
"""
from analyze_open import (ACC, ACC2, ACC4, FG, MUTED, agg, axes, legend,
                          save, sel, svg_close, svg_open)


def fig_e2b(rows):
    """记忆增益关于「空闲率」呈倒 U 型 —— 统一 Phase 1/2/3 的钥匙。"""
    r = sel(rows, group="E2_fleet")
    ns = [4, 6, 8, 12, 16, 24, 32]
    pts = []
    for n in ns:
        nn = sel(r, method="nearest_n%d" % n)
        ac = sel(r, method="aco_n%d" % n)
        sv = sel(r, method="static_n%d" % n)
        if not nn or not ac:
            continue
        idle = agg(nn, "idle_ratio")[0] * 100
        g = (agg(ac)[0] / agg(nn)[0] - 1) * 100
        gs = (agg(sv)[0] / agg(nn)[0] - 1) * 100 if sv else None
        pts.append((n, idle, g, gs))
    pts.sort(key=lambda x: x[1])

    W, H = 780, 450
    s = svg_open(W, H, "记忆增益关于空闲率呈倒 U 型：统一三个阶段的钥匙",
                 "空闲率（%）：agent 无事可做的 timestep 占比",
                 "ACO 相对 nearest 的增益 %（越负越好）")
    x0, y0, x1, y1 = 85, 66, W - 30, H - 66
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts] + [0]
    xmin, xmax = -3, max(xs) * 1.05
    ymin, ymax = min(ys) * 1.2, 3
    px, py = axes(s, x0, y0, x1, y1, xmin, xmax, ymin, ymax,
                  [0, 20, 40, 60, 80],
                  [round(ymin + (ymax - ymin) * k / 5) for k in range(6)])
    s.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
             'stroke-dasharray="3,3"/>'
             % (x0, py(0), x1, py(0), MUTED))
    d = " ".join("%g,%g" % (px(a), py(b)) for _, a, b, _ in pts)
    s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.6"/>'
             % (d, ACC))
    ok = [(a, b) for _, a, _, b in pts if b is not None]
    ds = " ".join("%g,%g" % (px(a), py(b)) for a, b in ok)
    s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.8" '
             'stroke-dasharray="5,3"/>' % (ds, ACC4))
    for n, idle, g, _ in pts:
        s.append('<circle cx="%g" cy="%g" r="4.2" fill="%s"/>'
                 % (px(idle), py(g), ACC))
        s.append('<text x="%g" y="%g" font-size="9.5" fill="%s" '
                 'text-anchor="middle">%d</text>'
                 % (px(idle), py(g) - 11, FG, n))
    # 区域注记
    s.append('<text x="%g" y="%g" font-size="10" fill="%s" text-anchor="middle">'
             '从不空闲</text>' % (px(3), py(-3.5), MUTED))
    s.append('<text x="%g" y="%g" font-size="10" fill="%s" text-anchor="middle">'
             '无从调度</text>' % (px(3), py(-7.5), MUTED))
    s.append('<text x="%g" y="%g" font-size="10.5" fill="%s" text-anchor="middle">'
             '稀缺但可调度 → 峰值</text>' % (px(62), y0 + 18, ACC))
    s.append('<text x="%g" y="%g" font-size="10" fill="%s" text-anchor="middle">'
             '覆盖已充分</text>' % (px(86), py(-6), MUTED))
    s.append('<text x="%g" y="%g" font-size="10" fill="%s" text-anchor="middle">'
             '预测无价值</text>' % (px(86), py(-10), MUTED))
    legend(s, x0 + 16, y0 + 40, [("ACO 记忆（预定位）", ACC),
                                 ("static 纯分散对照", ACC4)])
    s.append('<text x="%g" y="%g" font-size="9.5" fill="%s" text-anchor="middle">'
             '（数据点旁数字 = agent 数 |A|）</text>' % ((x0 + x1) / 2, y1 + 40, MUTED))
    return save("fig_e2b_u_shape.svg", svg_close(s))


def fig_e5b(rows):
    """漂移下 ORACLE 输给 ACO：事后平均真值没有时间分辨率。"""
    r = sel(rows, group="E5_drift")
    conds = [("static0", "静止", ""), ("periodic1500", "周期", "1500"),
             ("periodic750", "周期", "750"), ("shift1500", "突变", "1500"),
             ("shift750", "突变", "750"), ("shift400", "突变", "400")]
    series = [("aco_t100", "ACO τ=100", ACC),
              ("meta60", "ACO+RL 元适应", ACC2),
              ("oracle", "ORACLE（事后真实分布）", MUTED)]
    W, H = 780, 440
    s = svg_open(W, H, "漂移环境下 ORACLE 输给 ACO：事后平均真值没有时间分辨率",
                 "", "相对 nearest 的增益 %（越负越好）")
    x0, y0, x1, y1 = 85, 66, W - 30, H - 86
    vals, allg = {}, []
    for key, _, _ in conds:
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
    ymin, ymax = min(allg) * 1.18, max(allg) * 0.86
    px, py = axes(s, x0, y0, x1, y1, 0, len(conds), ymin, ymax, [],
                  [round(ymin + (ymax - ymin) * k / 5) for k in range(6)])
    nser = len(series)
    bw = (x1 - x0) / len(conds)
    for i, (key, l1, l2) in enumerate(conds):
        for j, (pre, _, col) in enumerate(series):
            g = vals.get((pre, key))
            if g is None:
                continue
            cx = x0 + i * bw + bw * (j + 0.5) / nser
            s.append('<rect x="%g" y="%g" width="%g" height="%g" fill="%s" '
                     'opacity="0.92"/>'
                     % (cx - bw / nser * 0.38, py(g), bw / nser * 0.76,
                        max(y1 - py(g), 1), col))
        s.append('<text x="%g" y="%g" font-size="10" fill="%s" '
                 'text-anchor="middle">%s</text>'
                 % (x0 + i * bw + bw / 2, y1 + 18, FG, l1))
        if l2:
            s.append('<text x="%g" y="%g" font-size="9" fill="%s" '
                     'text-anchor="middle">%s</text>'
                     % (x0 + i * bw + bw / 2, y1 + 30, MUTED, l2))
    legend(s, x0 + 12, y0 + 10, [(lab, col) for _, lab, col in series])
    return save("fig_e5b_oracle_fails.svg", svg_close(s))


def all_figures(rows):
    return [fig_e2b(rows), fig_e5b(rows)]
