"""把 Fig.1–Fig.8 的 SVG 内联到一个可直接预览/打印的 HTML 页面。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results_final", "figs")
OUT = os.path.join(ROOT, "论文配图_Fig1-Fig8.html")

CAPS = [
    ("fig1_fleet.svg", "Fig.1  主对照：fleet 规模扫描", [
        "(a) 全量程可见 |A|=4 的饱和悬崖：idle≈1%，任何泊位策略都无从调度，"
        "六条曲线退化到 ≈152。这是此前「信息素分配调制」失败的工作区间。",
        "(b) 进入可调度区间后 MEMIR 立刻与三个基线分离，峰值在 |A|=12"
        "（−26.2%）；ψ=0 单调恶化并在 |A|≥32 反超 static（变负收益）。",
    ]),
    ("fig2_legs.svg", "Fig.2  ★ 论文灵魂图：三段时延的堆叠分解", [
        "ψ=0 把 L2(assign→pickup) 从 17.4 压到 5.1 —— 这是加权 1-中位数的"
        "全部功劳；代价是 L3(pickup→delivery) 随 fleet 从 29.8 涨到 34.1。",
        "ψ=30 几乎只动 L3（31.7→29.8 @|A|=16，33.8→30.2 @|A|=32），"
        "L2 仅 +0.1~+0.4，因此增益得以净实现。三段之和严格等于 service time"
        "（残差 <0.01 timestep），探针是可信的。",
    ]),
    ("fig3_psi.svg", "Fig.3  ψ 的内部最优与迁移", [
        "每条 fleet 曲线都呈 U 型：ψ=0 不够、ψ≥80 过头。"
        "ψ* 随规模迁移 6→20、8→30、16→30、32→40。",
        "不存在跨规模通用的 ψ；取 ψ=30 的理由是在 |A|=16 附近既接近最优、"
        "又对迁移最不敏感（相邻档位落差 <1%）。",
    ]),
    ("fig4_hetero.svg", "Fig.4  记忆何时有用：α × λ 的联合调制", [
        "α 从 0 增大到 3，增益单调变负（越好）——需求越集中，"
        "未来分布越可预测，外生记忆越有价值。",
        "α=0 且高空闲率时记忆明确有害（+2.7%~+10.1%），"
        "这是必须写进 limitation 的边界条件。",
    ]),
    ("fig5_map.svg", "Fig.5  机制可视化：簇聚的空闲 agent 压在载货干道上", [
        "用**经验载货交通**（已取货、正送往 delivery 的 agent 的实际轨迹）"
        "定义真正的干道，而不是用任何先验。",
        "干扰强度 = 驻留质量 × 载货交通的重叠：static 0.41 → ψ=0 1.36"
        "（3.3 倍）→ ψ=30 0.75。与之同步，wait 0.102%→0.458%→0.237%，"
        "detour 1.0116→1.0287→1.0171，L3 29.57→32.77→29.75。",
        "整套因果链在同一批任务流、同一 fleet 上取得，"
        "排除了吞吐量差异与选址错误的替代解释。",
    ]),
    ("fig6_baselines.svg", "Fig.6  新增基线总对照（2026-09-26）", [
        "(a) p-median（greedy 设施选址）用的是与 MEMIR 完全相同的 p̂、β、ψ，"
        "却仍差 0.19–1.96 pp —— 增益不来自「1-中位数」这个优化器，"
        "而来自「外生需求场 + 流载先验」这两味料。",
        "ψ 跨策略族迁移：|A|=32 下 pmedian0→pmedian 从 +9.03% 变 −5.31%"
        "（14.3 pp），popsample→popsample_f 从 +0.74%(n.s.) 变 −1.19%。"
        "三种结构完全不同的策略都被同一个 ψ 挽救。",
        "(b) 学习类（REINFORCE 线性打分器 / 上下文 bandit / 注意力指针网络）"
        "在小 fleet + 长训练窗口下可与 MEMIR 统计不可区分（|A|=8：−0.37% n.s.），"
        "但差距随 fleet 单调扩大到 +5.19 pp（|A|=24）。",
        "(c) MEMIR 的决策开销 0.93–1.85 ms，比 p-median 快 4–7 倍、"
        "比学习类快 2–7 倍，且零训练。",
    ]),
    ("fig7_generalization.svg", "Fig.7  泛化与边界（2026-09-26 第三轮）", [
        "(a) MovingAI 标准 benchmark（B1 = warehouse-10-20-10-2-1，"
        "161×63 / 5 699 states；B2 = warehouse-10-20-10-2-2，"
        "170×84 / 9 776 states）上 Δ vs static 为 −19.6% ~ −25.8%，"
        "且增益全部来自 L2、L3 基本不动 —— 与 W1 的机制图谱一致。",
        "(b) 最优学习类相对 MEMIR 的差距在 W1/W2/W3 × |A|∈{8,16} 上全为正："
        "离开 W1 后没有任何配置能追平 MEMIR。W3/|A|=16 上 ψ 的价值 ≈19 pp"
        "（memir0 +16.31% → memir −2.88%），学习类在该格全线崩溃。",
        "(c) TER-Safety 守护强度 dwell_probe ∈ {4,8,16,32,64}（16 倍范围）"
        "时四条曲线全部水平：40 组配对比较 n.s.，|Δ| < 0.8 pp，plan_fail = 0。",
    ]),
    ("fig8_scale_collapse.svg",
     "Fig.8  ★ 尺度检验 H-scale（2026-09-26 第四轮）", [
         "(a) 把 O_scale 全部 (地图, |A|, λ) 格子画在 (idle rate, Δ service time) "
         "平面上，按地图着色。**18/18 格同向**（符号检验 p ≈ 3.8e-6），"
         "增益 −0.86% ~ −28.01%。灰色阴影是饱和区：idle 低于约 15% 时 "
         "ΔL2/L2 ≈ −1%（L2 反而变差），增益消失 —— 倒 U 型左端在大图上原样复现。",
         "(b) 在 idle ∈ [35%, 50%] 这一档内画 Δ vs 地图规模（对数轴）："
         "B1（5 699 free）给 −14.61/−22.70%，B3（22 599 free，面积 4 倍）给 "
         "−16.01/−23.73% —— 几乎相同，倾向「增益只是 idle rate 的函数、与尺度无关」"
         "（H-scale 的坍缩分支）。",
         "统计口径：本组 3 seeds，**Wilcoxon 符号秩在 n=3 时数学上不可能显著**"
         "（精确最小双侧 p = 0.25），故改用配对 t（df=2，已对标准 t 表校验到 <5e-5）。",
         "开销必须如实报告：ms/decision W4 6–32、W5 28–52、B1 57–91、B3 102–314，"
         "而 W1 <2 ms —— 不能只报小图数字。",
     ]),
]

HEAD = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>论文配图 Fig.1–Fig.8</title>
<style>
:root{--ink:#1e293b;--sub:#64748b;--bg:#f1f5f9;--card:#fff;--line:#e2e8f0}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px 64px;background:var(--bg);color:var(--ink);
 font-family:"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
 line-height:1.7}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:24px;margin:0 0 6px}
.lead{color:var(--sub);font-size:14px;margin-bottom:28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:22px 24px;margin-bottom:26px;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.card h2{font-size:17px;margin:0 0 14px;padding-bottom:10px;
 border-bottom:1px solid var(--line)}
.svgbox{background:var(--card);border-radius:8px;overflow:auto;margin-bottom:14px}
.svgbox svg{display:block;width:100%;height:auto}
ul{margin:0;padding-left:20px}
li{font-size:13.5px;color:#334155;margin:6px 0}
.pill{display:inline-block;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;
 border-radius:6px;padding:1px 8px;font-size:12px;font-weight:600;margin-left:8px}
footer{color:var(--sub);font-size:12.5px;text-align:center;margin-top:34px}
</style></head><body><div class="wrap">
<h1>论文配图 Fig.1 &ndash; Fig.8</h1>
<div class="lead">MEMIR：面向 lifelong MAPD 的交通感知空闲 agent 预定位。
数据来自 <code>results_final/raw_*.csv</code>（4076 次 run），
所有 SVG 由 <code>mapf_sim/plot_paper.py</code> 零依赖生成。</div>
"""

FOOT = """<footer>Generated by mapf_sim/build_figures_html.py &middot;
共同随机数的 open-loop 任务流 &middot;
配对 bootstrap + Wilcoxon（n&ge;6）/ 配对 t（n&lt;6）+ Holm</footer>
</div></body></html>
"""


def main():
    parts = [HEAD]
    for name, title, notes in CAPS:
        p = os.path.join(FIG, name)
        if not os.path.exists(p):
            print("missing", name)
            continue
        svg = open(p, encoding="utf-8").read()
        svg = svg.replace('font-family=\'"Segoe UI","PingFang SC",'
                          '"Microsoft YaHei","Helvetica Neue",system-ui,'
                          'sans-serif\'', "")
        parts.append('<div class="card"><h2>%s</h2>' % title)
        parts.append('<div class="svgbox">%s</div>' % svg)
        parts.append("<ul>")
        for nt in notes:
            pill = ""
            if nt.startswith("ψ=30 几乎") or "干扰强度" in nt:
                pill = '<span class="pill">核心</span>'
            parts.append("<li>%s%s</li>" % (nt, pill))
        parts.append("</ul></div>")
    parts.append(FOOT)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("->", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
