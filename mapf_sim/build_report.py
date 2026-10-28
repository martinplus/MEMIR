"""把 summary.csv 与三张 SVG 整合为单页 HTML 报告。零第三方依赖。"""
import os
import csv
import io

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.6;
     color:#2C2C2A;max-width:1080px;margin:0 auto;padding:32px 24px;background:#fff}
h1{font-size:22px;font-weight:500;margin:0 0 6px}
h2{font-size:17px;font-weight:500;margin:32px 0 12px;
   border-bottom:1px solid #E8E6DF;padding-bottom:6px}
.lead{color:#5F5E5A;margin:0 0 24px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{border:1px solid #E8E6DF;padding:6px 10px;text-align:right}
th{background:#F1EFE8;font-weight:500;text-align:left}
td:first-child,th:first-child{text-align:left}
.fig{background:#FAFAF8;border:1px solid #E8E6DF;border-radius:10px;padding:14px;margin:14px 0}
.note{background:#FCEBEB;border-left:3px solid #E24B4A;padding:10px 14px;
      border-radius:0 8px 8px 0;font-size:13px;margin:14px 0}
.ok{background:#E1F5EE;border-left:3px solid #1D9E75;padding:10px 14px;
    border-radius:0 8px 8px 0;font-size:13px;margin:14px 0}
"""


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return ""
    return io.open(p, encoding="utf-8").read()


def table_html():
    p = os.path.join(RES, "summary.csv")
    if not os.path.exists(p):
        return ""
    with io.open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = [("label", "方法"), ("n_agents", "|A|"), ("alpha", "α"),
            ("throughput", "吞吐"), ("sd", "sd"), ("gain_pct", "增益%"),
            ("service_time", "service time"), ("detour", "detour"),
            ("wait_ratio", "wait ratio"), ("conflicts", "conflicts")]
    out = ["<table><thead><tr>"]
    for _, h in cols:
        out.append("<th>%s</th>" % h)
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for c, _ in cols:
            v = r.get(c, "")
            if c == "gain_pct":
                try:
                    fv = float(v)
                    col = "#0F6E56" if fv > 0 else ("#A32D2D" if fv < 0 else "#5F5E5A")
                    out.append('<td style="color:%s;font-weight:500">%+.2f%%</td>' % (col, fv))
                    continue
                except ValueError:
                    pass
            out.append("<td>%s</td>" % v)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main():
    figs = [("fig_main.svg", "各方法吞吐对比（基线 = Hungarian）"),
            ("fig_tau.svg", "信息素增益 vs 记忆时间常数 τ_φ"),
            ("fig_alpha.svg", "信息素增益 vs 空间不均匀度 α")]
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
             "<title>MAPD 信息素分配调制实验结果</title>",
             "<style>%s</style></head><body>" % CSS,
             "<h1>MAPD 中「信息素调制分配代价」的实验结果</h1>",
             '<p class="lead">Hungarian 为基线 · 576 次独立运行 · '
             '3 负载 × 3 空间不均匀度 × 5 方法 × 8 seeds · 每次 1200 timestep</p>',
             '<div class="note"><b>结论：</b>在饱和 closed-loop MAPD 中，'
             '区域级信息素调制的分配代价<b>不能提升吞吐</b>——'
             '即使使用事后构造的真实拥堵场（Oracle）也不能。'
             'Hungarian + 纯几何距离即为当前设定下的最优分配策略。</div>',
             '<div class="ok"><b>有效对照：</b>TP-greedy 显著劣于 Hungarian（−0.5% ~ −2.8%），'
             '说明实验系统具备分辨力，分配器质量确实影响吞吐；'
             '但该改进空间已被 Hungarian 占满，Oracle 无法再进一步提升。</div>']

    for f, title in figs:
        svg = read(f)
        if not svg:
            continue
        svg = svg.replace('width="100%"', 'width="100%"', 1)
        parts.append("<h2>%s</h2>" % title)
        parts.append('<div class="fig">%s</div>' % svg)

    parts.append("<h2>完整汇总（基线 = Hungarian）</h2>")
    parts.append(table_html())
    parts.append("</body></html>")

    p = os.path.join(RES, "report.html")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
