"""生成 Phase 2 整合报告页：把 5 张 SVG 图表内嵌进 Markdown 报告的 HTML。"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results_mt")

SRC = os.path.join(BASE, "实验结果报告_MTMAPD序列级信息素.html")
DST = os.path.join(RES, "report.html")

FIGS = [
    ("fig_capacity.svg", "E1 容量扫描"),
    ("fig_construct.svg", "E2 序列构造器对比"),
    ("fig_ph.svg", "E3 信息素参数扫描"),
    ("fig_cond.svg", "E4 条件矩阵"),
    ("fig_drift.svg", "E5 非平稳漂移"),
]

CSS = """
body{margin:0;background:#f6f7f9;color:#2c3446;
     font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.75;}
.wrap{max-width:1040px;margin:0 auto;padding:32px 24px 80px;background:#fff;
      box-shadow:0 1px 3px rgba(0,0,0,.06);}
h1{font-size:26px;margin:0 0 8px;}
h2{font-size:20px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid #eef0f3;}
h3{font-size:15.5px;margin:24px 0 8px;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px;}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:left;}
th{background:#f2f4f7;font-weight:600;}
code{background:#f2f4f7;padding:2px 5px;border-radius:4px;font-size:12.5px;}
pre{background:#f7f8fa;padding:12px 14px;border-radius:8px;overflow-x:auto;}
blockquote{border-left:4px solid #185FA5;background:#f4f8fd;margin:14px 0;
           padding:10px 16px;border-radius:0 8px 8px 0;}
.fig{margin:26px 0;padding:16px;background:#fff;border:1px solid #e3e6ea;
     border-radius:10px;}
.fig h3{margin:0 0 10px;font-size:15px;}
.badge{display:inline-block;background:#fdecec;color:#A32D2D;font-size:12px;
       padding:3px 10px;border-radius:12px;margin-left:8px;}
.good{display:inline-block;background:#e6f4ef;color:#0F6E56;font-size:12px;
      padding:3px 10px;border-radius:12px;margin-left:8px;}
"""


def svg_inline(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return "<p>（缺图：%s）</p>" % name
    s = io.open(p, encoding="utf-8").read()
    i = s.find("<svg")
    return s[i:]


def main():
    body = ""
    if os.path.exists(SRC):
        h = io.open(SRC, encoding="utf-8").read()
        i = h.find("<body")
        if i >= 0:
            j = h.find(">", i)
            k = h.find("</body>")
            body = h[j + 1:k] if k > j else h[j + 1:]

    parts = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             "<title>Phase 2: MT-MAPD + 序列级信息素 — 实验结果</title>",
             "<style>%s</style></head><body><div class=\"wrap\">" % CSS]
    parts.append('<h1>Phase 2：MT-MAPD（带容量）+ 序列级信息素'
                 '<span class="good">容量 +95.8%</span>'
                 '<span class="badge">信息素 0%</span></h1>')
    parts.append("<p style='color:#6b7280;font-size:13.5px'>"
                 "534 次仿真运行 · 零第三方依赖 · "
                 "<a href='../实验结果报告_MTMAPD序列级信息素.html'>完整文字报告</a> · "
                 "<a href='raw.csv'>原始数据</a></p>")
    parts.append("<h2>图表</h2>")
    for fn, cap in FIGS:
        parts.append('<div class="fig"><h3>%s</h3>%s</div>' % (cap, svg_inline(fn)))
    parts.append("<h2>完整报告</h2>")
    parts.append(body)
    parts.append("</div></body></html>")
    io.open(DST, "w", encoding="utf-8").write("\n".join(parts))
    print("written", DST, os.path.getsize(DST), "bytes")


if __name__ == "__main__":
    main()
