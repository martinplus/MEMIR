# -*- coding: utf-8 -*-
"""扫描指定 markdown 文件中的非中日常用字符（西里尔 / 谚文 / 假名 / 阿拉伯文）。"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PATS = {
    "cyr": re.compile(r"[\u0400-\u04FF]"),
    "hangul": re.compile(r"[\uAC00-\uD7AF]"),
    "kana": re.compile(r"[\u3040-\u30FF]"),
    "arab": re.compile(r"[\u0600-\u06FF]"),
}

SUSPECT = ["Omega", "ISA", "Porter", "tiene", "onen", "Vereinfacht",
           "Mou", "stitch", "message passing", "Omega"]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "论文大纲_成文版.md"
    p = os.path.join(ROOT, name)
    lines = io.open(p, encoding="utf-8").read().splitlines()
    out = []
    n = 0
    for i, ln in enumerate(lines, 1):
        hit = []
        for k, pat in PATS.items():
            if pat.search(ln):
                hit.append(k)
        for w in SUSPECT:
            if w in ln:
                hit.append("word:" + w)
        if hit:
            n += 1
            out.append("%4d [%s] %s" % (i, ",".join(hit), ln[:160]))
    out.append("---- total flagged lines = %d / %d" % (n, len(lines)))
    io.open(os.path.join(ROOT, "_scanbad.txt"), "w",
            encoding="utf-8").write("\n".join(out))


if __name__ == "__main__":
    main()
