# -*- coding: utf-8 -*-
"""在 Windows 下运行子脚本并把 stdout/stderr 落盘为 UTF-8 文本。

本机的 Bash shim 不可靠、PowerShell 的 Out-File 会把中文写乱，所以统一用
Python 子进程 + io.open(encoding="utf-8") 收输出。
"""
import io
import os
import subprocess
import sys

PY = r"C:\Users\HunterChaw\.workbuddy\binaries\python\versions\3.13.12\python.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "mapf_sim")

target = sys.argv[1]
outfile = os.path.join(ROOT, sys.argv[2])
cwd = os.path.join(ROOT, "mapf_sim")
r = subprocess.run([PY, os.path.join(D, target)], cwd=cwd,
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
msg = r.stdout + "\n[STDERR]\n" + r.stderr + "\n[EXIT=%d]\n" % r.returncode
io.open(outfile, "w", encoding="utf-8").write(msg)
