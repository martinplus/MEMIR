# -*- coding: utf-8 -*-
"""校验 paired_t / student_t_sf2 与已知 t 分布临界值一致。"""
import io, os, math
from stats_util import student_t_sf2, paired_t, betainc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_tstat_check.txt")

# (df, t_crit_two_sided_0.05, t_crit_two_sided_0.01) 标准 t 表
REF = [(1, 12.706, 63.657), (2, 4.303, 9.925), (3, 3.182, 5.841),
       (5, 2.571, 4.032), (10, 2.228, 3.169), (30, 2.042, 2.750)]


def main():
    out = ["=== student_t_sf2 对照标准 t 表（临界值处 p 应≈0.05 / 0.01）==="]
    ok = True
    for df, t05, t01 in REF:
        p05 = student_t_sf2(t05, df)
        p01 = student_t_sf2(t01, df)
        e5 = abs(p05 - 0.05)
        e1 = abs(p01 - 0.01)
        bad = e5 > 2e-3 or e1 > 2e-3
        ok = ok and not bad
        out.append("df=%-3d t.05=%-8.3f p=%.5f   t.01=%-8.3f p=%.5f  %s"
                   % (df, t05, p05, t01, p01, "OK" if not bad else "MISMATCH"))
    out.append("")
    out.append("=== betainc 已知值 ===")
    # I_0.5(1,1) = 0.5 ; I_0.5(2,2) = 0.5 ; I_1(a,b) = 1
    for a, b, x, want in ((1, 1, 0.5, 0.5), (2, 2, 0.5, 0.5), (1, 3, 0.25, 1 - 0.75 ** 3),
                          (2, 3, 1.0, 1.0)):
        got = betainc(a, b, x)
        out.append("I_%.2f(%.0f,%.0f) = %.8f  (want %.8f)  %s"
                   % (x, a, b, got, want, "OK" if abs(got - want) < 1e-6 else "MISMATCH"))
    out.append("")
    out.append("=== paired_t 示例 ===")
    for d in ([-1, -2, -3], [-10, -12, -11], [-1, -1, -1], [1, -5, -2]):
        t, df, p, m, sd = paired_t(d)
        out.append("d=%-18s mean=%+.3f sd=%.3f t=%+.3f df=%d p=%.5f" % (d, m, sd, t, df, p))
    out.append("")
    out.append("ALL_OK" if ok else "CRITICAL_VALUES_MISMATCH")
    io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
