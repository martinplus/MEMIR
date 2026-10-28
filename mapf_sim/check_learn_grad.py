"""用有限差分校验 AttnPointer 的手写解析梯度。

若解析梯度有误，学习基线就退化成"随机扰动"，实验结论不可信 ——
因此这个校验是学习类基线实验的前置条件。

校验两个量：
    J1 = s_a            （REINFORCE 中的 ∇ log π 第一项）
    J2 = Σ_c π_c · s_c  （第二项）
对全部参数（W_e / W_a / w1 / w2 / w3）逐元素做中心差分比对。
"""
import math
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from learn import AttnPointer, LinearScorer     # noqa: E402


def _params(m):
    """把全部参数摊平成一维列表，返回 (values, setter)。"""
    vals = []
    for row in m.W_e:
        vals.extend(row)
    for row in m.W_a:
        vals.extend(row)
    for name in ("w1", "w2", "w3"):
        vals.extend(getattr(m, name))
    return vals


def _set(m, vals):
    i = 0
    for row in m.W_e:
        for k in range(len(row)):
            row[k] = vals[i]
            i += 1
    for row in m.W_a:
        for k in range(len(row)):
            row[k] = vals[i]
            i += 1
    for name in ("w1", "w2", "w3"):
        v = getattr(m, name)
        for k in range(len(v)):
            v[k] = vals[i]
            i += 1


def _grad_flat(m, f, a, which):
    ga, ge = m._grad_single(f, a)
    out = []
    for name, key in (("We", "We"), ("Wa", "Wa"), ("w1", "w1"),
                      ("w2", "w2"), ("w3", "w3")):
        g = (ga if which == "single" else ge)[key]
        if key in ("We", "Wa"):
            for row in g:
                out.extend(row)
        else:
            out.extend(g)
    return out


def _obj(m, X, z, a, which):
    f = m.forward(X, z)
    if which == "single":
        return f["S"][a]
    return sum(p * s for p, s in zip(f["pi"], f["S"]))


def main():
    rng = random.Random(20260926)
    dx, dz, dh, K = 8, 4, 8, 6
    X = [[rng.gauss(0, 1) for _ in range(dx)] for _ in range(K)]
    z = [rng.gauss(0, 1) for _ in range(dz)]
    worst = 0.0
    for trial in range(6):
        m = AttnPointer(dx, dz, dh, lr=0.05, temp=0.7, seed=trial)
        a = trial % K
        for which in ("single", "E"):
            f = m.forward(X, z)
            g_ana = _grad_flat(m, f, a, which)
            vals = _params(m)
            g_num = []
            eps = 1e-6
            for i in range(len(vals)):
                vp = list(vals)
                vp[i] += eps
                _set(m, vp)
                jp = _obj(m, X, z, a, which)
                vm = list(vals)
                vm[i] -= eps
                _set(m, vm)
                jm = _obj(m, X, z, a, which)
                g_num.append((jp - jm) / (2 * eps))
                _set(m, vals)
            # 相对误差（在量级足够大的元素上比较）
            errs = []
            for ga_, gn_ in zip(g_ana, g_num):
                den = max(1.0, abs(ga_), abs(gn_))
                errs.append(abs(ga_ - gn_) / den)
            e = max(errs)
            worst = max(worst, e)
            print("trial=%d which=%-6s  max_rel_err=%.3e  n=%d"
                  % (trial, which, e, len(errs)))
    print("WORST_REL_ERR=%.3e" % worst)
    print("OK" if worst < 1e-5 else "FAIL")

    # ---- 顺带校验线性打分器的 REINFORCE 梯度方向 ----
    lin = LinearScorer(dx, lr=0.1, temp=0.5, seed=1)
    lin.theta = [rng.gauss(0, 0.5) for _ in range(dx)]
    u0 = lin.scores(X)
    j = max(range(K), key=lambda i: u0[i])
    pi = lin._softmax(u0, lin.T)
    before = sum(p * s for p, s in zip(pi, u0))
    lin.update(X, j, pi, adv=1.0)
    pi2 = lin._softmax(lin.scores(X), lin.T)
    after = pi2[j]
    print("linear: pi(best) before=%.4f after=%.4f  %s"
          % (pi[j], after, "OK" if after >= pi[j] else "FAIL"))


if __name__ == "__main__":
    main()
