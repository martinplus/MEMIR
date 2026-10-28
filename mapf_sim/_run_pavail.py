# -*- coding: utf-8 -*-
"""P_avail —— 检验 O_scale 大 fleet 反转是否是「执行器可用性规则」的产物。

背景：
  O_scale 360 runs 中，B1 / |A|>=128 / idle>=74% 两格 memir 反而比 static 差
  （+24.3% 与 +1.6%）。三段分解显示反转**全部**落在 leg_assign（派单排队段），
  leg_approach 与 leg_haul 均不变差；plan_fail == 0，排除规划失败级联。
  根因定位到 sim._assign_and_plan：默认把「正在驶向泊位途中」的空闲 agent
  排除在派单之外（`a["idx"] >= len(a["path"]) - 1`）。|A| 大、空闲率高时，
  memir 让大量 agent 持续改位 ⇒ 同时可用的 agent 数骤降 ⇒ 排队段暴涨。

  现实中 AGV 在移库途中完全可以被重新下任务，因此这条规则是**保守建模**，
  而不是机制。本实验用 assign_moving=1 打开「途中可派单」，检验反转是否消失。

设计（只跑 assign_moving=1，assign_moving=0 的行直接复用 O_scale 已有行）：
    B1/128/λ=0.32  反转格   static, memir0, memir × 3 seeds =  9
    B1/256/λ=0.64  反转格   static, memir            × 3 seeds =  6
    B1/64 /λ=0.16  正常格   static, memir0, memir    × 3 seeds =  9
    W4/128/λ=0.64  正常格   static, memir            × 3 seeds =  6
合计 30 runs。判据：反转格增益由正转负，正常格增益基本不变。
"""
import io, os, sys, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUTCSV = os.path.join(ROOT, "results_final", "raw_P_avail.csv")
LOG = os.path.join(ROOT, "_pavail_log.txt")


def say(s):
    print(s, flush=True)
    try:
        with io.open(LOG, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    except Exception:
        pass


def main():
    import run_suite as rs

    t0 = time.time()
    try:
        os.remove(LOG)
    except OSError:
        pass

    # ---- 0. 冒烟：先跑一个最小格子确认 assign_moving=1 不崩 ----
    say("=== smoke ===")
    sm = rs._mk("P_avail", "memir", 0, 32, pol="memir", map_name="W4",
                T=400, warmup=100, lmbda=0.08, assign_moving=1)
    r = rs._run_job(sm)
    say("  smoke err=%s svc=%.1f assign=%.2f moving_idle=%s"
        % (r.get("error", "")[:80], float(r.get("service_post", -1)),
           float(r.get("leg_assign", -1)), r.get("moving_idle")))
    if r.get("error"):
        say("  SMOKE FAILED -> abort")
        return

    GRID = (("B1", 128, 0.32, 1500, 500, ("static", "memir0", "memir")),
            ("B1", 256, 0.64, 1500, 500, ("static", "memir")),
            ("B1", 64, 0.16, 1500, 500, ("static", "memir0", "memir")),
            ("W4", 128, 0.64, 2000, 600, ("static", "memir")))
    jobs = []
    for mname, n, lam, T, warm, pols in GRID:
        for pol in pols:
            for s in (0, 1, 2):
                jobs.append(rs._mk("P_avail", pol, s, n, pol=pol,
                                   map_name=mname, T=T, warmup=warm,
                                   lmbda=lam, assign_moving=1))
    say("jobs = %d" % len(jobs))
    workers = int(os.environ.get("WORKERS", "6"))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(rs._run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 5 == 0 or i == len(jobs):
                say("  [%3d/%3d %6.0fs]" % (i, len(jobs), time.time() - t0))
    err = [r for r in rows if r.get("error")]
    say("errors = %d / %d" % (len(err), len(rows)))
    for r in err[:5]:
        say("   %s" % str(r.get("error"))[:160])
    good = [r for r in rows if not r.get("error")]
    # 追加写
    old = []
    fields = []
    if os.path.exists(OUTCSV):
        with io.open(OUTCSV, encoding="utf-8", errors="replace") as f:
            rd = csv.DictReader(f)
            old = [x for x in rd]
            fields = list(rd.fieldnames or [])
    for r in good:
        for k in r:
            if k not in fields:
                fields.append(k)
    fields = sorted(fields)
    with open(OUTCSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in old + good:
            w.writerow(r)
    say("wrote %d (+%d old) -> %s" % (len(good), len(old), OUTCSV))
    say("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
