"""负信息素（覆盖耗减）诊断：破解「1-中位数诅咒」。

现象：n≥24 时 memir 与 hindsight oracle 双双输给 uniform static。
猜想：所有 agent 独立解同一个加权 1-中位数 -> 全部吸到同一热点 -> 互相阻塞。
      这是 ride-hailing 空车调度中著名的 curse of the 1-median
      （Braverman, Dai, Lei, Shi & Xie 2019, Operations Research）。
      注意 strong evidence：连事后真值 oracle 也一起退化 -> 不是估计误差，
      是**目标函数本身**在大 fleet 下错了。

修正：在目标里做覆盖耗减（次模覆盖贪心）
        p̃_r = p̂_r · Π_{q∈occupied} (1 − η·κ(d̄(q,r))),  κ(d)=exp(-d/R)
     η=0 退化为纯 1-中位数（即现在的 memir / oracle）。

本脚本扫描 η × R，确认：
  (a) 大 fleet 能否追上/超过 static；
  (b) 小 fleet 的增益是否被吃掉。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maps import get_map                     # noqa: E402
from sim import MAPDSim, SimConfig           # noqa: E402

BASE = dict(T=4000, warmup=1000, zipf_alpha=2.0, arrival_mode="open",
            lmbda=0.08, capacity=1, rest_set="any", drift_mode="static",
            drift_param=2000, planner_mode="tp", assign_mode="hungarian",
            map_name="W1", dwell_safety=True, dwell_probe=8, preempt=0,
            tp_dummy=False, reeval_interval=40, move_thresh=0.0)

MEM = dict(rest_policy="exp", demand_mode="ema", tau_phi3=100, rest_beta=0.5)
STA = dict(rest_policy="static", demand_mode="none")
NEA = dict(rest_policy="nearest", demand_mode="none")


def job_of(pol, eta, rad, n, s, **extra):
    j = dict(seed=s, n_agents=n, map_name="W1",
             rest_deplete=eta, svc_radius=rad)
    j.update({k: v for k, v in BASE.items() if k != "map_name"})
    j.update(pol)
    j.update(extra)
    return j


def run(j):
    j = dict(j)
    m = j.pop("map_name", "W1")
    pol = j.pop("__pol", "")
    eta = j.pop("__eta", 0)
    rad = j.pop("__rad", 0)
    n = j.pop("__n", 0)
    seed = j.get("seed", 0)
    wh = get_map(m)
    try:
        r = MAPDSim(wh, SimConfig(**j)).run()
    except Exception as e:                       # noqa: BLE001
        return dict(pol=pol, eta=eta, rad=rad, n=n, seed=seed,
                    svc=-1, err=repr(e))
    return dict(pol=pol, eta=eta, rad=rad, n=n, seed=seed,
                svc=r["service_time"], afail=r["plan_attempt_fail"],
                mv=r["rest_moves"], dblk=r.get("dwell_blocked", 0),
                idle=r["idle_ratio"], err="")


def main():
    ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else [6, 8, 12, 16, 24, 32, 40]
    seeds = 5
    combos = []
    for eta in (0.0, 0.3, 0.6, 1.0):
        for rad in (3.0, 6.0):
            combos.append(("memir", eta, rad, MEM))
    combos.append(("static", 0.0, 6.0, STA))
    combos.append(("nearest", 0.0, 6.0, NEA))
    combos.append(("oracle0", 0.0, 6.0, dict(rest_policy="oracle",
                                             demand_mode="oracle")))
    combos.append(("oracle_d", 0.6, 6.0, dict(rest_policy="oracle",
                                              demand_mode="oracle")))

    jobs = []
    for n in ns:
        for pol, eta, rad, pk in combos:
            for s in range(seeds):
                j = job_of(pk, eta, rad, n, s)
                j.update(__pol=pol, __eta=eta, __rad=rad, __n=n)
                jobs.append(j)
    print("jobs=%d" % len(jobs), flush=True)
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 50 == 0:
                print("  [%4d/%4d %5.0fs]" % (i, len(jobs), time.time() - t0),
                      flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        if r["err"]:
            print("ERR", r)
            continue
        agg[(r["n"], r["pol"], r["eta"], r["rad"])].append(r)
    print("\n%-4s %-9s %5s %5s | %8s %7s %7s" %
          ("n", "pol", "eta", "R", "svc", "afail", "moves"))
    for n in ns:
        for pol, eta, rad, _ in combos:
            g = agg.get((n, pol, eta, rad))
            if not g:
                continue
            svc = sum(x["svc"] for x in g) / len(g)
            af = sum(x["afail"] for x in g) / len(g)
            mv = sum(x["mv"] for x in g) / len(g)
            print("%-4d %-9s %5.1f %5.1f | %8.2f %7.0f %7.0f" %
                  (n, pol, eta, rad, svc, af, mv))
        print("-" * 58)
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "diag_deplete.json"), "w", encoding="utf-8") as f:
        json.dump([r for r in out if not r["err"]], f, ensure_ascii=False)


if __name__ == "__main__":
    main()
