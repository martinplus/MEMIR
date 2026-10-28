"""定位 |A| 较大时 memir 反劣于 static 的真实成因。

已确认事实（A_main）：
  - resp_dist：memir 在所有 fleet 规模上都严格优于 static（n=32: 3.26 vs 5.77）
  - 但 mean_pending / service_time 反而更差（2.90 vs 2.81；37.08 vs 35.89）
  - detour：memir 1.03-1.05 vs static 1.01 -> 存在额外绕行代价
  - rest_moves：memir 613 vs static 329 -> 改位频率差近 2 倍

候选假设：
  H1 搬迁代价：多余改位自身消耗 agent 时间并制造冲突 -> 增大 β / move_thresh 应能修复
  H2 目标错误（1-中位数诅咒）：簇聚损害覆盖 -> coverage depletion 应能修复（已证伪）
本脚本对 H1 做扫描，并用 marg-benefit 判据给出与 fleet 规模无关的统一解释。
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


def run(j):
    j = dict(j)
    m = j.pop("map_name", "W1")
    tag = j.pop("__tag", "")
    n = j.pop("__n", 0)
    seed = j.get("seed", 0)
    wh = get_map(m)
    try:
        r = MAPDSim(wh, SimConfig(**j)).run()
    except Exception as e:                       # noqa: BLE001
        return dict(tag=tag, n=n, seed=seed, svc=-1, err=repr(e))
    return dict(tag=tag, n=n, seed=seed, svc=r["service_time"],
                resp=r["resp_dist"], mv=r["rest_moves"],
                det=r["detour"], pend=r["mean_pending"],
                idle=r["idle_ratio"], afail=r["plan_attempt_fail"],
                dec=r["decide_cnt"], err="")


def mk(tag, n, s, **kw):
    j = dict(seed=s, n_agents=n, map_name="W1")
    j.update({k: v for k, v in BASE.items() if k != "map_name"})
    j.update(dict(rest_policy="exp", demand_mode="ema", tau_phi3=100))
    j.update(kw)
    j.update(__tag=tag, __n=n)
    return j


def main():
    ns = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else [16, 24, 32, 40]
    seeds = 5
    combos = [("beta0.5", dict(rest_beta=0.5)),
              ("beta1.5", dict(rest_beta=1.5)),
              ("beta4", dict(rest_beta=4.0)),
              ("beta12", dict(rest_beta=12.0)),
              ("b.5_mt2", dict(rest_beta=0.5, move_thresh=2.0)),
              ("b.5_mt6", dict(rest_beta=0.5, move_thresh=6.0)),
              ("b.5_dep.15", dict(rest_beta=0.5, rest_deplete=0.15)),
              ("b.5_dep.4", dict(rest_beta=0.5, rest_deplete=0.4))]
    ref = [("static", dict(rest_policy="static", demand_mode="none")),
           ("random", dict(rest_policy="random", demand_mode="none"))]
    jobs = []
    for n in ns:
        for tag, kw in combos:
            for s in range(seeds):
                jobs.append(mk(tag, n, s, **kw))
        for tag, kw in ref:
            for s in range(seeds):
                j = mk(tag, n, s, **kw)
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
        agg[(r["n"], r["tag"])].append(r)
    tags = [t for t, _ in combos] + [t for t, _ in ref]
    print("\n%-4s %-11s | %7s %7s %7s %7s %7s" %
          ("n", "variant", "svc", "resp", "moves", "detour", "idle"))
    for n in ns:
        for tg in tags:
            g = agg.get((n, tg))
            if not g:
                continue
            f = lambda k: sum(x[k] for x in g) / len(g)   # noqa: E731
            print("%-4d %-11s | %7.2f %7.2f %7.0f %7.3f %7.2f" %
                  (n, tg, f("svc"), f("resp"), f("mv"), f("det"), f("idle")))
        print("-" * 62)


if __name__ == "__main__":
    main()
