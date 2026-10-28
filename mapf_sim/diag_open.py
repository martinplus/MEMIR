"""Phase 3 诊断：在什么条件下「知道未来任务分布」才比「简单分散」更有价值？

第一次诊断的关键发现（必须正面处理，不能绕开）：
    nearest 46.71 | random 41.76 | static 37.55 | oracle 37.16
即——**oracle 相对 static 的增益只有 1%**，绝大部分收益来自"分散"本身，
而非来自"知道需求"。这与 Phase 1/2 的教训同构：信息素只是搭了一个别人也
能拿到的便宜。因此必须找到 **static 拿不到、只有可自适应记忆能拿到** 的条件。

候选条件（逐个否决或确认）：
    C1 动作空间：停泊位置从 non-task endpoint 扩展到全部通道格
    C2 资源稀缺：减少 agent 数，使空闲 agent 数量 K 变小（覆盖半径变大）
    C3 需求非平稳：热点随时间漂移 —— static 完全无法跟随，这是决定性条件
"""
import math
import statistics as st
import sys
import time

sys.path.insert(0, r"C:\Users\HunterChaw\WorkBuddy\2026-09-24-18-52-47\mapf_sim")

from warehouse import Warehouse
from sim import MAPDSim, SimConfig

_WH = None


def get_wh():
    global _WH
    if _WH is None:
        _WH = Warehouse(width=41, height=21, station_stride=6)
    return _WH


def run(seeds, **kw):
    wh = get_wh()
    return [MAPDSim(wh, SimConfig(seed=s, **kw)).run() for s in seeds]


def build_oracle_lam(seeds, **kw):
    wh = get_wh()
    cum = [0.0] * wh.n_regions
    for r in run(seeds, **kw):
        c = r["demand_cum"]
        for i in range(wh.n_regions):
            cum[i] += c[i]
    tot = sum(cum)
    if tot <= 0:
        return [1.0 / wh.n_regions] * wh.n_regions
    return [x / tot for x in cum]


def stat(rs, key="service_post"):
    v = [r[key] for r in rs if r[key] > 0]
    return (st.mean(v), st.pstdev(v)) if v else (float("nan"), 0.0)


def spread_of(p):
    H = -sum(x * math.log(x) for x in p if x > 1e-15)
    return 1.0 - H / math.log(len(p))


SEEDS = list(range(8))


def scenario(tag, base, policies, seeds=SEEDS):
    print("=" * 78)
    print("### %s" % tag)
    print("    " + " ".join("%s=%s" % (k, v) for k, v in sorted(base.items())))
    orc = build_oracle_lam(seeds, rest_policy="nearest",
                           **{k: v for k, v in base.items()})
    print("    oracle spread=%.3f  top1=%.3f" % (spread_of(orc), max(orc)))
    out = {}
    for name, kw in policies:
        cfg = dict(base)
        cfg.update(kw)
        if name.lower() == "oracle":
            cfg["oracle_lam"] = orc
        rs = run(seeds, **cfg)
        out[name] = rs
    bname = policies[0][0]
    bm, bs = stat(out[bname])
    bidle = st.mean([r["idle_ratio"] for r in out[bname]]) * 100
    print("    %-8s svc=%7.2f (sd %4.2f)  idle=%4.1f%%  pend=%5.2f"
          % (bname, bm, bs, bidle, st.mean([r["mean_pending"] for r in out[bname]])))
    for name, _ in policies[1:]:
        m, s = stat(out[name])
        print("    %-8s svc=%7.2f (sd %4.2f)  Δ=%+6.2f%%"
              % (name, m, s, (m / bm - 1) * 100))
    return out, bm


POL = [("nearest", dict(rest_policy="nearest")),
       ("static", dict(rest_policy="static")),
       ("oracle", dict(rest_policy="oracle", demand_mode="oracle"))]


def main():
    print(get_wh().summary())
    print()
    common = dict(T=1500, warmup=500, zipf_alpha=2.0, arrival_mode="open",
                  lmbda=0.15, capacity=1, demand_mode="none")

    # C1: 动作空间
    scenario("C1a  rest_set=park (20 agents)",
             dict(common, n_agents=20, rest_set="park"), POL)
    scenario("C1b  rest_set=any  (20 agents)",
             dict(common, n_agents=20, rest_set="any"), POL)
    print()

    # C2: 资源稀缺
    scenario("C2a  rest_set=any, 8 agents",
             dict(common, n_agents=8, rest_set="any"), POL)
    scenario("C2b  rest_set=any, 5 agents",
             dict(common, n_agents=5, rest_set="any"), POL)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print()
    print("elapsed %.0fs" % (time.time() - t0))
