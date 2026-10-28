"""诊断：区域级（或转移级）载体本身能否承载有效信息？

做法（两阶段）：
  1. 用 nn 基线跑若干 seed，收集每个批次的 (区域序列, 效率 o)
  2. 构造事后 Oracle 场 τ*[r][r'] = E[o | 边(r,r') 出现在该批] − E[o]
  3. 用 τ* 调制重跑，测几何序列长度与吞吐

判据：
  - 若 Oracle 场的 geom/吞吐 显著优于 nn  -> 区域级载体有效，
    问题只在学习动力学（沉积/蒸发的定义），继续调即可
  - 若 Oracle 场也无效                    -> 载体粒度不够，
    序列优化需要实例级（具体位置）信息，必须换载体
"""
import sys
import statistics

sys.path.insert(0, r"C:\Users\HunterChaw\WorkBuddy\2026-09-24-18-52-47\mapf_sim")

from warehouse import Warehouse           # noqa: E402
from sim import MAPDSim, SimConfig        # noqa: E402
import seq                                # noqa: E402

WH = Warehouse(width=41, height=21, station_stride=6)
N, T, WARM, PF = 40, 1200, 400, 8.0
SEEDS = list(range(6))


def run(cap, mode, oracle=None, gamma=2.0, seeds=SEEDS):
    outs = []
    for s in seeds:
        cfg = SimConfig(n_agents=N, T=T, seed=s, warmup=WARM, zipf_alpha=1.0,
                        capacity=cap, seq_mode=mode, pending_factor=PF,
                        ph2_mode=("oracle" if oracle is not None else "none"),
                        ph2_oracle=oracle, gamma=gamma)
        outs.append(MAPDSim(WH, cfg).run())
    return outs


def collect(cap):
    """阶段 1：收集 (区域序列, 效率) 样本。"""
    import sim as S
    samples = []

    orig = S.MAPDSim._finish_batch

    def patched(self, a, b, t):
        c = len(b["tasks"])
        L = t - b["start_t"]
        margin = max(L - b["first_leg"], 0)
        per_task = margin / max(c - 1, 1) if c > 1 else margin
        o = seq.MARGIN_SCALE / (seq.MARGIN_SCALE + per_task)
        samples.append((list(b["region_seq"]), o))
        return orig(self, a, b, t)

    S.MAPDSim._finish_batch = patched
    try:
        run(cap, "nn")
    finally:
        S.MAPDSim._finish_batch = orig
    return samples


def build_field(samples, n, min_cnt=15):
    """阶段 2：构造事后 Oracle 场（效率优势）。"""
    tot = [0.0] * 1
    s_sum = [[0.0] * n for _ in range(n)]
    s_cnt = [[0] * n for _ in range(n)]
    all_o = []
    for rs, o in samples:
        all_o.append(o)
        seen = set()
        for i in range(len(rs) - 1):
            r, s2 = rs[i], rs[i + 1]
            if r < 0 or s2 < 0 or (r, s2) in seen:
                continue
            seen.add((r, s2))
            s_sum[r][s2] += o
            s_cnt[r][s2] += 1
    gmean = statistics.mean(all_o) if all_o else 0.0
    field = []
    nz = 0
    for r in range(n):
        row = []
        for s2 in range(n):
            if s_cnt[r][s2] >= min_cnt:
                v = s_sum[r][s2] / s_cnt[r][s2] - gmean
                if abs(v) > 1e-9:
                    nz += 1
            else:
                v = 0.0
            row.append(v)
        field.append(row)
    vals = [v for row in field for v in row]
    return field, gmean, nz, (max(vals) - min(vals)) if vals else 0.0


def main():
    for cap in [3, 4]:
        print("=" * 74)
        print("cap = %d" % cap)
        samples = collect(cap)
        print("  收集到批次样本: %d" % len(samples))
        field, gmean, nz, rng_ = build_field(samples, WH.n_regions)
        print("  全局平均效率 o_bar = %.4f" % gmean)
        print("  Oracle 场非零边数 = %d, 极差 = %.4f" % (nz, rng_))

        base = run(cap, "nn")
        bm = statistics.mean([r["throughput_post"] for r in base])
        bg = statistics.mean([r["geom_per_batch"] for r in base])
        print("  nn      geom/b = %7.2f   thr = %7.1f" % (bg, bm))
        for g in [1.0, 5.0, 20.0]:
            o = run(cap, "ph", oracle=field, gamma=g)
            m = statistics.mean([r["throughput_post"] for r in o])
            gg = statistics.mean([r["geom_per_batch"] for r in o])
            print("  oracle g=%-5.1f geom/b = %7.2f (%+.2f%%)  thr = %7.1f (%+.2f%%)"
                  % (g, gg, (gg / bg - 1) * 100, m, (m / bm - 1) * 100))
        orc = run(cap, "oracle")
        m = statistics.mean([r["throughput_post"] for r in orc])
        gg = statistics.mean([r["geom_per_batch"] for r in orc])
        print("  seq-oracle   geom/b = %7.2f (%+.2f%%)  thr = %7.1f (%+.2f%%)"
              % (gg, (gg / bg - 1) * 100, m, (m / bm - 1) * 100))


if __name__ == "__main__":
    main()
