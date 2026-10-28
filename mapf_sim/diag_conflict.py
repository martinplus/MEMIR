"""精确定位 open-loop 仿真中的冲突来源。

在每个 vertex conflict 发生瞬间，dump 双方 agent 的完整状态与预约表记录，
从而判断是「规划器漏检」还是「某个 agent 缺少 reservation」。
"""
import sys

sys.path.insert(0, r"C:\Users\HunterChaw\WorkBuddy\2026-09-24-18-52-47\mapf_sim")

from warehouse import Warehouse
from sim import MAPDSim, SimConfig

_run = MAPDSim.run


def run_with_t(self, *a, **kw):
    return _run(self, *a, **kw)


MAPDSim.run = run_with_t

_detect = MAPDSim._detect_conflicts


def patched(self):
    t = getattr(self, "_cur_t", -1)
    seen = {}
    for a in self.agents:
        p = a["pos"]
        if p in seen:
            b = self.agents[seen[p]]
            rec = self.stats.setdefault("CLOG", [])
            if len(rec) < 8:
                perm = self.rt.permanent.get(p)
                here = self.rt.cell.get((p[0], p[1], t))
                rec.append(dict(
                    t=t, pos=p,
                    A=(a["id"], a["state"], a["path"] is None,
                       a.get("idx"), None if a["path"] is None else len(a["path"])),
                    B=(b["id"], b["state"], b["path"] is None,
                       b.get("idx"), None if b["path"] is None else len(b["path"])),
                    perm=perm, cell_at_t=here,
                    region=self.wh.region(*p),
                ))
        seen[p] = a["id"]
    return _detect(self)


MAPDSim._detect_conflicts = patched

orig_loop_anchor = "            self.stats[\"pending_sum\"] += len(self.pending)"


def main():
    wh = Warehouse(width=41, height=21, station_stride=6)
    for rs in ["park", "any"]:
        cfg = SimConfig(n_agents=6, T=3000, seed=1, warmup=500, zipf_alpha=2.0,
                        arrival_mode="open", lmbda=0.06, capacity=1, rest_set=rs)
        sim = MAPDSim(wh, cfg)
        r = sim.run()
        print("=" * 70)
        print("rest_set=%s  conflicts=%d" % (rs, r["conflicts"]))
        for c in sim.stats.get("CLOG", []):
            print("  t=%(t)s pos=%(pos)s region=%(region)s perm=%(perm)s cell@t=%(cell_at_t)s"
                  % c)
            print("      A=%s  B=%s" % (c["A"], c["B"]))
        sim.stats["CLOG"] = []


if __name__ == "__main__":
    main()
