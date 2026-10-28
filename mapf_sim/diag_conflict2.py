"""冲突根因定位：dump 冲突格上的全部预约事件历史。

思路：冲突发生时双方都声称"合法"，说明其中一方的信息是过时的。
把该格上发生的每一次 permanent 声明 / path 注册 / agent 移除按时间排出来，
就能看出是谁先、谁后，以及为什么检查没有拦住。
"""
import sys

sys.path.insert(0, r"C:\Users\HunterChaw\WorkBuddy\2026-09-24-18-52-47\mapf_sim")

from planner import ReservationTable
from warehouse import Warehouse
from sim import MAPDSim, SimConfig

_add_path = ReservationTable.add_path
_remove_agent = ReservationTable.remove_agent


def add_path(self, agent, path, t0, permanent=True):
    if getattr(self, "_watch", None):
        for i, (x, y) in enumerate(path):
            if (x, y) == self._watch:
                self._logcell((t0 + i, "ADDpath a%d t=%d%s"
                               % (agent, t0 + i,
                                  " PERM" if (permanent and i == len(path) - 1) else "")))
    if permanent and getattr(self, "_watch", None) and path:
        lx, ly = path[-1]
        if (lx, ly) == self._watch:
            self._logcell((t0 + len(path) - 1, "PERM a%d start=%d"
                           % (agent, t0 + len(path) - 1)))
    return _add_path(self, agent, path, t0, permanent)


def remove_agent(self, agent):
    if getattr(self, "_watch", None):
        self._logcell((getattr(self, "_t", 0), "REMOVE a%d" % agent))
    return _remove_agent(self, agent)


ReservationTable.add_path = add_path
ReservationTable.remove_agent = remove_agent
ReservationTable._logcell = lambda self, rec: None

_detect = MAPDSim._detect_conflicts


def patched(self):
    seen = {}
    for a in self.agents:
        p = a["pos"]
        if p in seen:
            rec = self.stats.setdefault("CLOG", [])
            if len(rec) < 3:
                t = getattr(self, "_cur_t", -1)
                rec.append((t, p, seen[p], a["id"]))
        seen[p] = a["id"]
    return _detect(self)


MAPDSim._detect_conflicts = patched


def main():
    wh = Warehouse(width=41, height=21, station_stride=6)
    # 已知冲突点（来自 diag_conflict.py）：t=298 处 Cell=(7,17)
    watch = (7, 17)
    cfg = SimConfig(n_agents=6, T=320, seed=1, warmup=100, zipf_alpha=2.0,
                    arrival_mode="open", lmbda=0.06, capacity=1, rest_set="any")
    sim = MAPDSim(wh, cfg)
    events = []
    sim.rt._watch = watch
    sim.rt._logcell = lambda rec: events.append(rec)
    r = sim.run()
    print("conflicts:", r["conflicts"])
    print("=== event history on %s (last 60, chronological) ===" % (watch,))
    events.sort(key=lambda x: x[0])
    for tt, desc in events[-60:]:
        print("  t=%-6d %s" % (tt, desc))
    print()
    print("permanent now:", sim.rt.permanent.get(watch))
    ts = sorted(sim.rt.by_cell.get(watch, ()))
    print("reserved times:", ts)


if __name__ == "__main__":
    main()
