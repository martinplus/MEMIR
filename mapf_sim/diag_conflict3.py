"""最终定位：给 future_clear 与 _rest_decide 加上 logging（含决策时刻）。"""
import sys

sys.path.insert(0, r"C:\Users\HunterChaw\WorkBuddy\2026-09-24-18-52-47\mapf_sim")

from planner import ReservationTable
from warehouse import Warehouse
import sim as SIM
from sim import SimConfig

WATCH = (7, 17)
LOG = []

_fc = ReservationTable.future_clear
_ap = ReservationTable.add_path


def future_clear(self, x, y, t_from, agent):
    r = _fc(self, x, y, t_from, agent)
    if (x, y) == WATCH:
        ts = sorted(self.by_cell.get((x, y), ()))
        own = {tt: self.cell.get((x, y, tt)) for tt in ts}
        # 直接与主表核对，判断 by_cell 是否失真
        raw = sorted(t for (cx, cy, t) in self.cell if (cx, cy) == (x, y))
        rawown = {tt: self.cell.get((x, y, tt)) for tt in raw}
        LOG.append(("fc", getattr(self, "_now", -1),
                    "t_from=%d a=%d -> %s | by_cell=%s | cell_raw=%s"
                    % (t_from, agent, r, own, rawown)))
    return r


def add_path(self, agent, path, t0, permanent=True):
    if permanent:
        lx, ly = path[-1]
        if (lx, ly) == WATCH:
            LOG.append(("perm", getattr(self, "_now", -1),
                        "agent=%d start=%d len=%d now=%s"
                        % (agent, t0 + len(path) - 1, len(path),
                           getattr(self, "_now", -1))))
    return _ap(self, agent, path, t0, permanent)


ReservationTable.future_clear = future_clear
ReservationTable.add_path = add_path

_detect = SIM.MAPDSim._detect_conflicts


def _d(self):
    seen = {}
    for a in self.agents:
        p = a["pos"]
        if p in seen:
            self.stats.setdefault("CLOG", []).append(
                (getattr(self, "_cur_t", -1), p, seen[p], a["id"]))
        seen[p] = a["id"]
    return _detect(self)


SIM.MAPDSim._detect_conflicts = _d

_plan = SIM.MAPDSim._plan_task
_rest = SIM.MAPDSim._rest_decide


def rest_decide(self, a, t):
    self.rt._now = t
    return _rest(self, a, t)


def plan_task(self, a, tk, t, pickup=None, delivery=None):
    self.rt._now = t
    return _plan(self, a, tk, t, pickup, delivery)


SIM.MAPDSim._rest_decide = rest_decide
SIM.MAPDSim._plan_task = plan_task


def main():
    wh = Warehouse(width=41, height=21, station_stride=6)
    cfg = SimConfig(n_agents=6, T=310, seed=1, warmup=100, zipf_alpha=2.0,
                    arrival_mode="open", lmbda=0.06, capacity=1, rest_set="any")
    s = SIM.MAPDSim(wh, cfg)
    r = s.run()
    print("conflicts:", r["conflicts"])
    for c in s.stats.get("CLOG", []):
        print("  CONFLICT t=%s pos=%s  agents %d & %d" % c)
    LOG.sort(key=lambda x: x[1])
    print("=== events on %s (now>=250) ===" % (WATCH,))
    for kind, now, desc in LOG:
        if now >= 250 or "298" in desc:
            print("  %-5s now=%-5d %s" % (kind, now, desc))


if __name__ == "__main__":
    main()
