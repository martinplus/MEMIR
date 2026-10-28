# -*- coding: utf-8 -*-
"""标准地图（MovingAI）适配层冒烟测试：构建 + 预计算 + 一次短 run。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = open(os.path.join(HERE, "_smokeb.txt"), "w", encoding="utf-8")


def emit(s):
    LOG.write(s + "\n")
    LOG.flush()


def main():
    from maps import get_map
    from rest import (build_haul_load, build_region_dist_maps,
                      rest_cells_for)
    from sim import MAPDSim, SimConfig

    for nm in ("B1", "B2", "B3"):
        t0 = time.perf_counter()
        try:
            wh = get_map(nm)
            emit("%s  %s   build=%.1fs" % (nm, wh.summary(),
                                           time.perf_counter() - t0))
        except Exception as e:                 # noqa: BLE001
            import traceback
            emit("%s  BUILD-EXC %r" % (nm, e))
            traceback.print_exc(file=LOG)
            continue
        t1 = time.perf_counter()
        try:
            cells = rest_cells_for(wh, "any")
            build_region_dist_maps(wh, cells)
            build_haul_load(wh, cells)
            emit("    precompute %.1fs   rest_cand=%d"
                 % (time.perf_counter() - t1, len(cells)))
        except Exception as e:                 # noqa: BLE001
            import traceback
            emit("    PRECOMPUTE-EXC %r" % (e,))
            traceback.print_exc(file=LOG)
            continue

    BASE = dict(T=800, warmup=300, zipf_alpha=2.0, arrival_mode="open",
                lmbda=0.08, capacity=1, rest_set="any", drift_mode="static",
                drift_param=2000, planner_mode="tp", assign_mode="hungarian",
                dwell_safety=True, dwell_probe=8, preempt=0, tp_dummy=False,
                reeval_interval=40, move_thresh=0.0, rest_deplete=0.0,
                svc_radius=4.0, flow_cost=0.0)
    POL = {
        "static": dict(rest_policy="static", demand_mode="none"),
        "nearest": dict(rest_policy="nearest", demand_mode="none"),
        "memir": dict(rest_policy="exp", demand_mode="ema", tau_phi3=100,
                      rest_beta=0.5, flow_cost=30.0),
    }
    for nm in ("B1",):
        for n in (16, 32):
            for pol in ("nearest", "static", "memir"):
                try:
                    cfg = SimConfig(seed=1, n_agents=n, map_name=nm,
                                    **dict(BASE, **POL[pol]))
                    t0 = time.perf_counter()
                    r = MAPDSim(get_map(nm), cfg).run()
                    emit("%s |A|=%-3d %-8s svc=%8.2f thr=%6.1f "
                         "plan_fail=%4d wall=%.1fs"
                         % (nm, n, pol, r.get("service_time", -1),
                            r.get("throughput", -1), r.get("plan_fail", 0),
                            time.perf_counter() - t0))
                except Exception as e:         # noqa: BLE001
                    import traceback
                    emit("%s |A|=%-3d %-8s EXC %r" % (nm, n, pol, e))
                    traceback.print_exc(file=LOG)


try:
    main()
except Exception:                              # noqa: BLE001
    import traceback
    emit("TOPLEVEL EXC")
    traceback.print_exc(file=LOG)
LOG.close()
