
import os, sys, time
sys.path.insert(0, r"C:\Users\HunterChaw\WorkBuddy\2026-09-24-18-52-47\mapf_sim")
from maps import get_map
for nm in ("B1", "B2", "B3"):
    t0 = time.perf_counter()
    try:
        wh = get_map(nm)
        print("%s  %s   build=%.1fs" % (nm, wh.summary(), time.perf_counter()-t0))
        t1 = time.perf_counter()
        from rest import build_haul_load, build_region_dist_maps, rest_cells_for
        cells = rest_cells_for(wh, "any")
        build_region_dist_maps(wh, cells)
        build_haul_load(wh, cells)
        print("    precompute %.1fs   rest_cand=%d" % (time.perf_counter()-t1, len(cells)))
    except Exception as e:
        import traceback
        print("%s  EXC %r" % (nm, e))
        traceback.print_exc()
