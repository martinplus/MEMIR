"""论文用测试地图集。

三张地图构成结构梯度：
    W1 (41x21)  —— 主地图：POGEMA Warehouse 同族（KIVA 形态），15 货架块 / 17 区域
    W2 (46x33)  —— 大尺度 KIVA-like 仓库（POGEMA Warehouse 33x46 的转置形态）
    W3 (21x13)  —— 小尺度密集仓库，通道资源稀缺，用于检验拥堵 / liveness 压力

三张图都由同一个 Warehouse 生成器产生，因此 endorsements 与工作站的语义完全一致，
差异只在拓扑尺寸与通道宽度 —— 便于把"方法增益"与"地图特例"分离。
"""
from movingai import get_bench          # noqa: E402
from warehouse import Warehouse         # noqa: E402

SPECS = {
    "W1": dict(width=41, height=21, block_w=5, block_h=2, gap_x=2, gap_y=3,
               margin=1, park_rows=2, station_rows=2, station_stride=6),
    "W2": dict(width=46, height=33, block_w=3, block_h=2, gap_x=2, gap_y=2,
               margin=2, park_rows=2, station_rows=2, station_stride=6),
    "W3": dict(width=21, height=13, block_w=5, block_h=2, gap_x=2, gap_y=3,
               margin=1, park_rows=2, station_rows=2, station_stride=4),
    # ---- 尺度阶梯（2026-09-26 新增）：与 W1 同一族拓扑，只放大尺寸 ----
    # 用途：把"地图尺寸"作为一个可分离因子扫过去，检验增益是 idle rate 的函数
    # 而不是地图尺寸的函数。station_stride 随宽度等比放大，使工作站密度不变。
    "W4": dict(width=81, height=41, block_w=5, block_h=2, gap_x=2, gap_y=3,
               margin=1, park_rows=2, station_rows=2, station_stride=12),
    "W5": dict(width=121, height=61, block_w=5, block_h=2, gap_x=2, gap_y=3,
               margin=1, park_rows=2, station_rows=2, station_stride=18),
    "W6": dict(width=161, height=81, block_w=5, block_h=2, gap_x=2, gap_y=3,
               margin=1, park_rows=2, station_rows=2, station_stride=24),
}

_CACHE = {}


def get_map(name="W1", seed=0):
    if str(name).upper() in ("B1", "B2", "B3", "B4"):   # MovingAI 标准地图
        return get_bench(str(name).upper())
    k = (name, seed)
    if k not in _CACHE:
        _CACHE[k] = Warehouse(seed=seed, **SPECS[name])
    return _CACHE[k]


def map_names():
    return list(SPECS) + ["B1", "B2", "B3", "B4"]


if __name__ == "__main__":
    for n in map_names():
        wh = get_map(n)
        print(f"--- {n} ---")
        print(wh.summary())
        print(wh.ascii_map())
