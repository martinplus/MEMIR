"""MovingAI 标准 MAPF 地图适配层：把社区公开的标准仓库地图接入本文的 MAPD 仿真。

MovingAI MAPF benchmark（Stern et al., SoCS 2019）只提供 one-shot MAPF 所需的
拓扑与 scenario（start/goal 对），**没有** MAPD 所需的 task endpoint（pickup /
delivery）与 non-task endpoint（parking）语义。要在标准拓扑上做 MAPD 评估，
就必须叠加一层端口语义。本模块用一套**对所有地图完全相同、且不含任何与被测
方法相关的可调量**的规则（R1–R6）来完成这件事 —— 规则里唯一的自由度是常量，
如国民 station 目标数与候选集上限，它们对所有地图、所有策略一视同仁。

    R1 grid      '.' / 'G' 可通行，其余字符（@ O T W 等）为障碍；
                 只保留最大 4-连通分量（若干标准图存在孤立小块）。
    R2 pickup    取货位 = 与货架（障碍）4-邻接的自由格，排除最外两圈环形廊道，
                 并按 PICKUP_SPACING 做准规则抽稀（模拟 pod 的间隔摆放）。
    R3 delivery  工作站 = 环形廊道上沿周长顺序按固定间隔取样的格。
                 真实 KIVA 仓的拣选站沿仓库周边布置，故定义在最外两圈；
                 取样间隔自适应收紧，直到工作站数达到 STATION_TARGET 为止。
    R4 parking   泊位 = 环形廊道的剩余格；不足 PARK_MIN 时用「到最近取货位
                 距离最大」的内部通道格补足 —— well-formed 实例要求
                 non-task endpoint 的数量足以容纳整个 fleet。
    R5 region    对自由格做等面积矩形网格分块，只保留含取货位的块
                 （目标 ~region_target 个），另加 station 区与 parking 区，
                 语义与本文 Warehouse 完全一致。
    R6 rest      候选驻留格 = 泊位 ∪ 非取货位/非工作站格的空间均匀抽样，
                 上限 REST_CAP；所有被测策略共享同一候选集，保证对照公平。

地图文件来源：https://movingai.com/benchmarks/mapf/index.html（ODbL），
由 benchmarks/fetch_benchmarks.py 下载到 benchmarks/maps/。
"""
import os
from collections import deque

from warehouse import Warehouse, FREE, OBST

PASSABLE = frozenset(".Gg+")
MOVES4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PICKUP_SPACING = 3       # 取货位之间的最小间距（格）
STATION_TARGET = 24      # 工作站数量目标（远大于任何被测 fleet）
PARK_MIN_FRAC = 0.05     # 泊位至少占自由格的比例
PARK_MIN_ABS = 64
REST_CAP = 360           # 候选驻留格上限
REGION_TARGET = 16       # 货架分区目标数


def read_mapfile(path):
    """解析 MovingAI .map 格式，返回 (grid, W, H)。grid[y][x] ∈ {0,1}。"""
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    hdr = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        low = ln.strip().lower()
        if low == "map" or low.startswith("map "):
            i += 1
            break
        parts = ln.split()
        if len(parts) >= 2:
            hdr[parts[0].lower()] = parts[1]
        i += 1
    rows = lines[i:]
    H = int(hdr.get("height", len(rows)))
    W = int(hdr.get("width", len(rows[0]) if rows else 0))
    rows = [r.ljust(W, "@") for r in rows[:H]]
    while len(rows) < H:
        rows.append("@" * W)
    grid = [[FREE if rows[y][x] in PASSABLE else OBST
             for x in range(W)] for y in range(H)]
    return grid, W, H


def largest_component(grid, W, H):
    """保留最大 4-连通可通行分量，其余置为障碍。返回是否在外部被修改。"""
    seen = [[False] * W for _ in range(H)]
    best = []
    for y0 in range(H):
        for x0 in range(W):
            if grid[y0][x0] != FREE or seen[y0][x0]:
                continue
            comp = [(x0, y0)]
            seen[y0][x0] = True
            head = 0
            while head < len(comp):
                x, y = comp[head]
                head += 1
                for dx, dy in MOVES4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == FREE \
                            and not seen[ny][nx]:
                        seen[ny][nx] = True
                        comp.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    keep = set(best)
    if len(keep) == sum(1 for y in range(H) for x in range(W)
                        if grid[y][x] == FREE):
        return False
    for y in range(H):
        for x in range(W):
            if grid[y][x] == FREE and (x, y) not in keep:
                grid[y][x] = OBST
    return True


def multi_source_dist(grid, W, H, sources):
    """从 sources 出发的 multi-source BFS 距离（不可达为 -1）。"""
    dist = [[-1] * W for _ in range(H)]
    q = deque()
    for (x, y) in sources:
        if 0 <= x < W and 0 <= y < H and grid[y][x] == FREE:
            dist[y][x] = 0
            q.append((x, y))
    while q:
        x, y = q.popleft()
        nd = dist[y][x] + 1
        for dx, dy in MOVES4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == FREE \
                    and dist[ny][nx] < 0:
                dist[ny][nx] = nd
                q.append((nx, ny))
    return dist


class StandardWarehouse(Warehouse):
    """在 MovingAI 标准拓扑上叠加固定 MAPD 端口语义后的地图对象。

    对外接口与 Warehouse 完全一致（sim / planner / assign / rest 均不知情）。
    """

    def __init__(self, mapfile, station_stride=6, region_target=REGION_TARGET,
                 rest_cap=REST_CAP, seed=0):
        self.name = os.path.splitext(os.path.basename(mapfile))[0]
        grid, W, H = read_mapfile(mapfile)
        self.dropped_components = largest_component(grid, W, H)

        self.W, self.H = W, H
        self.grid = grid
        self.rng = __import__("random").Random(seed)
        free_set = {(x, y) for y in range(H) for x in range(W)
                    if grid[y][x] == FREE}
        self.free_cells = sorted(free_set)

        self.park_rows = 0
        self.station_top = H          # 标准图无"工作站在下方"的线性布局
        self.blocks = []

        # ---- R2 / R3 / R4：border ring 与端口 ----
        d_edge = self._edge_rank(W, H)
        ring = [c for c in self.free_cells if d_edge[c] <= 1]
        ring_key = {c: i for i, c in enumerate(ring)}

        station = self._sample_stations(ring, station_stride)
        station_set = set(station)

        pk_set = set()
        for (x, y) in self.free_cells:
            if d_edge[(x, y)] <= 1:
                continue
            for dx, dy in MOVES4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == OBST:
                    pk_set.add((x, y))
                    break
        # 间距化：货架旁的每个通道格理论上都能摆一个料箱，但真实仓库的 pod 是
        # 按间隔摆放的。这里按切比雪夫间距 PICKUP_SPACING 做准规则抽稀，
        # 等价于"每隔若干货位取一个"，使 standard 地图的 endpoint 密度与本文
        # W1/W2/W3 同量级（该间距只影响实例构造，与被测方法无关）。
        pk_set = self._subsample(sorted(pk_set), PICKUP_SPACING)
        pk_set = {c for c in pk_set if c not in station_set}
        self.pickup_cells = sorted(pk_set)

        # 泊位：环形廊道剩余格 + 远离取货位的内部通道格补足
        park = [c for c in ring if c not in station_set and c not in pk_set]
        target = max(PARK_MIN_ABS, int(PARK_MIN_FRAC * len(self.free_cells)))
        if len(park) < target:
            dpk = multi_source_dist(grid, W, H, self.pickup_cells)
            inner = [c for c in self.free_cells
                     if c not in station_set and c not in pk_set
                     and c not in ring_key]
            inner.sort(key=lambda c: -dpk[c[1]][c[0]])
            need = target - len(park)
            park += [c for c in inner[:need]]
        self.parking_cells = sorted(set(park))
        self.station_cells = station

        self._park_set = set(self.parking_cells)
        self._station_set = station_set

        # ---- R5：区域划分（等面积矩形网格 + station/parking 两个区）----
        self._build_regions(region_target)

        # ---- R6：候选驻留格 ----
        self.rest_candidates = self._build_rest_candidates(rest_cap)

        self._dist_cache = {}
        self._cache_order = deque()
        # 标准图上候选格数量（数百）本身就可能超过站点上的距离场预算，
        # 因此这里按下限 600 保底，避免 dist_field 被反复淘汰重算。
        # 与 Warehouse 一致的**内存预算**口径（2 字节/格，128 MB 预算）。
        # 旧口径在大图上会把上限压到 600，远小于工作集（B3 的 rest 候选 1 129
        # + 取货位 2 893），导致 dist_field 被反复淘汰重算 —— B3 慢的根因。
        self._cache_limit = max(500, min(6000, int(128_000_000 / (2 * W * H))))

    # ---------------- 内部构造 ----------------
    @staticmethod
    def _edge_rank(W, H):
        """每格到地图矩形边界的切比雪夫距离（即第几圈）。"""
        return {(x, y): min(x, W - 1 - x, y, H - 1 - y)
                for y in range(H) for x in range(W)}

    def _sample_stations(self, ring, stride):
        """沿环周长顺序按间隔取样；不足 STATION_TARGET 时自动收紧间隔。"""
        order = sorted(ring, key=lambda c: (-min(c[1], self.H - 1 - c[1]),
                                            min(c[0], self.W - 1 - c[0])))
        st = stride
        while st >= 1:
            out = order[::st]
            if len(out) >= STATION_TARGET:
                return sorted(out[:STATION_TARGET * 2])
            st //= 2
        return sorted(order)

    @staticmethod
    def _subsample(cells, spacing):
        """准规则抽稀：任意两个保留点的切比雪夫距离不小于 spacing。"""
        if spacing <= 1:
            return set(cells)
        buckets = {}
        kept = set()
        for (x, y) in cells:
            bx, by = x // spacing, y // spacing
            ok = True
            for i in (bx - 1, bx, bx + 1):
                for j in (by - 1, by, by + 1):
                    for (qx, qy) in buckets.get((i, j), ()):
                        if max(abs(qx - x), abs(qy - y)) < spacing:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                kept.add((x, y))
                buckets.setdefault((bx, by), []).append((x, y))
        return kept

    def _build_regions(self, region_target):
        W, H, grid = self.W, self.H, self.grid
        free = self.free_cells
        if not free:
            raise RuntimeError("standard map has no free cells")
        # 等面积矩形分块
        aspect = max(W, 1) / float(max(H, 1))
        nx = max(1, int(round((region_target * aspect) ** 0.5)))
        ny = max(1, int(round(region_target / float(nx))))
        while nx * ny < region_target:
            ny += 1

        def tile_of(c):
            ix = min(nx - 1, c[0] * nx // W)
            iy = min(ny - 1, c[1] * ny // H)
            return iy * nx + ix

        pk_tiles = sorted({tile_of(c) for c in self.pickup_cells})
        tile2r = {t: i for i, t in enumerate(pk_tiles)}
        k = len(tile2r)
        self.n_block_regions = k
        self.R_STATION = k
        self.R_PARK = k + 1
        self.n_regions = k + 2

        self.region_of = [[-1] * W for _ in range(H)]
        self.pickup_by_region = {r: [] for r in range(self.n_regions)}
        for c in free:
            if c in self._station_set:
                r = self.R_STATION
            elif c in self._park_set:
                r = self.R_PARK
            else:
                r = tile2r.get(tile_of(c), -1)
            self.region_of[c[1]][c[0]] = r
        for c in self.pickup_cells:
            self.pickup_by_region[self.region_of[c[1]][c[0]]].append(c)

    def _build_rest_candidates(self, rest_cap):
        """泊位全量 + 其余可用格的空间均匀抽样，上限 rest_cap。"""
        avail = [c for c in self.free_cells
                 if c not in self._station_set and c not in self._park_set
                 and self.region_of[c[1]][c[0]] not in (-1,)]
        pk = set(self.pickup_cells)
        avail = [c for c in avail if c not in pk]
        out = list(self.parking_cells)
        if len(out) < rest_cap and avail:
            step = max(1, len(avail) // (rest_cap - len(out)))
            out += avail[::step][:rest_cap - len(out)]
        # 去重且保持确定性顺序
        seen = set()
        res = []
        for c in sorted(set(out)):
            if c not in seen:
                seen.add(c)
                res.append(c)
        return res

    # ---------------- 覆盖 ----------------
    def summary(self):
        return (f"{self.name} {self.W}x{self.H} | free={len(self.free_cells)} "
                f"| regions={self.n_regions} "
                f"| pickup={len(self.pickup_cells)} "
                f"station={len(self.station_cells)} "
                f"parking={len(self.parking_cells)} "
                f"rest={len(self.rest_candidates)}")

    def ascii_map(self):
        rows = []
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                if self.grid[y][x] == OBST:
                    s += "#"
                elif (x, y) in self._station_set:
                    s += "S"
                elif (x, y) in self._park_set:
                    s += "p"
                else:
                    s += "."
            rows.append(s)
        return "\n".join(rows)


# 地图名 -> 文件名映射（短名便于在实验矩阵里书写）
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "benchmarks", "maps")
BENCH = {
    "B1": "warehouse-10-20-10-2-1.map",   # 161x63,  5,699 states
    "B2": "warehouse-10-20-10-2-2.map",   # 170x84,  9,776 states
    "B3": "warehouse-20-40-10-2-1.map",   # 321x123, 22,599 states
    "B4": "warehouse-20-40-10-2-2.map",   # 340x164, 30k+ states（建图很慢，慎用）
}

_CACHE = {}


def get_bench(name):
    if name not in BENCH:
        raise KeyError("unknown benchmark map %r (have %s)"
                       % (name, ",".join(sorted(BENCH))))
    if name not in _CACHE:
        _CACHE[name] = StandardWarehouse(os.path.join(BENCH_DIR, BENCH[name]))
    return _CACHE[name]


if __name__ == "__main__":
    for n in sorted(BENCH):
        wh = get_bench(n)
        print("--- %s ---" % n)
        print(wh.summary())
