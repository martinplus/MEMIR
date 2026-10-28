"""栅格仓库地图：货架布局、区域划分、BFS 距离场。 零第三方依赖（纯标准库）。

术语遵循 MAPF 社区：location(=cell)、agent、task endpoint、non-task endpoint。
"""
import random
from array import array
from collections import deque

FREE, OBST = 0, 1

# wait, right, left, down, up
DIRS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
MOVES = DIRS[1:]


class Warehouse:
    """栅格仓库（G2P 形态）。

    布局（自上而下）：
        park_rows 行      -> non-task endpoint（停车位），保证 well-formed
        货架区            -> 矩形货架块 + 通道；通道 location 可作 pickup endpoint
        station_rows 行   -> delivery endpoint（工作站）

    区域划分：每个货架块一个区域；工作站区、停车区各一个区域；
    通道 location 归入曼哈顿距离最近的货架块区域。
    """

    def __init__(self, width=41, height=21, block_w=5, block_h=2,
                 gap_x=2, gap_y=3, margin=1, park_rows=2, station_rows=2,
                 station_stride=0, seed=0):
        self.W, self.H = width, height
        self.rng = random.Random(seed)
        self.grid = [[FREE] * width for _ in range(height)]

        self.park_rows = park_rows
        self.station_rows = station_rows
        shelf_top = margin + park_rows
        self.station_top = height - margin - station_rows

        # ---- 货架块 ----
        self.blocks = []
        y = shelf_top
        while y + block_h <= self.station_top:
            x = margin
            while x + block_w <= width - margin:
                for yy in range(y, y + block_h):
                    for xx in range(x, x + block_w):
                        self.grid[yy][xx] = OBST
                self.blocks.append((x, y, x + block_w, y + block_h))
                x += block_w + gap_x
            y += block_h + gap_y

        n_blocks = len(self.blocks)
        self.n_block_regions = n_blocks
        self.R_STATION = n_blocks
        self.R_PARK = n_blocks + 1
        self.n_regions = n_blocks + 2

        # ---- endpoint ----
        self.parking_cells = []
        for yy in range(margin, margin + park_rows):
            for xx in range(margin, width - margin):
                if self.grid[yy][xx] == FREE:
                    self.parking_cells.append((xx, yy))

        self.station_cells = []
        if station_stride and station_stride > 0:
            # 有限数量的工作站（真实仓库形态）：底行按固定间隔取 delivery endpoint
            yy = height - margin - 1
            for xx in range(margin, width - margin, station_stride):
                if self.grid[yy][xx] == FREE:
                    self.station_cells.append((xx, yy))
        else:
            for yy in range(self.station_top, height - margin):
                for xx in range(margin, width - margin):
                    if self.grid[yy][xx] == FREE:
                        self.station_cells.append((xx, yy))

        self._park_set = set(self.parking_cells)
        self.pickup_cells = self._find_pickup_cells()
        if not self.pickup_cells:
            raise RuntimeError("no pickup cells generated; adjust layout params")
        if not self.station_cells:
            raise RuntimeError("no station cells generated; adjust layout params")

        # ---- 区域划分 ----
        self.region_of = [[-1] * width for _ in range(height)]
        centers = [((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0)
                   for (x0, y0, x1, y1) in self.blocks]
        for yy in range(height):
            row = self.grid[yy]
            for xx in range(width):
                if row[xx] != FREE:
                    continue
                if yy >= self.station_top:
                    self.region_of[yy][xx] = self.R_STATION
                elif (xx, yy) in self._park_set:
                    self.region_of[yy][xx] = self.R_PARK
                else:
                    best, bd = -1, 1e18
                    for i, (cx, cy) in enumerate(centers):
                        d = abs(xx - cx) + abs(yy - cy)
                        if d < bd:
                            bd, best = d, i
                    self.region_of[yy][xx] = best
        for i, (x0, y0, x1, y1) in enumerate(self.blocks):
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    self.region_of[yy][xx] = i

        self.free_cells = [(x, y) for y in range(height) for x in range(width)
                           if self.grid[y][x] == FREE]

        self.pickup_by_region = {r: [] for r in range(self.n_regions)}
        for (x, y) in self.pickup_cells:
            self.pickup_by_region[self.region_of[y][x]].append((x, y))

        self._dist_cache = {}
        self._cache_order = deque()
        # 距离场缓存上限：按**内存预算**自适应（每个距离场 2 字节/格）。
        # 预算 ~128 MB/地图对象，小图仍受 1200 的上限约束，保证既有实验行为不变；
        # 大图（22 k 格 MovingAI 图）每个场仅 44 KB，可缓存 ~2 900 个 —— 这才是
        # 大图可行性的关键：缓存装得下整个候选驻留集 + 取货位的工作集。
        self._cache_limit = max(500, min(6000, int(128_000_000 / (2 * width * height))))

    # ---------------- 布局辅助 ----------------
    def _find_pickup_cells(self):
        """货架块四邻的通道 location（不含工作站区与停车区）。"""
        out = set()
        for (x0, y0, x1, y1) in self.blocks:
            for xx in range(x0, x1):
                for yy in (y0 - 1, y1):
                    if 0 <= yy < self.H and self.grid[yy][xx] == FREE:
                        out.add((xx, yy))
            for yy in range(y0, y1):
                for xx in (x0 - 1, x1):
                    if 0 <= xx < self.W and self.grid[yy][xx] == FREE:
                        out.add((xx, yy))
        # 排除停车位（non-task endpoint 不得作为 pickup endpoint）与工作站区
        out = [c for c in out
               if c not in self._park_set and c[1] < self.station_top]
        return sorted(out)

    def is_free(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H and self.grid[y][x] == FREE

    def region(self, x, y):
        return self.region_of[y][x]

    # ---------------- 距离场 ----------------
    def dist_field(self, goal):
        """从 goal 出发的 BFS 距离场（无向图对称，即各点到 goal 的最短步数）。

        返回**扁平** array('h')，索引为 y * W + x；不可达/障碍为 -1。

        为什么扁平：list-of-lists 的每个距离场在 MovingAI 级地图上要吃掉
        ~300 KB（22 k 格），缓存几百个就是上百 MB × worker 数；而 array('h')
        只有 2 字节/格（22 k 格 ≈ 44 KB），同样的内存预算可以缓存 **6~8 倍**
        数量的距离场 —— 大图上真正的瓶颈不是 BFS 本身，而是缓存抖动导致的
        反复重算（实测 B3/|A|=32 在旧实现下 T=400 要跑 10 分钟以上）。
        """
        d = self._dist_cache.get(goal)
        if d is not None:
            return d
        W, H = self.W, self.H
        dist = array("h", bytes(2 * W * H))
        for i in range(W * H):
            dist[i] = -1
        grid = self.grid
        start = goal[1] * W + goal[0]
        dist[start] = 0
        q = deque([(goal[0], goal[1])])
        pop = q.popleft
        push = q.append
        while q:
            x, y = pop()
            nd = dist[y * W + x] + 1
            for dx, dy in MOVES:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == FREE:
                    i = ny * W + nx
                    if dist[i] < 0:
                        dist[i] = nd
                        push((nx, ny))
        self._dist_cache[goal] = dist
        self._cache_order.append(goal)
        if len(self._dist_cache) > self._cache_limit:
            old = self._cache_order.popleft()
            self._dist_cache.pop(old, None)
        return dist

    def dist(self, a, b):
        return self.dist_field(b)[a[1] * self.W + a[0]]

    def reachable(self, a, b):
        return self.dist(a, b) >= 0

    def summary(self):
        return (f"warehouse {self.W}x{self.H} | free={len(self.free_cells)} "
                f"| blocks={self.n_block_regions} regions={self.n_regions} "
                f"| pickup={len(self.pickup_cells)} station={len(self.station_cells)} "
                f"| parking={len(self.parking_cells)}")

    def ascii_map(self):
        rows = []
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                if self.grid[y][x] == OBST:
                    s += "#"
                elif y >= self.station_top:
                    s += "S"
                elif y < self.park_rows:
                    s += "p"
                else:
                    s += "."
            rows.append(s)
        return "\n".join(rows)


def make_default_warehouse(seed=0, scale=1.0):
    w = max(int(41 * scale), 21)
    h = max(int(21 * scale), 13)
    return Warehouse(width=w, height=h, seed=seed)


if __name__ == "__main__":
    wh = make_default_warehouse()
    print(wh.summary())
    print(wh.ascii_map())
    print("pickup per region:",
          {k: len(v) for k, v in sorted(wh.pickup_by_region.items()) if v})
