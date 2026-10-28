"""时空 A*（space-time A*）与预约表（reservation table）。零第三方依赖。

冲突模型（标准 MAPF 设定）：
  - vertex conflict：同一 timestep 两个 agent 占据同一 location，禁止
  - edge conflict (swap)：同一 timestep 两个 agent 交换位置，禁止
  - following conflict：允许
终点处理：agent 到达路径末端后永久停留（permanent reservation），
这与 Token Passing (Ma et al. 2017) 一致。
"""
import heapq
from warehouse import MOVES, FREE


class ReservationTable:
    """时空预约表。支持按 agent 增量增删。"""

    def __init__(self):
        self.cell = {}        # (x, y, t) -> agent_id
        self.edge = {}        # (x1, y1, x2, y2, t) -> agent_id  (t -> t+1)
        self.permanent = {}   # (x, y) -> (agent_id, t_start)
        self.owned = {}       # agent_id -> list[(kind, key)]
        self.by_cell = {}     # (x, y) -> set(t)，用于「未来清空」检查
        self.max_t = 0        # 已预约到的最远时刻

    # ---------- 查询 ----------
    def future_clear(self, x, y, t_from, agent):
        """该 location 在 t_from 之后是否没有任何**他人**的预约。

        这是 rest（永久停留）成立的必要条件：若某 agent 早已规划好一条路径，
        会在更晚时刻经过此处，则此刻声称永久停留会在未来撞车。permanent 字典
        只能拦住「之后再规划」的 agent，拦不住「已经规划完」的——本方法补上这个洞。
        """
        ts = self.by_cell.get((x, y))
        if not ts:
            return True
        for tt in ts:
            if tt >= t_from:
                o = self.cell.get((x, y, tt))
                if o is not None and o != agent:
                    return False
        return True
    def vertex_free(self, x, y, t, agent):
        o = self.cell.get((x, y, t))
        if o is not None and o != agent:
            return False
        p = self.permanent.get((x, y))
        if p is not None and p[0] != agent and t >= p[1]:
            return False
        return True

    def edge_free(self, x1, y1, x2, y2, t, agent):
        o = self.edge.get((x2, y2, x1, y1, t))
        if o is not None and o != agent:
            return False
        return True

    # ---------- 增删 ----------
    def add_path(self, agent, path, t0, permanent=True):
        """注册路径预约。

        permanent=True  -> 路径末端永久停留（agent 停在 non-task endpoint）
        permanent=False -> 末端不永久占用（agent 到达 delivery 后立即离开，
                           仅预约到 t_end 为止；这是 MAPD 的关键，
                           否则 delivery endpoint 会被前一个 agent 锁死）
        """
        keys = []
        for i, (x, y) in enumerate(path):
            tt = t0 + i
            k = (x, y, tt)
            self.cell[k] = agent
            self.by_cell.setdefault((x, y), set()).add(tt)
            keys.append(("c", k))
        if t0 + len(path) > self.max_t:
            self.max_t = t0 + len(path)
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            k = (x1, y1, x2, y2, t0 + i)
            self.edge[k] = agent
            keys.append(("e", k))
        if permanent:
            lx, ly = path[-1]
            t_end = t0 + len(path) - 1
            k = (lx, ly)
            self.permanent[k] = (agent, t_end)
            keys.append(("p", k))
        self.owned.setdefault(agent, []).extend(keys)

    def try_hold(self, agent, pos, t, dur=3):
        """原地停留预约（不设 permanent）。**绝不覆盖他人已预约的时刻**——
        覆盖会让对方误以为安全，从而产生真实冲突。

        返回 True 表示成功占位；False 表示该位置在未来已被他人预约，
        调用方必须让该 agent 移动（前往 non-task endpoint）。
        """
        keys = []
        for dt in range(dur):
            k = (pos[0], pos[1], t + dt)
            occ = self.cell.get(k)
            if occ is not None and occ != agent:
                # 回滚已预约的部分
                for kk in keys:
                    if self.cell.get(kk) == agent:
                        del self.cell[kk]
                return False
            perm = self.permanent.get((pos[0], pos[1]))
            if perm is not None and perm[0] != agent and t + dt >= perm[1]:
                for kk in keys:
                    if self.cell.get(kk) == agent:
                        del self.cell[kk]
                return False
        for dt in range(dur):
            tt = t + dt
            k = (pos[0], pos[1], tt)
            self.cell[k] = agent
            self.by_cell.setdefault((pos[0], pos[1]), set()).add(tt)
            keys.append(("c", k))
        if t + dur > self.max_t:
            self.max_t = t + dur
        self.owned.setdefault(agent, []).extend(keys)
        return True

    def remove_agent(self, agent):
        keys = self.owned.pop(agent, [])
        for kind, k in keys:
            if kind == "c":
                if self.cell.get(k) == agent:
                    del self.cell[k]
                s = self.by_cell.get(k[:2])
                if s is not None:
                    s.discard(k[2])
                    if not s:
                        del self.by_cell[k[:2]]
            elif kind == "e":
                if self.edge.get(k) == agent:
                    del self.edge[k]
            else:
                if self.permanent.get(k, (None,))[0] == agent:
                    del self.permanent[k]
        return len(keys)


class PlanFailure(Exception):
    pass


def st_astar(wh, rt, start, goal, t0, agent, max_expand=40000,
             max_time=400, hold_check=True, blockers=None):
    """从 (start, t0) 规划到 goal，返回 location 序列（含起点）。

    启发式为精确 BFS 最短路距离（忽略其他 agent），故为 admissible。

    blockers: 若传入 dict，则在搜索过程中把**造成阻塞的 permanent 预约持有者**
    记入 {agent_id: 计数}。这是 liveness 机制（preempt / 让行）所需要的诊断信息：
    时空 A* 失败时，我们需要知道「谁挡住了路」才能要求它让开。
    """
    if not wh.is_free(*goal):
        raise PlanFailure("goal blocked")
    df = wh.dist_field(goal)
    Wdf = wh.W                # 距离场为扁平 array('h')，索引 y * W + x
    sx, sy = start
    if df[sy * Wdf + sx] < 0:
        raise PlanFailure("goal unreachable")

    def note_block(x, y, tt):
        if blockers is None:
            return
        p = rt.permanent.get((x, y))
        if p is not None and p[0] != agent and tt >= p[1]:
            blockers[p[0]] = blockers.get(p[0], 0) + 1

    W, H = wh.W, wh.H
    grid = wh.grid
    gx, gy = goal

    # 终点若已被他人永久占用，则不可作为终点
    perm = rt.permanent.get(goal)
    if perm is not None and perm[0] != agent:
        note_block(gx, gy, max(0, perm[1]))
        raise PlanFailure("goal permanently occupied")

    h0 = df[sy * Wdf + sx]
    open_h = [(h0, 0, sx, sy, 0)]
    came = {}          # (x,y,t) -> (prev_key, move)
    gbest = {(sx, sy, 0): 0}
    expanded = 0

    while open_h:
        f, g, x, y, t = heapq.heappop(open_h)
        key = (x, y, t)
        if gbest.get(key, -1) != g:
            continue
        expanded += 1
        if expanded > max_expand:
            raise PlanFailure("expand limit")
        if (x, y) == goal:
            # 终点需能停留：下一 timestep 该 location 仍空闲
            if (not hold_check) or rt.vertex_free(gx, gy, t0 + t + 1, agent):
                return _reconstruct(came, key)
            note_block(gx, gy, t0 + t + 1)
            # 无法停留：不扩展该状态，继续搜索更晚到达的时刻
            continue
        if t >= max_time:
            continue

        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny][nx] != FREE:
                continue
            h = df[ny * Wdf + nx]
            if h < 0:
                continue
            nt = t + 1
            if not rt.vertex_free(nx, ny, t0 + nt, agent):
                note_block(nx, ny, t0 + nt)
                continue
            if (dx, dy) != (0, 0):
                if not rt.edge_free(x, y, nx, ny, t0 + t, agent):
                    continue
            # 穿过他人永久停留点：若到达时刻晚于其停留起点则不可
            nkey = (nx, ny, nt)
            if gbest.get(nkey, 10 ** 9) > nt:
                gbest[nkey] = nt
                came[nkey] = key
                heapq.heappush(open_h, (nt + h, nt, nx, ny, nt))
        nt = t + 1
        nkey = (x, y, nt)
        if rt.vertex_free(x, y, t0 + nt, agent):
            if gbest.get(nkey, 10 ** 9) > nt:
                gbest[nkey] = nt
                came[nkey] = key
                heapq.heappush(open_h, (nt + df[y * Wdf + x], nt, x, y, nt))
        else:
            note_block(x, y, t0 + nt)

    raise PlanFailure("no path")


def _reconstruct(came, key):
    path = [key]
    while key in came:
        key = came[key]
        path.append(key)
    path.reverse()
    return [(x, y) for (x, y, t) in path]


def plan_task(wh, rt, start, pickup, delivery, t0, agent, **kw):
    """两段规划：start -> pickup -> delivery。返回完整路径与到达 pickup 的时刻。"""
    p1 = st_astar(wh, rt, start, pickup, t0, agent, **kw)
    t1 = t0 + len(p1) - 1
    p2 = st_astar(wh, rt, pickup, delivery, t1, agent, **kw)
    return p1 + p2[1:], t1


def _dwell_adj(wh):
    """残余图邻接表（一次性构建并缓存在 warehouse 上）。

    liveness 守护每个 decision 要跑若干次残余图遍历，邻接表预建可省去重复的
    边界判断与 tuple 构造。
    """
    adj = getattr(wh, "_dwell_adj_cache", None)
    if adj is None:
        W, H, grid = wh.W, wh.H, wh.grid
        adj = {}
        for (x, y) in wh.free_cells:
            nb = []
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] == FREE:
                    nb.append((nx, ny))
            adj[(x, y)] = nb
        wh._dwell_adj_cache = adj
    return adj


def dwell_components(wh, blocked, max_comp=0):
    """残余图连通分量数。max_comp>0 时一旦超过即提前返回（剪枝）。"""
    adj = _dwell_adj(wh)
    blocked = set(blocked)
    seen = set()
    comp = 0
    for c in wh.free_cells:
        if c in blocked or c in seen:
            continue
        comp += 1
        if max_comp and comp > max_comp:
            return comp
        seen.add(c)
        q = [c]
        while q:
            q2 = []
            for v in q:
                for u in adj[v]:
                    if u in seen or u in blocked:
                        continue
                    seen.add(u)
                    q2.append(u)
            q = q2
    return comp


def dwell_safe(wh, blocked):
    """Dwell-Safety（强式/whole-graph）：移除全部永久停驻格后，残余图整体连通。

    一次 BFS，O(|V|+|E|)。这是**充分但过强**的条件：它连"没有任何 endpoint
    的死角"也要求连通，故一旦历史累积已经违规，所有候选都会被一律否决 ->
    空闲 agent 集体冻结（这正是第一版守护在 |A|>=24 上反而更差的根因）。
    在线判定因此改用下面的 TER-Safety（弱式，可度量、不冻结）。
    """
    return dwell_components(wh, blocked, max_comp=1) == 1


def dwell_violation(wh, blocked, required):
    """TER-Safety（Task-Endpoint Reachability，弱式）**。

    required = 全部 task endpoint ∪ 全部 agent 当前位置。
    移除永久停驻格后，只要求 **required 集合仍在同一连通分量内**；
    不含任何 endpoint 的空口袋被切断不影响任何规划，故允许。

    返回**不可达的 required 节点数**（0 = 安全）。这是一个软指标，调用方据此
    在候选里取最小值 —— 因此永远不会出现"全部候选被拒 -> agent 冻结"。
    """
    adj = _dwell_adj(wh)
    blocked = set(blocked)
    need = set(required) - blocked
    if not need:
        return 0
    start = next(iter(need))
    need.discard(start)
    missing = len(need)
    seen = {start}
    q = [start]
    while q:
        q2 = []
        for v in q:
            for u in adj[v]:
                if u in seen or u in blocked:
                    continue
                seen.add(u)
                if u in need:
                    missing -= 1
                    if missing == 0:
                        return 0
                q2.append(u)
        q = q2
    return missing
