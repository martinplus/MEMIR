"""空闲 agent 预定位（rest / repositioning）策略集。零第三方依赖。

问题模型：
    空闲 agent 位于 pos，未来任务按未知分布 p(r) 在区域 r 出现。
    选择休息位置 c，使**期望响应时间**最小：
        E(c) = Σ_r p̂_r · d̄(c, r)          （期望到下一个 task endpoint 的距离）
    同时付出改位代价 d(pos, c)。目标：
        c* = argmin_c [ E(c) + β · d(pos, c) ]

这是一个 **加权 1-中位数（weighted 1-median）** 问题；信息素场提供的是权重 p̂，
它必须是**对未来任务分布的外生估计**，与 Phase 1/2 的内生观测量相对。

策略清单（对照的关键：static vs oracle）：
    nearest  -> 最近的可用 non-task endpoint          （系统默认行为，主基线）
    random   -> 随机可用 rest cell                     （分辨力阴性对照）
    static   -> 固定 home slot，均匀分层铺开            （**决定性对照**）
    exp      -> 信息素估计的加权中位（本文方法）
    oracle   -> 事后真实分布的加权中位（genie 上界）
    softmax  -> 按 exp(-score/T) 采样（ACO 概率构造 / RL 探索版）

static 的作用：若 exp 的增益只来自"把 agent 分散开"而非"知道需求分布"，
则 static 会吃掉绝大部分收益，从而暴露结论不成立。因此本模块所有对照都围绕
"static 能不能拿到同样的收益"来设计。

注意：所有策略都停泊在允许的 rest cell 集合内（'park' 时即 non-task endpoint，
保证 well-formed），因此对照只差在**选哪一个**，不差在是否停泊。
"""
import math
import random

from learn import LearnedPolicy


def build_region_dist_maps(wh, cand_cells):
    """对每个区域 r，建立 [cell -> 到该区域 pickup endpoint 集合的平均最短路]。

    一次性计算并缓存在 warehouse 对象上（BFS 次数 = pickup endpoint 数量），
    后续所有 run 共享。

        返回 list[r] -> dict[cell] -> 平均距离
    """
    cache = getattr(wh, "_rdist_maps", None)
    if cache is None:
        cache = {}
        wh._rdist_maps = cache
    ck = tuple(sorted(cand_cells))
    if ck in cache:
        return cache[ck]

    maps = [dict() for _ in range(wh.n_regions)]
    cell_list = list(cand_cells)
    for r in range(wh.n_regions):
        cells_r = wh.pickup_by_region.get(r) or []
        if not cells_r:
            continue
        acc = dict.fromkeys(cell_list, 0)
        W = wh.W               # 距离场扁平：索引 y * W + x
        for p in cells_r:
            df = wh.dist_field(p)
            for c in cell_list:
                d = df[c[1] * W + c[0]]
                acc[c] += d if d >= 0 else 0
        n = len(cells_r)
        for c in cell_list:
            maps[r][c] = acc[c] / n
    cache[ck] = maps
    return maps


_NB4 = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _nearest_station(wh, p, stations, df=None):
    """pickup p 的最近工作站（delivery endpoint）。"""
    Wdf = wh.W
    df = wh.dist_field(p) if df is None else df
    best, bd = None, None
    for s in stations:
        d = df[s[1] * Wdf + s[0]]
        if d > 0 and (bd is None or d < bd):
            best, bd = s, d
    return best, bd


def _walk_one_path(wh, load, p, best, rng=None):
    """沿「一条」最短路径 p -> best 累加负荷。

    rng=None  -> 固定邻居序 tie-break（既有全部实验的定义，flow_def="single"）
    rng 给定  -> 等长候选里**随机**取一个（flow_def="rand"，用于检验 tie-break
                敏感性；等代价最短路通常有指数多条，详见 flow_def="multi"）
    """
    Wdf = wh.W
    dsu = wh.dist_field(best)
    cur = p
    hops = 0
    seen = set()
    while cur != best and hops < 4096:
        hops += 1
        seen.add(cur)
        cd = dsu[cur[1] * Wdf + cur[0]]
        cand = []
        for dx, dy in _NB4:
            q = (cur[0] + dx, cur[1] + dy)
            if q in seen:
                continue
            if not (0 <= q[0] < wh.W and 0 <= q[1] < wh.H):
                continue
            if wh.grid[q[1]][q[0]] != 0:
                continue
            dq = dsu[q[1] * Wdf + q[0]]
            if dq >= 0 and dq < cd:
                cand.append(q)
        if not cand:
            break
        nxt = rng.choice(cand) if rng is not None else cand[0]
        cur = nxt
        if cur in load:
            load[cur] += 1.0


def _add_any_path(wh, load, ds, dw, D):
    """flow_def="any"：只要落在**某一条**最短路上就计 1（不看路径条数）。"""
    Wdf = wh.W
    for c in list(load):
        i = c[1] * Wdf + c[0]
        a, b = ds[i], dw[i]
        if a >= 0 and b >= 0 and a + b == D:
            load[c] += 1.0


def _add_multi_path(wh, load, ds, dw, D, p, best):
    """flow_def="multi"：概率化路径流（accounting for route multiplicity）。

    把所有等长最短路径视为**均匀分布**的样本空间（即 agent 等概率地走其中
    任意一条），则单元 c 承载的流量为

        Pr[c ∈ 路径] = #(经过 c 的最短路) / #(全部最短路)
                     = σ_s(c) · σ_w(c) / σ_s(w)

    其中 σ_s(c) 是 p→c 的最短路条数、σ_w(c) 是 c→w 的最短路条数（均在
    「只走位于最短路上的格子」这一 DAG 上计数）。这是标准的路径多重性
    （route multiplicity）流量分解，且**完全不依赖任何 tie-break**：它是
    single / rand 两种取法在 tie-break 上的期望。
    """
    Wdf = wh.W
    ip = p[1] * Wdf + p[0]
    iw = best[1] * Wdf + best[0]
    on = []
    for y in range(wh.H):
        row = wh.grid[y]
        base = y * Wdf
        for x in range(wh.W):
            if row[x] != 0:
                continue
            i = base + x
            a, b = ds[i], dw[i]
            if a >= 0 and b >= 0 and a + b == D:
                on.append((x, y))
    if not on:
        return
    onset = set(on)
    ns = dict.fromkeys(on, 0.0)
    nw = dict.fromkeys(on, 0.0)
    ns[p] = 1.0
    nw[best] = 1.0
    for c in sorted(on, key=lambda q: ds[q[1] * Wdf + q[0]]):
        if c == p:
            continue
        dc = ds[c[1] * Wdf + c[0]]
        tot = 0.0
        for dx, dy in _NB4:
            q = (c[0] + dx, c[1] + dy)
            if q in onset and ds[q[1] * Wdf + q[0]] == dc - 1:
                tot += ns[q]
        ns[c] = tot
    for c in sorted(on, key=lambda q: -ds[q[1] * Wdf + q[0]]):
        if c == best:
            continue
        dc = ds[c[1] * Wdf + c[0]]
        tot = 0.0
        for dx, dy in _NB4:
            q = (c[0] + dx, c[1] + dy)
            if q in onset and ds[q[1] * Wdf + q[0]] == dc + 1:
                tot += nw[q]
        nw[c] = tot
    tot_paths = ns.get(best, 0.0)
    if tot_paths <= 0:
        return
    for c in on:
        if c in load:
            load[c] += ns[c] * nw[c] / tot_paths


def build_haul_load(wh, cand_cells, mode="single", seed=0):
    """每个候选泊位格落在多少条「pickup → 工作站」最短搬运路径上。

    这是**驻留干扰（dwell-induced interference）**的结构性先验：空闲 agent 停在
    主干搬运通道上时，正在送货（haul leg）的 agent 必须绕行或等待 —— 诊断
    diag_legs 显示 L3(pickup→delivery) 段随 fleet 增大从 29.4 涨到 33.7，
    完全抵消了 L2(response) 段省下的距离。

    mode（对应 cfg.flow_def）：
        single  -> 每条 pickup→工作站 取**一条**最短路，固定邻居序 tie-break
        rand    -> 同上但 tie-break 随机（检验「选哪条」的敏感性）
        any     -> 落在**任一**最短路上即计 1（集合版，不看路径条数）
        multi   -> 概率化路径流，按路径多重性加权（tie-break 无关的期望版）

    返回一个把负荷归一化到 [0,1] 的 dict；按 (mode, seed) 缓存在 warehouse 上。
    """
    key = (mode, int(seed), len(cand_cells))
    cache = getattr(wh, "_haul_load_cache", None)
    if cache is None:
        cache = {}
        wh._haul_load_cache = cache
    if key in cache:
        return cache[key]
    load = dict.fromkeys(cand_cells, 0.0)
    stations = list(wh.station_cells)
    rng = random.Random(int(seed) * 7919 + 13) if mode == "rand" else None
    for p in wh.pickup_cells:
        df = wh.dist_field(p)
        best, bd = _nearest_station(wh, p, stations, df)
        if best is None:
            continue
        if mode == "any":
            _add_any_path(wh, load, df, wh.dist_field(best), bd)
        elif mode == "multi":
            _add_multi_path(wh, load, df, wh.dist_field(best), bd, p, best)
        else:
            _walk_one_path(wh, load, p, best, rng)
    mx = max(load.values()) or 1.0
    out = {c: v / mx for c, v in load.items()}
    cache[key] = out
    return out


class RestPolicy:
    def __init__(self, wh, cand_cells, policy="nearest", beta=0.0,
                 temperature=0.0, seed=0, n_agents=1, deplete=0.0,
                 svc_radius=4.0, flow_cost=0.0,
                 learn_kind="lin", learn_lr=0.05, learn_temp=0.5,
                 learn_eps=0.10, learn_anchor=16, learn_topk=20,
                 learn_mode="rl", learn_greedy=False, pop_flow=0.0,
                 flow_def="single", flow_seed=None, learn_hidden=8):
        self.wh = wh
        self.cells = list(cand_cells)
        self.n_agents = max(1, int(n_agents))
        self.policy = policy
        self.beta = beta
        self.T = temperature
        # ---- 负信息素 / 覆盖耗减（curse of the 1-median 的解药）----
        # deplete η ∈ [0,1]：已承诺驻守的 agent 从共享场中扣减自己的覆盖份额
        # svc_radius R：服务半径，超出该距离几乎不能响应那里的任务
        self.deplete = max(0.0, min(1.0, float(deplete)))
        self.svc_radius = max(0.5, float(svc_radius))
        # 驻留干扰避让强度：>0 时惩罚落在主干搬运通道上的泊位
        self.flow_cost = float(flow_cost)
        # 学习类基线即使 ψ=0 也需要 flow 作为输入特征（信息对齐，见 learn.py）
        self.pop_flow = float(pop_flow)
        self.flow = build_haul_load(wh, self.cells, mode=flow_def,
                                    seed=seed if flow_seed is None else flow_seed) \
            if (self.flow_cost > 0 or policy in ("learn", "popsample")) \
            else None
        self.rng = random.Random(seed * 104729 + 7)
        self.maps = build_region_dist_maps(wh, self.cells)
        self._cellset = set(self.cells)
        self._rng_ = self.wh.n_regions
        # ---- 学习类基线（RL / AM 直接输出驻留位置）----
        self.lp = None
        if policy == "learn":
            self.lp = LearnedPolicy(wh, self.cells, kind=learn_kind,
                                    lr=learn_lr, temp=learn_temp,
                                    eps=learn_eps, anchors=learn_anchor,
                                    topk=learn_topk, seed=seed,
                                    mode=learn_mode, greedy=learn_greedy,
                                    hidden=learn_hidden)

    def learn_observe(self, agent_id, d0):
        """把一次派单时兑现的响应距离回传给学习类策略（若有）。"""
        if self.lp is not None:
            self.lp.feedback(agent_id, d0)

    # ---------- 覆盖耗减（负信息素）----------
    def _kernel(self, d):
        """一个驻守在距离 d 处的 agent 对某区域需求的响应能力 κ(d)∈(0,1]。"""
        return math.exp(-d / self.svc_radius)

    def residual(self, p, occupied):
        """把已被他人占位的影响计入 → 返回**剩余未被覆盖**的需求权重。

            p̃_r = p̂_r · Π_{q ∈ occupied} (1 − η · κ(d̄(q, r)))

        这正是覆盖类次模函数的贪心边际增益：第 k 个 agent 不再最大化同一个
        1-中位数（那会把所有 agent 吸到同一热点，即 curse of the 1-median），
        而是最大化自己在**已有人驻守之后还能贡献**的那部分需求。
        η = 0 时退化为纯加权中位（原文 baseline），η = 1 且 κ→1 时退化为
        「完全互斥覆盖」，行为趋近均匀分散。
        """
        eta = self.deplete
        if eta <= 0 or not occupied:
            return list(p)
        f = [1.0] * self._rng_
        for i in range(self._rng_):
            fi = 1.0
            m = self.maps[i]
            for q in occupied:
                d = m.get(q)
                if d is None:
                    continue
                fi *= (1.0 - eta * self._kernel(d))
            f[i] = fi
        return [p[i] * f[i] for i in range(self._rng_)]

    # ---------- 覆盖型目标（greedy p-median / 设施选址）----------
    def cover_cost(self, S, p):
        """F(S) = Σ_r p_r · min_{q∈S} d̄(q, r)：驻守集合 S 的期望响应代价。

        与 expected_cost（**单个** agent 的 1-中位数）的区别：这里衡量的是
        **集合**的覆盖能力 —— 这正是运筹学里 p-median / 设施选址的目标函数。
        作为对照，它回答一个关键问题：MEMIR 的增益是否只是"把 agent 分散开"
        就能拿到？p-median 用的是**完全相同的需求信息 p̂**，因此是公平对照。
        """
        tot = 0.0
        for r in range(len(p)):
            w = p[r]
            if w <= 0:
                continue
            m = self.maps[r]
            best = None
            for q in S:
                d = m.get(q)
                if d is None:
                    continue
                if best is None or d < best:
                    best = d
            tot += w * (best if best is not None else 0.0)
        return tot

    def _cover_pick(self, pos, p, occupied, avail, cap=24):
        """贪心 p-median：在他人已驻守集合的基础上，选边际覆盖增益最大的格。

        F(S ∪ {c}) = Σ_r p_r · min(best_r, d̄(c, r))，其中 best_r = min_{q∈S} d̄(q,r)
        可在决策开头算一次，故单次决策代价 O(|S|·R + |C|·R) 而非 O(|C|·|S|·R)。
        """
        S = [q for q in occupied if q in self._cellset]
        R = len(p)
        best_r = [None] * R
        for r in range(R):
            if p[r] <= 0:
                continue
            m = self.maps[r]
            b = None
            for q in S:
                d = m.get(q)
                if d is None:
                    continue
                if b is None or d < b:
                    b = d
            best_r[r] = 0.0 if b is None else b
        cand = avail
        if len(cand) > cap:
            cand = sorted(avail,
                          key=lambda c: self.expected_cost(c, p))[:cap]
        best, bv = None, None
        for c in cand:
            v = 0.0
            for r in range(R):
                w = p[r]
                if w <= 0:
                    continue
                br = best_r[r]
                if br is None:
                    continue
                d = self.maps[r].get(c, 0.0)
                v += w * (d if d < br else br)
            if self.beta > 0 and c != pos:
                v += self.beta * self.wh.dist(pos, c)
            if self.flow is not None:
                v += self.flow_cost * self.flow.get(c, 0.0)
            if bv is None or v < bv:
                best, bv = c, v
        return best

    def _pop_sample(self, pos, p, avail, radius=3.0, cap=40):
        """按「局部需求质量」比例采样驻留格（AMoD 社区的需求吸引式基线）。"""
        cand = avail
        if len(cand) > cap:
            cand = sorted(avail,
                          key=lambda c: self.expected_cost(c, p))[:cap]
        ws = []
        for c in cand:
            s = 0.0
            for r in range(len(p)):
                w = p[r]
                if w > 0:
                    s += w * math.exp(-self.maps[r].get(c, 0.0) / radius)
            if self.pop_flow > 0 and self.flow is not None:
                s *= math.exp(-self.pop_flow * self.flow.get(c, 0.0))
            ws.append(max(s, 1e-9))
        tot = sum(ws)
        x = self.rng.random() * tot
        acc = 0.0
        for c, w in zip(cand, ws):
            acc += w
            if x <= acc:
                return c
        return cand[-1]

    # ---------- 期望响应代价 ----------
    def expected_cost(self, cell, p):
        """Σ_r p_r · d̄(cell, r)。"""
        s = 0.0
        for r in range(self.wh.n_regions):
            w = p[r]
            if w > 0:
                s += w * self.maps[r].get(cell, 0.0)
        return s

    def score(self, cell, pos, p):
        c = self.expected_cost(cell, p)
        if self.beta > 0 and cell != pos:
            c += self.beta * self.wh.dist(pos, cell)
        if self.flow is not None:
            c += self.flow_cost * self.flow.get(cell, 0.0)
        return c

    # ---------- 候选排序 ----------
    def rank(self, pos, p, occupied, agent_id=0, exclude_self=None, limit=0,
             others=None):
        """按策略偏好返回**有序候选序列**（可被 guard 约束逐个性别通过）。

        每种策略原有的选择行为 = rank 的第一个元素，因此引入 guard 不会改变
        关闭 guard 时的任何行为，保证前后对照严格可比。
        """
        avail = [c for c in self.cells
                 if c not in occupied or c == exclude_self]
        if not avail:
            return []
        pol = self.policy
        # 耗减只在「要预测」的策略上生效，nearest/random/static 不受影响
        p = self.residual(p, occupied) if self.deplete > 0 and \
            pol in ("exp", "softmax", "oracle") else p
        if pol == "static":
            k = len(self.cells)
            home = self.cells[(agent_id * k) // self.n_agents % k]
            out = [home] if home in avail else []
            out += [c for c in avail if c != home]
        elif pol == "nearest":
            out = sorted(avail, key=lambda c: self.wh.dist(pos, c))
        elif pol == "random":
            out = list(avail)
            self.rng.shuffle(out)
        elif pol == "stay":
            # 完全不预定位：能不动就不动（检验"预定位本身"是否值得做）
            out = [pos] if pos in avail else []
            out += [c for c in sorted(avail,
                                      key=lambda c: self.wh.dist(pos, c))
                    if c != pos]
            return out[:limit] if limit else out
        elif pol == "pmedian":
            first = self._cover_pick(pos, p, occupied, avail)
            out = [first] if first is not None else []
            out += [c for c in sorted(avail,
                                      key=lambda c: self.expected_cost(c, p))
                    if c != first]
            return out[:limit] if limit else out
        elif pol == "popsample":
            first = self._pop_sample(pos, p, avail)
            out = [first] if first is not None else []
            out += [c for c in sorted(avail,
                                      key=lambda c: self.expected_cost(c, p))
                    if c != first]
            return out[:limit] if limit else out
        elif pol == "learn":
            # 学习类策略的首选项即其直接输出的驻留位置；guard 否决时按期望代价
            # 依次退让，与 exp 策略的退让顺序一致，保证对照只差在「首选项」。
            tf = self._teacher_fn(pos, p) if self.lp.mode == "bc" else None
            first = self.lp.choose(pos, p, occupied,
                                   others if others is not None else occupied,
                                   self.n_agents, self.maps, self.flow, avail,
                                   agent_id=agent_id, exclude_self=exclude_self,
                                   teacher_fn=tf)
            out = [first] if first is not None else []
            out += [c for c in sorted(avail,
                                      key=lambda c: self.expected_cost(c, p))
                    if c != first]
            return out[:limit] if limit else out
        elif pol == "softmax" and self.T > 0:
            # Plackett–Luce 采样：等价于按 exp(-score/T) 不放回逐个抽取，
            # 因此「偏好序列的首元素」就是 softmax 采样结果，与 choose() 一致。
            noise = {}
            for c in avail:
                u = self.rng.random()
                u = 1e-12 if u < 1e-12 else (1 - 1e-12 if u > 1 - 1e-12 else u)
                noise[c] = -self.T * math.log(-math.log(u))
            out = sorted(avail, key=lambda c: self.score(c, pos, p) + noise[c])
        else:
            cand = avail
            if len(cand) > 30:
                cand = sorted(avail, key=lambda c: self.expected_cost(c, p))[:30]
            out = [c for c in sorted(cand, key=lambda c: self.score(c, pos, p))]
        return out[:limit] if limit else out

    # ---------- 决策 ----------
    def choose(self, pos, p, occupied, agent_id=0, exclude_self=None,
               guard=None, others=None):
        """返回目标 cell；None 表示原地不动。

        occupied: 已被其它 agent 永久占用的 cell 集合
        guard:    可选的硬约束 cell -> bool（如 Dwell-Safety 不变量）。
                  逐个沿偏好顺序取第一个通过 guard 的候选。
        """
        pol = self.policy
        avail = [c for c in self.cells
                 if c not in occupied or c == exclude_self]
        if not avail:
            return None
        if pol == "static":
            k = len(self.cells)
            return self.cells[(agent_id * k) // self.n_agents % k]
        if pol == "nearest":
            return min(avail, key=lambda c: self.wh.dist(pos, c))
        if pol == "random":
            return self.rng.choice(avail)
        if pol == "stay":
            # 完全不预定位：能不动就不动（检验"预定位本身"是否值得做）
            return pos if pos in avail else \
                min(avail, key=lambda c: self.wh.dist(pos, c))
        if pol == "pmedian":
            return self._cover_pick(pos, p, occupied, avail)
        if pol == "popsample":
            return self._pop_sample(pos, p, avail)
        if pol == "learn":
            tf = self._teacher_fn(pos, p) if self.lp.mode == "bc" else None
            return self.lp.choose(pos, p, occupied,
                                  others if others is not None else occupied,
                                  self.n_agents, self.maps, self.flow, avail,
                                  agent_id=agent_id,
                                  exclude_self=exclude_self, teacher_fn=tf)

    def _teacher_fn(self, pos, p):
        """bc 模式的教师：解析式加权 1-中位数（即 MEMIR 自己的决策）。

        这是"蒸馏"式学习基线——把 MEMIR 的解析解当作示范，看一个从零开始
        训练的学习策略能否复现它。若能，说明解析解不是偶然的启发式；
        若不能，说明学习过程本身在这种延迟/稀疏回报下存在额外损失。
        注意 hindsight oracle 在在线设定下不可得，故不作为教师。
        """
        def _f(cand):
            j, bv = 0, None
            for i, c in enumerate(cand):
                v = self.score(c, pos, p)
                if bv is None or v < bv:
                    j, bv = i, v
            return j
        return _f

        if self.deplete > 0 and pol in ("exp", "softmax", "oracle"):
            p = self.residual(p, occupied)

        # 加权中位类：先按期望代价粗筛，控制规划开销
        cand = avail
        if len(cand) > 30:
            cand = sorted(avail, key=lambda c: self.expected_cost(c, p))[:30]
        scored = [(self.score(c, pos, p), c) for c in cand]
        if pol == "softmax" and self.T > 0:
            m = min(s for s, _ in scored)
            ws = [(math.exp(-(s - m) / self.T), c) for s, c in scored]
            tot = sum(w for w, _ in ws)
            if tot <= 0:
                return scored[0][1]
            x = self.rng.random() * tot
            acc = 0.0
            for w, c in ws:
                acc += w
                if x <= acc:
                    return c
            return ws[-1][1]
        return min(scored, key=lambda sc: sc[0])[1]


def rest_cells_for(wh, mode="park"):
    """候选休息位置集合。

    park -> 仅 non-task endpoint（严格 well-formed）
    any  -> 通道任意空闲格（排除 pickup endpoint 与工作站行）
            初始配置仍在 non-task endpoint，故不破坏 well-formed；
            运行期停在通道由在线规划器负责让行。
    """
    if mode == "any":
        # 标准地图（MovingAI）上「通道任意空闲格」的规模可达数千，直接全量枚举
        # 会让每次决策的开销与方法本身耦合。故若地图对象显式给出候选集
        # （见 movingai.py 规则 R6），所有策略共享同一子集，保证对照公平。
        base = getattr(wh, "rest_candidates", None)
        if base is not None:
            return list(base)
        pk = set(wh.pickup_cells)
        return [c for c in wh.free_cells
                if c[1] < wh.station_top and c not in pk]
    return list(wh.parking_cells)
