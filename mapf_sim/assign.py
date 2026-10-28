"""任务分配（task assignment）。零第三方依赖。

提供两种分配器：
  - greedy：Token Passing 风格的逐个最近分配（Ma et al. 2017 的 TP）
  - hungarian：全局最优一对一匹配（CENTRAL 的分配层）
代价可由信息素场调制，调制方式见 modulate 参数。
"""

INF = float("inf")


def hungarian(cost):
    """Kuhn-Munkres（Jonker-Volgenant 形式），O(n^3)，要求 n <= m。

    cost: n x m 的非负代价矩阵（list of lists）
    返回 [(row, col), ...] 的一一匹配（长度 n）。
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m < n:
        raise ValueError("hungarian requires n <= m")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            row = cost[i0 - 1]
            ui = u[i0]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - ui - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [0] * (n + 1)
    for j in range(1, m + 1):
        if p[j] > 0:
            ans[p[j]] = j
    return [(i - 1, ans[i] - 1) for i in range(1, n + 1) if ans[i] > 0]


def greedy_assign(cost):
    """逐个 agent 取当前最优任务（TP 风格）。返回 [(row, col), ...]。"""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    used = [False] * m
    out = []
    for i in range(n):
        best, bj = INF, -1
        row = cost[i]
        for j in range(m):
            if not used[j] and row[j] < best:
                best, bj = row[j], j
        if bj >= 0 and best < INF:
            used[bj] = True
            out.append((i, bj))
    return out


def build_cost(wh, agents, tasks, tau, modulate="none"):
    """构造代价矩阵；每个任务可有若干候选 pickup location（同一 SKU 多库位存放）。

    modulate:
      none    -> c = d(a, pickup)
      mean    -> c = d * (tau[r_start] + tau[r_pickup]) / 2     （M1' 主方案）
      target  -> c = d * tau[r_pickup]                          （M3）
    返回 (cost, choice)，choice[i][j] 为该配对选中的 pickup location。
    """
    n, m = len(agents), len(tasks)
    cost = [[0.0] * m for _ in range(n)]
    choice = [[None] * m for _ in range(n)]
    BIG = 10 ** 6
    for i, a in enumerate(agents):
        apos = a["pos"]
        ar = wh.region(*apos)
        for j, tk in enumerate(tasks):
            pks = tk.get("pickups") or [tk["pickup"]]
            dls = tk.get("deliveries") or [tk["delivery"]]
            best, bp, bd = BIG, pks[0], dls[0]
            for pk in pks:
                d1 = wh.dist(apos, pk)
                if d1 < 0:
                    continue
                pr = wh.region(*pk)
                for dl in dls:
                    d2 = wh.dist(pk, dl)
                    if d2 < 0:
                        continue
                    if modulate == "none" or tau is None:
                        c = float(d1 + d2)
                    elif modulate == "mean":
                        dr = wh.region(*dl)
                        c = (float(d1) * 0.5 * (tau[ar] + tau[pr])
                             + float(d2) * 0.5 * (tau[pr] + tau[dr]))
                    else:  # target
                        c = float(d1) * tau[pr] + float(d2) * tau[pr]
                    if c < best:
                        best, bp, bd = c, pk, dl
            cost[i][j] = best
            choice[i][j] = (bp, bd)
    return cost, choice


def assign(wh, agents, tasks, tau, mode="hungarian", modulate="none"):
    """返回 [(agent_idx, task_idx, chosen_pickup), ...]。"""
    if not agents or not tasks:
        return []
    cost, choice = build_cost(wh, agents, tasks, tau, modulate)
    if mode == "greedy":
        pairs = greedy_assign(cost)
    elif len(agents) <= len(tasks):
        pairs = hungarian(cost)
    else:
        ct = [[cost[i][j] for i in range(len(agents))] for j in range(len(tasks))]
        pairs = [(j, i) for (i, j) in hungarian(ct)]
    return [(i, j, choice[i][j]) for (i, j) in pairs]
