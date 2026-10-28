"""序列级转移信息素 τ[r][r'] 与多任务序列构造（MT-MAPD）。零第三方依赖。

=============================================================================
与 Phase 1 的根本区别（本阶段立论核心）
=============================================================================
Phase 1 的载体是**区域级拥堵系数** τ_r，语义「越大越差」，作用是让 agent
**规避**拥堵。已证无效，机制原因是拥堵总量近似守恒（Wardrop 特性）：
个体规避的收益是二阶的。

本阶段的载体是**区域间转移偏好** τ[r][r']，语义「越大越好」，作用是优化
agent 一次携带的**任务序列**，直接压缩**边际行程**（marginal travel）——
即「做完第一个任务之后，追加第 2..c 个任务所需的额外行程」。

边际行程不守恒：把同一区域的任务聚成一批，边际行程可接近 0；把分散区域
的任务凑一批，边际行程巨大。因此这是真实可优化的量。

=============================================================================
观测量的定义（关键设计，v2）
=============================================================================
批完成后观测：

    margin = L_actual − [d(start, p_1) + d(p_1, d_1)]      # 追加任务的额外行程
    per_task = margin / max(c−1, 1)
    o = C / (C + per_task)                                  # 效率 ∈ (0, 1]

**v1 的教训**：直接把 o 沉积到所有边，会使每条边都收敛到全局平均值，
τ 的空间离散度（spread）仅 0.031 —— 基本无分化，调制退化为噪声，
实测序列几何长度反而比 NN 更差（81.98 vs 80.35）。

**v2 修正：沉积优势（advantage），即相对当前滑动基线的超出量**

    o_bar ← o_bar + β·(o − o_bar)        # 全局效率基线的 EMA
    adv   = o − o_bar                    # 这批比「当前平均水平」好多少
    τ[r][r'] ← τ[r][r'] + α·(adv − τ[r][r'])

于是 τ 的中性点为 0，正值表示该转移历史上常出现在**高于平均效率**的批次中。
这与 RL 中 advantage 的角色一致，也让 τ 具有明确的可解释性。

调制：cost = d · exp(−γ·τ)，γ 为信息素重要性。τ>0 → 代价下降 → 更倾向选择。

=============================================================================
避免趋同：局部更新（ACS 的 local update）
=============================================================================
在线 MAPD 中所有 agent 共享同一个 τ 场，若都追随同一模式会造成争抢。
故 agent 每选走一条边，立即向中性点收缩该边：

    τ[r][r'] ← (1 − ξ)·τ[r][r']

ξ = 0 时退化为无多样性机制（消融对照）。

=============================================================================
序列构造器
=============================================================================
nn      最近邻扩展（无记忆贪心；多数 MT-MAPD 求解器的默认做法）
ph      优势信息素调制的扩展（本文方法）
oracle  先用 nn 定集合，再穷举所有排列取几何总长最小者（顺序上界）
random  随机选、随机序（下界对照，验证系统有分辨力）
"""
import math
import random

MARGIN_SCALE = 25.0      # 边际行程的尺度常数（步）
TAU_MIN = -1.0
TAU_MAX = 1.0


class TransitionPheromone:
    """区域间转移偏好场（优势记忆）。EMA 更新，与 Phase 1 的场同构便于对照。

    mode:
        ema     有跨窗口记忆（本文方法）
        window  α = 1，纯当前窗口平均，无累积（无记忆对照）
        oracle  τ 固定（事后真实优势场）
        none    不更新，恒为 0（中性）
    """

    def __init__(self, n, tau_phi=500.0, window=10, mode="ema",
                 fixed_tau=None, xi=0.0):
        self.n = n
        self.tau_phi = float(tau_phi)
        self.window = int(window)
        self.mode = mode
        self.xi = float(xi)          # 局部更新强度（多样性机制）
        self.tau = [[0.0] * n for _ in range(n)]
        if fixed_tau is not None:
            self.tau = [list(r) for r in fixed_tau]
        self.obs_sum = [[0.0] * n for _ in range(n)]
        self.obs_cnt = [[0] * n for _ in range(n)]
        self.cum_sum = [[0.0] * n for _ in range(n)]
        self.cum_cnt = [[0] * n for _ in range(n)]
        self.o_bar = None            # 全局效率基线（滑动平均）
        self.n_batches = 0
        self.history = []
        self.record = False

    # ---------- 观测 ----------
    def observe_batch(self, region_seq, margin_per_task):
        """沉积一次批观测。region_seq 为该批经过的区域序列（含起点区域）。"""
        if margin_per_task < 0:
            margin_per_task = 0.0
        o = MARGIN_SCALE / (MARGIN_SCALE + margin_per_task)
        if self.o_bar is None:
            self.o_bar = o
        else:
            self.o_bar += 0.05 * (o - self.o_bar)      # 基线的慢速 EMA
        adv = o - self.o_bar
        self.n_batches += 1
        for i in range(len(region_seq) - 1):
            r, s = region_seq[i], region_seq[i + 1]
            if r < 0 or s < 0:
                continue
            self.obs_sum[r][s] += adv
            self.obs_cnt[r][s] += 1
            self.cum_sum[r][s] += adv
            self.cum_cnt[r][s] += 1

    def local_update(self, r, s):
        """多样性机制：选走一条边后立即向中性点收缩（ACS local update）。"""
        if r < 0 or s < 0 or self.xi <= 0:
            return
        self.tau[r][s] *= (1.0 - self.xi)

    # ---------- 更新 ----------
    def tick(self, t=0):
        if self.mode in ("oracle", "none"):
            self._clear_window()
            return
        alpha = 1.0 if self.mode == "window" else \
            min(1.0, self.window / max(self.tau_phi, 1e-9))
        for r in range(self.n):
            row = self.tau[r]
            os_, oc = self.obs_sum[r], self.obs_cnt[r]
            for s in range(self.n):
                if oc[s] > 0:
                    row[s] += alpha * (os_[s] / oc[s] - row[s])
                    if row[s] < TAU_MIN:
                        row[s] = TAU_MIN
                    elif row[s] > TAU_MAX:
                        row[s] = TAU_MAX
        self._clear_window()
        if self.record:
            self.history.append((t, [tuple(r) for r in self.tau]))

    def _clear_window(self):
        self.obs_sum = [[0.0] * self.n for _ in range(self.n)]
        self.obs_cnt = [[0] * self.n for _ in range(self.n)]

    # ---------- 查询 ----------
    def value(self, r, s):
        """原始优势值（中性 0）。"""
        if r < 0 or s < 0:
            return 0.0
        return self.tau[r][s]

    def oracle_tau(self, min_cnt=10):
        """事后真实优势场：由累计观测构造，样本不足的边为中性 0。"""
        out = []
        for r in range(self.n):
            row = []
            for s in range(self.n):
                if self.cum_cnt[r][s] >= min_cnt:
                    row.append(self.cum_sum[r][s] / self.cum_cnt[r][s])
                else:
                    row.append(0.0)
            out.append(row)
        return out

    def snapshot(self):
        return [tuple(r) for r in self.tau]

    def spread(self):
        """τ 的空间离散度（报告用）：(p95 − p5)。中性点 0，故为绝对差。"""
        vals = sorted(self.tau[r][s] for r in range(self.n)
                      for s in range(self.n))
        if not vals:
            return 0.0
        lo = vals[int(0.05 * len(vals))]
        hi = vals[int(0.95 * len(vals))]
        return hi - lo

    def absmax(self):
        m = 0.0
        for r in range(self.n):
            for s in range(self.n):
                if abs(self.tau[r][s]) > m:
                    m = abs(self.tau[r][s])
        return m


# =============================================================================
# 序列构造
# =============================================================================
def _pick_choice(wh, from_pos, task):
    """为 task 选择 pickup / delivery endpoint（多候选时取最近者）。"""
    pks = task.get("pickups") or [task["pickup"]]
    dls = task.get("deliveries") or [task["delivery"]]
    bp, bd, bd1 = pks[0], dls[0], -1
    for p in pks:
        d = wh.dist(from_pos, p)
        if d >= 0 and (bd1 < 0 or d < bd1):
            bd1, bp = d, p
    best, bd2 = -1, dls[0]
    for d in dls:
        dd = wh.dist(bp, d)
        if dd >= 0 and (best < 0 or dd < best):
            best, bd2 = dd, d
    return bp, bd2


def _seq_length(wh, start, picks, delivers):
    """几何总长：start → p_σ1 → … → p_σc → d_σ1 → … → d_σc。"""
    if not picks:
        return 0
    L = wh.dist(start, picks[0])
    if L < 0:
        return -1
    for i in range(len(picks) - 1):
        d = wh.dist(picks[i], picks[i + 1])
        if d < 0:
            return -1
        L += d
    d = wh.dist(picks[-1], delivers[0])
    if d < 0:
        return -1
    L += d
    for i in range(len(delivers) - 1):
        d = wh.dist(delivers[i], delivers[i + 1])
        if d < 0:
            return -1
        L += d
    return L


def build_sequence(wh, start_pos, tasks, ph, mode="nn", cap=1,
                   gamma=1.0, rng=None, cand_limit=40, seed=None):
    """从候选任务池中构造一个批次（至多 cap 个任务）及其执行顺序。

    tasks: 未分配任务列表（不会被修改）
    seed:  可选的 (task, pickup, delivery) 三元组，强制作为批次首个元素。
           提供时，本函数只负责「扩展后续任务」，cap 仍表示**批次总任务数**。
           ——这保证各对比方法共享同一批「种子任务」（由 Hungarian 分配），
           从而让对照只差在序列构造上。
    返回 [(task, pickup, delivery), ...]；失败返回 []。
    """
    cap = max(1, int(cap))
    core = mode
    use_2opt = False
    if core.endswith("2opt"):
        use_2opt = True
        core = core[:-4]
    anchor = (core == "cluster")

    if seed is not None:
        seq = [seed]
        used = {id(seed[0])}
        pool = list(tasks)
        if len(pool) > cand_limit:
            pool = sorted(pool, key=lambda tk: wh.dist(start_pos, tk["pickups"][0]))
            pool = pool[:cand_limit]
        if not pool:
            return list(seq)
        if core == "random":
            rng = rng or random.Random(0)
            k = min(max(cap - 1, 0), len(pool))
            for tk in rng.sample(pool, k):
                seq.append((tk,) + _pick_choice(wh, start_pos, tk))
        else:
            _extend(wh, seq, pool, used, cap, ph, core, gamma, anchor=anchor)
        if core == "oracle" and len(seq) > 1:
            seq = _best_order(wh, start_pos, seq)
        elif use_2opt and len(seq) > 2:
            seq, _ = two_opt(wh, start_pos, seq)
        return seq

    if not tasks:
        return []

    pool = sorted(tasks, key=lambda tk: wh.dist(start_pos, tk["pickups"][0])) \
        if len(tasks) > cand_limit else list(tasks)
    if len(pool) > cand_limit:
        pool = pool[:cand_limit]

    if core == "random":
        rng = rng or random.Random(0)
        k = min(cap, len(pool))
        chosen = rng.sample(pool, k)
        out = []
        for tk in chosen:
            p, d = _pick_choice(wh, start_pos, tk)
            out.append((tk, p, d))
        return out

    # ---- 首个任务：最近邻（各模式一致，保证对照干净）----
    first, fp, fd = None, None, None
    bd = -1
    for tk in pool:
        p, d = _pick_choice(wh, start_pos, tk)
        c = wh.dist(start_pos, p)
        if c >= 0 and (bd < 0 or c < bd):
            bd, first, fp, fd = c, tk, p, d
    if first is None:
        return []
    seq = [(first, fp, fd)]

    used = {id(first)}
    _extend(wh, seq, pool, used, cap, ph, core, gamma, anchor=anchor)

    if core == "oracle" and len(seq) > 1:
        seq = _best_order(wh, start_pos, seq)
    elif use_2opt and len(seq) > 2:
        seq, _ = two_opt(wh, start_pos, seq)
    return seq


def _extend(wh, seq, pool, used, cap, ph, mode, gamma, anchor=False):
    """扩展后续任务。

    anchor=False（默认）：链式最近邻——每步相对「上一个 pickup」选最近的。
                          会形成一条向外延伸的链，可能把整批拖长。
    anchor=True        ：聚簇式——始终相对「批次锚点」（第一个 pickup）选最近的。
                          形成紧凑簇，但不保证链的顺序最优。

    mode='ph' 时用 τ 调制代价：cost = d · exp(−γ·τ)，τ>0 表示历史上高效。
    """
    base_r = wh.region(*seq[0][1]) if anchor else None
    while len(seq) < cap:
        last_p = seq[-1][1]
        ref_p = seq[0][1] if anchor else last_p
        rl = wh.region(*last_p)
        best, btk, bp, bdl, brr = None, None, None, None, -1
        for tk in pool:
            if id(tk) in used:
                continue
            if tk.get("agent") is not None:
                continue
            p, d = _pick_choice(wh, ref_p, tk)
            dd = wh.dist(ref_p, p)
            if dd < 0:
                continue
            rr = wh.region(*p)
            if mode == "ph" and ph is not None:
                if anchor:
                    c = dd * math.exp(-gamma * ph.value(base_r, rr))
                else:
                    c = dd * math.exp(-gamma * ph.value(rl, rr))
            else:
                c = float(dd)
            if best is None or c < best:
                best, btk, bp, bdl, brr = c, tk, p, d, rr
        if btk is None:
            break
        used.add(id(btk))
        seq.append((btk, bp, bdl))
        if mode == "ph" and ph is not None:
            ph.local_update(rl, brr)


def two_opt(wh, start_pos, seq, max_pass=8):
    """对批次顺序做 2-opt 局部搜索（pick-all-then-deliver-all 结构）。

    序列 σ 同时决定 pickup 链与 delivery 链（同序），故目标是
        d(start, p_σ1) + Σ d(p_σi, p_σ{i+1}) + d(p_σc, d_σ1) + Σ d(d_σj, d_σ{j+1})
    2-opt 反转子区间并接受改进，直至无改进或达到轮数上限。
    """
    n = len(seq)
    if n < 3:
        return seq, _seq_length(wh, start_pos,
                                [it[1] for it in seq], [it[2] for it in seq])
    items = list(seq)

    def length(order):
        return _seq_length(wh, start_pos,
                           [items[i][1] for i in order],
                           [items[i][2] for i in order])

    order = list(range(n))
    best = length(order)
    if best < 0:
        return seq, -1.0
    for _ in range(max_pass):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                L = length(new)
                if 0 <= L < best - 1e-9:
                    best, order, improved = L, new, True
        if not improved:
            break
    return [items[i] for i in order], best


def _best_order(wh, start_pos, seq):
    """穷举排列取几何总长最小者。cap 很小（≤6）时开销可接受。"""
    n = len(seq)
    if n > 6:
        return seq
    items = list(seq)
    best, bl = None, -1
    idxs = list(range(n))

    def permute(cur, rest):
        nonlocal best, bl
        if not rest:
            picks = [items[i][1] for i in cur]
            dels = [items[i][2] for i in cur]
            L = _seq_length(wh, start_pos, picks, dels)
            if L >= 0 and (bl < 0 or L < bl):
                bl, best = L, list(cur)
            return
        for k, i in enumerate(rest):
            permute(cur + [i], rest[:k] + rest[k + 1:])

    permute([], idxs)
    if best is None:
        return seq
    return [items[i] for i in best]
