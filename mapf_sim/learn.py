"""学习类 repositioning 基线：RL / AM **直接输出驻留位置**。

与本文方法 MEMIR 的对照关系
------------------------------------------------------------------
MEMIR 是**解析式**的：把驻留位置选择形式化为加权 1-中位数，用外生的到达强度场
λ̂ 做权重、用离线算一次的结构化流载先验做惩罚项，一次前向即得，不需要训练。

本模块是**经验式**的：不告诉它任何结构，让它从自己经历过的回报里学
「空闲时该站在哪一格」。动作空间就是候选 rest cell 本身（与 MEMIR 相同），
因此是严格意义上的 "learned repositioning" 基线。

三个基线
------------------------------------------------------------------
    lin   线性候选打分器 + REINFORCE with baseline
    tab   上下文 bandit：Q(状态签名, 锚点格)，锚点为候选集的最远点采样子集
    attn  Attention Model 式指针网络（Kool et al. 2019 的 pointer/glimpse 结构）：
          候选集 → 嵌入 → 由 agent 上下文做注意力池化得到 glimpse u → 指针打分，
          用 REINFORCE 训练。梯度为手写解析梯度（已用有限差分校验，见
          `check_learn_grad.py`）。

信息对齐（公平性说明）
------------------------------------------------------------------
三者都拿到与 MEMIR **完全相同**的信息：
    · p̂_r（区域到达强度 EMA）   → 特征 expected_cost(c, p̂) = Σ_r p̂_r d̄(c,r)
    · 图距离场                   → 特征 d(pos, c)
    · 结构化流载先验 flow(c)     → 特征 flow(c)
并且额外拿到 MEMIR **没有**的信息（邻域空闲 agent 数、局部需求质量），
所以本对照对 MEMIR 是**偏严**的：若学习类仍不能取胜，不能归因于信息不足。

回报
------------------------------------------------------------------
驻留决策的收益要等到该 agent 下一次被派单才兑现，形式为**响应距离**
（派单时刻 agent 到 pickup endpoint 的最短路长度）。因此这是一个
延迟、稀疏、非平稳的 contextual bandit / 策略梯度问题。
优势量沿用 sim 既有约定：A = (baseline − d0) / baseline，正 = 优于平均。

零第三方依赖。
"""
import math
import random


# --------------------------------------------------------------------------
# 候选格特征
# --------------------------------------------------------------------------
def _anchors(wh, cells, k):
    """候选集上的最远点采样（farthest-point sampling），缓存到 warehouse 上。

    用于 tab 基线的动作离散化：动作空间从 |cells|（数百）压到 k 个锚点格，
    这是把 RL 用到连续位置输出上时的标准做法。
    """
    ck = (tuple(sorted(cells)), k)
    cache = getattr(wh, "_anchor_cache", None)
    if cache is None:
        cache = {}
        wh._anchor_cache = cache
    if ck in cache:
        return cache[ck]
    k = min(int(k), len(cells))
    out = [cells[0]]
    far = [wh.dist(cells[0], c) for c in cells]
    while len(out) < k:
        j = max(range(len(cells)), key=lambda i: far[i] if far[i] >= 0 else 0)
        if cells[j] in out:
            break
        out.append(cells[j])
        dj = wh.dist_field(cells[j])
        W = wh.W               # 距离场扁平：索引 y * W + x
        for i, c in enumerate(cells):
            d = dj[c[1] * W + c[0]]
            if d >= 0 and d < far[i]:
                far[i] = d
    cache[ck] = out
    return out


def _soft_mass(maps, cell, p, radius):
    """Σ_r p_r · exp(−d̄(cell, r) / radius)：cell 附近的软需求质量。"""
    s = 0.0
    for r in range(len(p)):
        w = p[r]
        if w > 0:
            d = maps[r].get(cell, 0.0)
            s += w * math.exp(-d / radius)
    return s


class Ctx:
    """一次决策的上下文（构造一次，供全部候选复用）。"""

    __slots__ = ("pos", "p", "occ", "others", "n_agents", "scale",
                 "top_r", "maps", "flow", "wh")

    def __init__(self, wh, pos, p, occ, others, n_agents, maps, flow):
        self.wh = wh
        self.pos = pos
        self.p = p
        self.occ = occ
        self.others = others          # 其它空闲 agent 当前所占的格
        self.n_agents = n_agents
        self.maps = maps
        self.flow = flow
        self.scale = float(wh.W + wh.H) + 1.0
        self.top_r = max(range(len(p)), key=lambda r: p[r]) if p else 0


def featurize(cell, ctx, wh):
    """候选格 cell 在上下文 ctx 下的特征向量 x ∈ R^8。"""
    p = ctx.p
    ec = 0.0
    for r in range(len(p)):
        w = p[r]
        if w > 0:
            ec += w * ctx.maps[r].get(cell, 0.0)
    sc = ctx.scale
    dpos = wh.dist(ctx.pos, cell)
    dpos = dpos / sc if dpos >= 0 else 1.0
    flowv = ctx.flow.get(cell, 0.0) if ctx.flow else 0.0
    # 邻域竞争：半径 3 内已停了多少个其它空闲 agent
    near = 0
    for q in ctx.others:
        d = wh.dist(q, cell)
        if 0 <= d <= 3:
            near += 1
    mass = _soft_mass(ctx.maps, cell, p, 3.0)
    # 到最近已被占用泊位格的距离（分散压力）
    docc = 1.0
    if ctx.occ:
        best = None
        for q in ctx.occ:
            d = wh.dist(q, cell)
            if d >= 0 and (best is None or d < best):
                best = d
        if best is not None:
            docc = min(1.0, best / 8.0)
    # 到最近站台的距离（越近越容易被后续送货流穿过）
    dst = 1.0
    st = getattr(wh, "station_cells", ())
    if st:
        best = None
        for s in st:
            d = wh.dist(s, cell)
            if d >= 0 and (best is None or d < best):
                best = d
        if best is not None:
            dst = min(1.0, best / sc)
    return [1.0, ec / 20.0, dpos, flowv, near / 4.0, mass, docc, dst]


def agent_context(ctx, wh):
    """agent 侧上下文 z ∈ R^4。"""
    p = ctx.p
    H = 0.0
    for x in p:
        if x > 1e-12:
            H -= x * math.log(x)
    n = max(1, len(p))
    spread = H / math.log(n) if n > 1 else 0.0
    dpos = wh.dist(ctx.pos, ctx.pos)
    near = 0
    for q in ctx.others:
        d = wh.dist(ctx.pos, q)
        if 0 <= d <= 5:
            near += 1
    return [1.0, spread, near / max(1.0, float(ctx.n_agents)),
            float(len(ctx.occ)) / max(1.0, float(ctx.n_agents))]


# --------------------------------------------------------------------------
# (a) 线性打分器 + REINFORCE
# --------------------------------------------------------------------------
class LinearScorer:
    """u(c) = θ·x(c)，策略 π ∝ exp(u / T)，REINFORCE with baseline 更新。"""

    def __init__(self, d, lr=0.05, temp=0.5, seed=0):
        self.d = d
        self.lr = lr
        self.T = max(1e-3, temp)
        self.theta = [0.0] * d
        self.rng = random.Random(seed * 7919 + 13)
        self.base = None
        self.n_upd = 0

    def scores(self, X):
        th = self.theta
        return [sum(th[i] * x[i] for i in range(self.d)) for x in X]

    @staticmethod
    def _softmax(u, T):
        m = max(u)
        w = [math.exp((v - m) / T) for v in u]
        s = sum(w)
        return [v / s for v in w]

    def act(self, X, greedy=False):
        u = self.scores(X)
        pi = self._softmax(u, self.T)
        if greedy:
            j = max(range(len(u)), key=lambda i: u[i])
        else:
            j = self._draw(pi)
        return j, pi, u

    def _draw(self, pi):
        x = self.rng.random()
        acc = 0.0
        for i, w in enumerate(pi):
            acc += w
            if x <= acc:
                return i
        return len(pi) - 1

    def update(self, X, j, pi, adv, bc=False):
        """REINFORCE：∇ log π = (x_j − E_π[x]) / T。bc=True 时等价于模仿学习。"""
        d = self.d
        ex = [0.0] * d
        for i, q in enumerate(pi):
            if q <= 0:
                continue
            xi = X[i]
            for k in range(d):
                ex[k] += q * xi[k]
        a = 1.0 if bc else adv
        g = self.lr * a / self.T
        th = self.theta
        xj = X[j]
        for k in range(d):
            th[k] += g * (xj[k] - ex[k])
        self.n_upd += 1
        # 简单权重裁剪，防止发散
        for k in range(d):
            if th[k] > 50.0:
                th[k] = 50.0
            elif th[k] < -50.0:
                th[k] = -50.0


# --------------------------------------------------------------------------
# (b) 上下文 bandit（表格 Q）
# --------------------------------------------------------------------------
class TabularBandit:
    """Q((当前锚点, 需求主区), 锚点)。

    驻留的收益在下一次派单时一次性兑现，后续状态由下一轮决策重新开启，
    因此按 contextual bandit（无自转移）更新：Q(s,a) ← Q + α(r − Q)。
    """

    def __init__(self, n_state, n_act, lr=0.2, eps=0.10, seed=0):
        self.ns = n_state
        self.na = n_act
        self.lr = lr
        self.eps = eps
        self.rng = random.Random(seed * 104729 + 7)
        self.q = {}
        self.base = None
        self.n_upd = 0

    def _key(self, s, a):
        return s * self.na + a

    def act(self, s, cand_actions, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return self.rng.choice(cand_actions)
        best, bv = cand_actions[0], None
        for a in cand_actions:
            v = self.q.get(self._key(s, a))
            if v is None:
                v = 0.0
            if bv is None or v > bv:
                best, bv = a, v
        return best

    def update(self, s, a, adv):
        k = self._key(s, a)
        old = self.q.get(k, 0.0)
        self.q[k] = old + self.lr * (adv - old)
        self.n_upd += 1


# --------------------------------------------------------------------------
# (c) Attention Model 式指针网络（手写解析梯度）
# --------------------------------------------------------------------------
class AttnPointer:
    """候选集 → 嵌入 e_c = tanh(W_e x_c)
       agent 上下文 → 查询 q = tanh(W_a z)
       注意力池化得到 glimpse u = Σ_c α_c e_c,  α = softmax(q·e / √d)
       指针打分 s_c = w1·e_c + w2·u + w3·(e_c ⊙ u)
       策略 π ∝ exp(s / T)，REINFORCE 更新。

    结构对应 Kool, van Hoof & Welling (ICLR 2019) 的 pointer/glimpse：
    上下文对候选集做一次注意力读（glimpse）后由指针头挑选一个候选。
    梯度为解析式，推导见 `grad_note()`；`check_learn_grad.py` 用有限差分校验。
    """

    def __init__(self, dx, dz, dh, lr=0.05, temp=0.5, seed=0):
        self.dx, self.dz, self.dh = dx, dz, dh
        self.lr = lr
        self.T = max(1e-3, temp)
        rng = random.Random(seed * 31337 + 17)
        sc = 0.35
        self.W_e = [[rng.gauss(0, sc) for _ in range(dx)] for _ in range(dh)]
        self.W_a = [[rng.gauss(0, sc) for _ in range(dz)] for _ in range(dh)]
        self.w1 = [0.0] * dh
        self.w2 = [0.0] * dh
        self.w3 = [0.0] * dh
        self.rng = random.Random(seed * 99991 + 3)
        self.base = None
        self.n_upd = 0

    # ---------- 前向 ----------
    def forward(self, X, z):
        dh, dx, dz = self.dh, self.dx, self.dz
        We, Wa = self.W_e, self.W_a
        E = []
        for x in X:
            e = []
            for i in range(dh):
                s = 0.0
                Wi = We[i]
                for j in range(dx):
                    s += Wi[j] * x[j]
                e.append(math.tanh(s))
            E.append(e)
        q = []
        for i in range(dh):
            s = 0.0
            Wi = Wa[i]
            for j in range(dz):
                s += Wi[j] * z[j]
            q.append(math.tanh(s))
        rt = math.sqrt(float(dh))
        logits = [sum(q[i] * e[i] for i in range(dh)) / rt for e in E]
        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        tot = sum(ex)
        al = [v / tot for v in ex]
        u = [0.0] * dh
        for c, a in enumerate(al):
            if a <= 0:
                continue
            ec = E[c]
            for i in range(dh):
                u[i] += a * ec[i]
        S = []
        w1, w2, w3 = self.w1, self.w2, self.w3
        for e in E:
            s = 0.0
            for i in range(dh):
                s += w1[i] * e[i] + w2[i] * u[i] + w3[i] * e[i] * u[i]
            S.append(s)
        m = max(S)
        ex = [math.exp((v - m) / self.T) for v in S]
        tot = sum(ex)
        pi = [v / tot for v in ex]
        return dict(E=E, q=q, al=al, u=u, S=S, pi=pi, X=X, z=z)

    def act(self, X, z, greedy=False):
        f = self.forward(X, z)
        pi = f["pi"]
        if greedy:
            j = max(range(len(pi)), key=lambda i: pi[i])
        else:
            j = self._draw(pi)
        return j, pi, f

    def _draw(self, pi):
        x = self.rng.random()
        acc = 0.0
        for i, w in enumerate(pi):
            acc += w
            if x <= acc:
                return i
        return len(pi) - 1

    # ---------- 解析梯度 ----------
    @staticmethod
    def grad_note():
        return (
            "令 A_j = ∂u/∂e_j = α_j I + (α_j/√d)(e_j − u) qᵀ，\n"
            "则 ∂s_c/∂e_j = 1[j=c]·(w1 + w3⊙u) + A_j (w2 + w3⊙e_c)。\n"
            "对 π 加权求和：G_j = π_j (w1 + w3⊙u) + A_j (w2 + w3⊙Ē)，Ē = Σ_c π_c e_c。\n"
            "q 的梯度：∂u/∂q = (1/√d) Σ_j α_j e_j (e_j − u)ᵀ，\n"
            "          ∂(Σ_c π_c s_c)/∂q = (∂u/∂q)ᵀ (w2 + w3⊙Ē)。\n"
            "tanh 的局部梯度按 (1 − v²) 逐元素回传。"
        )

    def _grad_single(self, f, a):
        """∂s_a/∂θ 与 ∂(Σ_c π_c s_c)/∂θ，返回统一结构的梯度字典。"""
        dh = self.dh
        E, q, al, u, pi = f["E"], f["q"], f["al"], f["u"], f["pi"]
        w1, w2, w3 = self.w1, self.w2, self.w3
        rt = math.sqrt(float(dh))

        def build(target_is_single, ta=None, tpi=None):
            # tvec_c：对 s_a 取 1[c=a]；对 Σπ s 取 π_c
            if target_is_single:
                tv = [0.0] * len(E)
                tv[ta] = 1.0
            else:
                tv = list(tpi)
            # Ē = Σ_c tv_c e_c
            Eb = [0.0] * dh
            for c, tc in enumerate(tv):
                if tc <= 0:
                    continue
                ec = E[c]
                for i in range(dh):
                    Eb[i] += tc * ec[i]
            # m1 = w2 + w3⊙Eb
            m1 = [w2[i] + w3[i] * Eb[i] for i in range(dh)]
            # m0 = w1 + w3⊙u
            m0 = [w1[i] + w3[i] * u[i] for i in range(dh)]
            # G_j（对 e_j 的梯度，长度 dh 的向量列表）
            G = []
            for j in range(len(E)):
                base = [tv[j] * m0[i] for i in range(dh)]
                # A_j m1 = α_j m1 + (α_j/√d)(e_j − u) (qᵀ m1)
                qm = sum(q[i] * m1[i] for i in range(dh))
                ej = E[j]
                co = al[j] / rt * qm
                for i in range(dh):
                    base[i] += al[j] * m1[i] + co * (ej[i] - u[i])
                G.append(base)
            # 对 W_e
            gWe = [[0.0] * self.dx for _ in range(dh)]
            for j, gj in enumerate(G):
                xj = f["X"][j]
                ej = E[j]
                for i in range(dh):
                    loc = gj[i] * (1.0 - ej[i] * ej[i])
                    if loc == 0.0:
                        continue
                    row = gWe[i]
                    for k in range(self.dx):
                        row[k] += loc * xj[k]
            # 对 q：gs_q = (∂u/∂q)ᵀ m1
            #   ∂u/∂q = (1/√d) Σ_j α_j e_j (e_j − u)ᵀ
            gs_q = [0.0] * dh
            for j in range(len(E)):
                ej = E[j]
                aj = al[j]
                if aj <= 0:
                    continue
                dot = sum((ej[i] - u[i]) * m1[i] for i in range(dh))
                co = aj * dot / rt
                for i in range(dh):
                    gs_q[i] += co * ej[i]
            # 对 W_a
            gWa = [[0.0] * self.dz for _ in range(dh)]
            z = f["z"]
            for i in range(dh):
                loc = gs_q[i] * (1.0 - q[i] * q[i])
                if loc == 0.0:
                    continue
                row = gWa[i]
                for k in range(self.dz):
                    row[k] += loc * z[k]
            # 对 w1/w2/w3
            gw1 = [Eb[i] for i in range(dh)]
            gw2 = list(u)
            gw3 = [Eb[i] * u[i] for i in range(dh)]
            return dict(We=gWe, Wa=gWa, w1=gw1, w2=gw2, w3=gw3)

        return build(True, ta=a), build(False, tpi=pi)

    def update(self, f, j, adv, bc=False):
        ga, ge = self._grad_single(f, j)
        a = 1.0 if bc else adv
        g = self.lr * a / self.T
        lr = g
        We = self.W_e
        for i in range(self.dh):
            ri, gi = We[i], ga["We"][i]
            ei = ge["We"][i]
            for k in range(self.dx):
                ri[k] += lr * (gi[k] - ei[k])
        Wa = self.W_a
        for i in range(self.dh):
            ri, gi = Wa[i], ga["Wa"][i]
            ei = ge["Wa"][i]
            for k in range(self.dz):
                ri[k] += lr * (gi[k] - ei[k])
        for name in ("w1", "w2", "w3"):
            v = getattr(self, name)
            gv = ga[name]
            ev = ge[name]
            for i in range(self.dh):
                v[i] += lr * (gv[i] - ev[i])
        self._clip()
        self.n_upd += 1

    def _clip(self):
        lim = 25.0
        for M in (self.W_e, self.W_a):
            for row in M:
                for i, v in enumerate(row):
                    if v > lim:
                        row[i] = lim
                    elif v < -lim:
                        row[i] = -lim
        for name in ("w1", "w2", "w3"):
            v = getattr(self, name)
            for i, x in enumerate(v):
                if x > lim:
                    v[i] = lim
                elif x < -lim:
                    v[i] = -lim


# --------------------------------------------------------------------------
# 统一包装
# --------------------------------------------------------------------------
class LearnedPolicy:
    """对外接口：choose(ctx-ish) / feedback(agent_id, d0)。

    kind: lin | tab | attn
    mode: rl（REINFORCE with baseline）| bc（模仿 hindsight oracle 的贪心选择）
    """

    def __init__(self, wh, cells, kind="lin", lr=0.05, temp=0.5, eps=0.10,
                 anchors=16, topk=20, seed=0, mode="rl", greedy=False,
                 hidden=8):
        self.wh = wh
        self.cells = list(cells)
        self.kind = kind
        self.mode = mode
        self.greedy = greedy
        self.topk = max(4, int(topk))
        self.rng = random.Random(seed * 6151 + 11)
        self.base = None            # 回报基线（响应距离的 EMA）
        self.n_fb = 0
        self.pending = {}           # agent_id -> 待结算的决策
        # dh（注意力隐层宽度）是唯一可调的容量旋钮；8 是论文主结果的配置，
        # Reviewer 2 Q5 的回应里把它放大到 32 重跑，检验"容量不足"这一解释。
        self.dx, self.dz, self.dh = 8, 4, max(4, int(hidden))
        if kind == "lin":
            self.m = LinearScorer(self.dx, lr=lr, temp=temp, seed=seed)
        elif kind == "tab":
            self.anchors = _anchors(wh, self.cells, anchors)
            n_reg = getattr(wh, "n_regions", 8)
            self.m = TabularBandit(len(self.anchors) * n_reg,
                                   len(self.anchors), lr=lr, eps=eps, seed=seed)
        elif kind == "attn":
            self.m = AttnPointer(self.dx, self.dz, self.dh, lr=lr, temp=temp,
                                 seed=seed)
        else:
            raise ValueError("unknown learn kind: %r" % kind)

    # ---------- 决策 ----------
    def choose(self, pos, p, occ, others, n_agents, maps, flow, avail,
               agent_id=0, exclude_self=None, teacher_fn=None):
        ctx = Ctx(self.wh, pos, p, occ, others, n_agents, maps, flow)
        cand = list(avail)
        if not cand:
            return None
        tidx = None
        if teacher_fn is not None:
            tidx = teacher_fn(cand)
        if self.kind == "tab":
            return self._choose_tab(ctx, cand, agent_id, tidx)
        # 预筛：与 MEMIR 一样先按期望代价取 top-K，控制开销
        if len(cand) > self.topk:
            cand = sorted(cand,
                          key=lambda c: sum(p[r] * maps[r].get(c, 0.0)
                                            for r in range(len(p))))[:self.topk]
            if teacher_fn is not None:
                tidx = teacher_fn(cand)
        X = [featurize(c, ctx, self.wh) for c in cand]
        if self.kind == "lin":
            j, pi, _u = self.m.act(X, greedy=self.greedy)
            self.pending[agent_id] = dict(kind="lin", X=X, j=j, pi=pi, t=tidx)
        else:
            z = agent_context(ctx, self.wh)
            j, pi, f = self.m.act(X, z, greedy=self.greedy)
            self.pending[agent_id] = dict(kind="attn", X=X, j=j, pi=pi, f=f,
                                          t=tidx)
        return cand[j]

    def _choose_tab(self, ctx, cand, agent_id, tidx=None):
        anchors = self.anchors
        wh = self.wh
        # 状态签名 = (离当前位置最近的锚点, 需求主区域)
        s_a = min(range(len(anchors)),
                  key=lambda i: wh.dist(ctx.pos, anchors[i]))
        n_reg = wh.n_regions
        st = s_a * n_reg + ctx.top_r
        # 候选动作 = 所有当前可用的锚点
        aset = set(anchors)
        free_idx = [i for i, a in enumerate(anchors) if a in cand]
        if not free_idx:
            # 锚点全被占：退化为"离当前位置最近的可用格"
            c = min(cand, key=lambda q: wh.dist(ctx.pos, q))
            self.pending[agent_id] = None
            return c
        j = self.m.act(st, free_idx, greedy=self.greedy)
        t = None
        if tidx is not None:
            aj = anchors[tidx]
            if aj in cand and tidx in free_idx:
                t = tidx
        self.pending[agent_id] = dict(kind="tab", st=st, a=j, t=t)
        cell = anchors[j]
        if cell in cand:
            return cell
        # 锚点被占则取其最近可用格
        return min(cand, key=lambda q: wh.dist(cell, q))

    # ---------- 回报 ----------
    def feedback(self, agent_id, d0):
        """结算该 agent 上一次驻留决策的回报。d0 = 派单时的响应距离。"""
        if self.base is None:
            self.base = float(d0)
        base = max(self.base, 1.0)
        self.base += 0.02 * (d0 - self.base)
        adv = (base - d0) / base           # 正 = 优于平均
        rec = self.pending.pop(agent_id, None)
        self.n_fb += 1
        if rec is None:
            return
        bc = (self.mode == "bc")
        if self.kind == "lin":
            j = rec["t"] if (bc and rec.get("t") is not None) else rec["j"]
            self.m.update(rec["X"], j, rec["pi"], adv, bc=bc)
        elif self.kind == "attn":
            j = rec["t"] if (bc and rec.get("t") is not None) else rec["j"]
            self.m.update(rec["f"], j, adv, bc=bc)
        else:
            a = rec["t"] if (bc and rec.get("t") is not None) else rec["a"]
            self.m.update(rec["st"], a, adv)

    def snapshot(self):
        if self.kind == "lin":
            return dict(kind="lin", theta=list(self.m.theta))
        if self.kind == "tab":
            return dict(kind="tab", q=dict(self.m.q))
        return dict(kind="attn", W_e=[list(r) for r in self.m.W_e],
                    W_a=[list(r) for r in self.m.W_a],
                    w1=list(self.m.w1), w2=list(self.m.w2),
                    w3=list(self.m.w3))
