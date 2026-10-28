"""区域级任务**到达强度**场 λ̂_r（Phase 3）。

与前两个阶段的信息素载体有本质区别，这是本阶段能否做出增益的关键：

    Phase 1  τ_r        = 乘性拥堵系数   观测量 = 实际通行耗时 / 自由流耗时
                                        ↓ **内生**：策略改变观测目标
    Phase 2  τ[r][r']   = 转移偏好       观测量 = 实际批边际行程
                                        ↓ **内生**：同上
    Phase 3  λ̂_r        = 到达强度       观测量 = 任务生成事件本身
                                        ↓ **外生**：独立于 act 策略

内生观测存在自我否定的反馈回路：策略一旦成功避开某区域，该区域就不再拥堵，
下一轮观测随即将信息素抹平——记忆永远追不上自己造成的改变。
外生观测（"下一个任务会在哪里出现"）不受 act 策略影响，是稳定的估计目标，
因而统计规律可以被积累并反复利用。

Phase 1/2 的失败因此可以被重新表述为**载体选择错误**，而非"记忆无用"。

基础更新（EMA，时间常数 τ_φ）：
    α = min(1, window / τ_φ)
    λ̂_r ← λ̂_r + α · (c_r / window − λ̂_r)
τ_φ → window 时退化为纯滑动平均（无记忆对照）。

--------------------------------------------------------------------------
元适应学习率（Phase 3b：RL 控制蒸发率）
--------------------------------------------------------------------------
动机来自实验观测：漂移环境下固定 τ_φ 表现出**强单调依赖**
    shift 周期 1500：τ=100 → −21.3%，τ=2000 → −13.2%（差 8 个百分点）
而平稳环境下 τ_φ 几乎无影响（−32.5% ~ −32.6%）。
即：**不存在跨环境最优的固定蒸发率**，这解释了为什么"调参型 RL"（P1 范式）
天花板很低——它只是在换一个常数。

真正的解是把蒸发率变成**由环境在线决定的量**：
    Tabone, Ermentrout & Doiron (2010, J. Theor. Biol. 266:391)
    的 mean-field 模型给出：最优蒸发率由**环境变化的时间尺度**决定。
本模块用**双时间尺度估计比较**在线估计该尺度（ADWIN 式漂移检测）：

    维护两套 EMA：快估计 p̂_fast（τ_fast）与慢估计 p̂_slow（τ_slow）
    漂移强度   s(t) = TV(p̂_fast, p̂_slow) / s_scale       ∈ [0, 1]
    有效学习率 α(t) = α_slow + (1 − α_slow) · s(t)

    s → 0（稳定）：α = α_slow，长记忆，充分降噪
    s → 1（剧变）：α → 1，完全遗忘，立即采用当前窗口

这在 ACO 语义下就是**自适应蒸发率** ρ_eff = 1 − α_eff：
环境骤变时 ρ → 1（抹掉过时记忆），稳定时 ρ → ρ_slow（保留积累）。
RL 的角色因此是"控场"而非"调参"——它学的是**何时该遗忘**。
"""
import math


class DemandField:
    def __init__(self, n_regions, tau_phi=500.0, window=10,
                 mode="ema", prior=0.0, fixed=None, xi=0.0,
                 meta=False, tau_fast=None, ratio=8.0, s_scale=0.30):
        """mode:
            ema     -> 跨窗口指数记忆（本文方法）
            window  -> α=1，纯当前窗口平均（无积累对照）
            oracle  -> λ̂ 固定为事后真实分布（genie 上界）
            none    -> 均匀先验，不更新（等价于无信息素基线）

        meta（元适应，RL 控制蒸发率）:
            meta=True  -> 双时间尺度漂移检测，α 在线自适应
            tau_fast   -> 快估计时间常数（默认 τ_φ）
            ratio      -> τ_slow / τ_fast（默认 8）
            s_scale    -> 漂移强度归一化尺度（TV 距离达到此值即 s=1）
        """
        self.n = n_regions
        self.tau_phi = float(tau_phi)
        self.window = int(window)
        self.mode = mode
        self.xi = float(xi)          # ACS 局部衰减强度（生态位分化）
        self.prior = float(prior)
        self.lam = [self.prior] * n_regions
        if fixed is not None:
            self.lam = list(fixed)
        self.cnt = [0] * n_regions
        self.cum = [0] * n_regions          # 长期累计（构造 oracle 用）
        self.hist = []
        self.record = False

        # ---- 元适应：双时间尺度 ----
        self.meta = bool(meta)
        self.tau_fast = float(tau_fast) if tau_fast else float(tau_phi)
        self.ratio = float(ratio)
        self.tau_slow = self.tau_fast * self.ratio
        self.s_scale = float(s_scale)
        self.lam_fast = [self.prior] * n_regions
        self.lam_slow = [self.prior] * n_regions
        self.drift = 0.0            # 当前漂移强度 s(t)
        self.alpha_eff = 0.0        # 当前有效学习率
        self.drift_hist = []
        self.alpha_hist = []

        # ---- RL 层：价值乘子 + 资格迹（延迟信用分配）----
        self.mult = [1.0] * n_regions       # 学到的每区域边际价值
        self.use_mult = False
        self.m_hist = []

    # ---------- RL：边际价值学习 ----------
    def set_mult(self, values):
        self.mult = list(values)

    def reinforce(self, elig_vec, advantage, eta):
        """REINFORCE with baseline：按每 agent 的资格迹把优势回溯到决策。

        到达频率 f̂_r 回答的是"哪里会出现任务"，但**多个 ant 同时存在时**，
        停在热点只有当自己真正被选中才有收益。热点挤满时边际收益趋近于零。
        频率场不含这个竞争信息；乘子 m̂_r 由**已实现优势**学出，
        正是多 agent 信用分配（multi-agent credit assignment）的解。
        """
        if not self.use_mult or not elig_vec:
            return
        for r in range(self.n):
            e = elig_vec[r]
            if e > 0:
                v = self.mult[r] + eta * advantage * e
                if v < 0.05:
                    v = 0.05
                elif v > 20.0:
                    v = 20.0
                self.mult[r] = v

    def decay_traces(self, vec, factor):
        for r in range(self.n):
            if vec[r]:
                vec[r] *= factor
                if vec[r] < 1e-3:
                    vec[r] = 0.0

    def new_trace(self, region):
        """新的预定位决策建立（替换型）资格迹。"""
        e = [0.0] * self.n
        if 0 <= region < self.n:
            e[region] = 1.0
        return e

    # ---------- 观测：一次任务到达事件 ----------
    def observe(self, region):
        if 0 <= region < self.n:
            self.cnt[region] += 1
            self.cum[region] += 1

    # ---------- 元适应：漂移检测 ----------
    def _alpha_of(self, tau):
        return min(1.0, self.window / max(tau, 1e-9))

    def _update_meta(self):
        """一步双时间尺度更新 + 漂移强度估计 + 有效学习率合成。"""
        w = float(self.window)
        a_fast = self._alpha_of(self.tau_fast)
        a_slow = self._alpha_of(self.tau_slow)
        for r in range(self.n):
            obs = self.cnt[r] / w
            self.lam_fast[r] += a_fast * (obs - self.lam_fast[r])
            self.lam_slow[r] += a_slow * (obs - self.lam_slow[r])

        # 总变差距离（归一化后 ∈ [0,1]）作为漂移强度
        pf = self._norm(self.lam_fast)
        ps = self._norm(self.lam_slow)
        tv = 0.5 * sum(abs(a - b) for a, b in zip(pf, ps))
        s = min(1.0, tv / self.s_scale) if self.s_scale > 1e-9 else 0.0
        self.drift = s

        # 有效学习率：在慢/快两个时间尺度之间插值。
        # 注意上界是 α_fast 而非 1.0 —— α=1 意味着只用当前窗口（本设定下
        # 仅约 0.8 个到达样本），噪声会淹没信号；插值到 α_fast 才是
        # "在长记忆与快速跟随之间切换"的正确语义。
        self.alpha_eff = a_slow + (a_fast - a_slow) * s

        # 主场：用 α_eff 更新（保证 prob() 读到的是自适应后的估计）
        for r in range(self.n):
            obs = self.cnt[r] / w
            self.lam[r] += self.alpha_eff * (obs - self.lam[r])

    @staticmethod
    def _norm(vec):
        # 用计数比例归一化；全零时返回均匀分布
        s = sum(vec)
        if s <= 1e-12:
            return [1.0 / len(vec)] * len(vec)
        return [x / s for x in vec]

    # ---------- 窗口更新 ----------
    def tick(self, t=0):
        if self.mode in ("oracle", "none", "fixed"):
            self.cnt = [0] * self.n
            return
        if self.meta:
            self._update_meta()
            if self.record:
                self.drift_hist.append((t, self.drift))
                self.alpha_hist.append((t, self.alpha_eff))
        else:
            alpha = 1.0 if self.mode == "window" \
                else min(1.0, self.window / max(self.tau_phi, 1e-9))
            w = float(self.window)
            for r in range(self.n):
                obs = self.cnt[r] / w
                self.lam[r] += alpha * (obs - self.lam[r])
        self.cnt = [0] * self.n
        if self.record:
            self.hist.append((t, tuple(self.lam)))

    # ---------- 归一化概率分布 ----------
    def prob(self):
        """归一化权重；开启 RL 乘子时为 f̂_r · m̂_r 的归一化。

        乘子是**相对量**，只改变各区域的相对吸引力，不改变总量纲。
        """
        if self.use_mult:
            ws = [max(self.lam[r], 0.0) * self.mult[r] for r in range(self.n)]
            s = sum(ws)
            if s <= 1e-12:
                return [1.0 / self.n] * self.n
            return [x / s for x in ws]
        s = sum(self.lam)
        if s <= 1e-12:
            return [1.0 / self.n] * self.n
        return [x / s for x in self.lam]

    def oracle_prob(self, floor=0.0):
        """事后真实到达分布（由长期计数构造）。"""
        c = list(self.cum)
        if floor > 0:
            c = [x + floor for x in c]
        s = sum(c)
        if s <= 1e-12:
            return [1.0 / self.n] * self.n
        return [x / s for x in c]

    # ---------- ACO 机制 ----------
    def local_decay(self, region):
        """ACS 局部更新：被选中后立即削弱，制造负反馈。

        空闲 agent 的预定位是**覆盖问题**而非单点优化——所有 agent 都想挤到同一个
        热点会让边际收益迅速消失。局部衰减正是 ACO 处理这类问题的标准机制
        （蚁群系统 Gambardella & Dorigo 1995），此处它产生 **生态位分化**：
        后来的 agent 自动被推向次热点，形成分布式覆盖。
        """
        if self.xi > 0 and 0 <= region < self.n:
            self.lam[region] *= (1.0 - self.xi)
            if self.lam[region] < 0.0:
                self.lam[region] = 0.0

    def global_deposit(self, region, amount):
        """MMAS 式精英沉积：只对产生高回报的决策加权。"""
        if 0 <= region < self.n:
            self.lam[region] += amount

    # ---------- 诊断 ----------
    def spread(self):
        """分布离散度（归一化熵的补），衡量记忆是否真的学到了结构。"""
        p = self.prob()
        if self.n <= 1:
            return 0.0
        H = -sum(x * math.log(x) for x in p if x > 1e-15)
        return 1.0 - H / math.log(self.n)

    def l1_to(self, other):
        """与参考分布（如 oracle）的 L1 距离，衡量估计质量。"""
        p, q = self.prob(), other
        return sum(abs(a - b) for a, b in zip(p, q))

    def snapshot(self):
        return tuple(self.lam)

    def mean_rate(self):
        return sum(self.lam)
