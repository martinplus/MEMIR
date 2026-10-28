"""区域级信息素场：τ_r 为乘性拥堵系数（实际通行耗时 / 自由流耗时）。

更新语义（与实验方案 §3.2 一致）：
    每个观测窗口统计各区域的「被迫等待比例」 p_r，
    观测值 o_r = 1 / (1 - p_r)（等待比例 p 意味着通行时间放大 1/(1-p) 倍），
    再以 EMA 更新 τ_r，时间常数 τ_φ：
        α = min(1, window / τ_φ)
        τ_r ← τ_r + α · (o_r - τ_r)

mode:
    ema     -> 有跨窗口记忆（本文方法）
    window  -> α = 1，纯当前窗口平均，无累积（「滑动平均」对照，L6）
    oracle  -> τ 固定不变（事后真实场，L0）
"""


class PheromoneField:
    def __init__(self, n_regions, tau_phi=1000.0, window=10,
                 tau_max=3.0, mode="ema", fixed_tau=None):
        self.n = n_regions
        self.tau_phi = float(tau_phi)
        self.window = int(window)
        self.tau_max = tau_max
        self.mode = mode
        self.tau = [1.0] * n_regions
        if fixed_tau is not None:
            self.tau = list(fixed_tau)
        # trip 观测：o = 实际耗时 / 自由流耗时（乘性拥堵系数）
        self.trip_cnt = [0] * n_regions
        self.trip_sum = [0.0] * n_regions
        self.cum_trip_cnt = [0] * n_regions
        self.cum_trip_sum = [0.0] * n_regions
        # wait 观测（仅用于报告拥堵度 L）
        self.step_cnt = [0] * n_regions
        self.wait_cnt = [0] * n_regions
        self.history = []          # [(t, tuple(tau))]
        self.record = False

    # ---------- 观测 ----------
    def observe_wait(self, region, waited):
        self.step_cnt[region] += 1
        if waited:
            self.wait_cnt[region] += 1

    def observe_trip(self, regions, o):
        """记录一次行程的拥堵系数观测，归因到起点区域与终点区域。"""
        if o > self.tau_max:
            o = self.tau_max
        for r in regions:
            self.trip_cnt[r] += 1
            self.trip_sum[r] += o
            self.cum_trip_cnt[r] += 1
            self.cum_trip_sum[r] += o

    # ---------- 更新 ----------
    def tick(self, t=0):
        if self.mode == "oracle":
            self.trip_cnt = [0] * self.n
            self.trip_sum = [0.0] * self.n
            return
        alpha = 1.0 if self.mode == "window" else min(1.0, self.window / max(self.tau_phi, 1e-9))
        for r in range(self.n):
            if self.trip_cnt[r] > 0:
                o = self.trip_sum[r] / self.trip_cnt[r]
                if o > self.tau_max:
                    o = self.tau_max
                self.tau[r] += alpha * (o - self.tau[r])
        self.trip_cnt = [0] * self.n
        self.trip_sum = [0.0] * self.n
        if self.record:
            self.history.append((t, tuple(self.tau)))

    # ---------- 查询 ----------
    def factor(self, r):
        return self.tau[r]

    def set_tau(self, values):
        self.tau = list(values)

    def oracle_tau(self, tau_max=None):
        """由长期累计统计构造真实拥堵系数场（事后 oracle）。样本不足的区域为 1。"""
        cap = tau_max or self.tau_max
        out = []
        for r in range(self.n):
            if self.cum_trip_cnt[r] >= 20:
                o = self.cum_trip_sum[r] / self.cum_trip_cnt[r]
                out.append(min(o, cap))
            else:
                out.append(1.0)
        return out

    def wait_ratio(self):
        tot = sum(self.step_cnt)
        return (sum(self.wait_cnt) / tot) if tot else 0.0

    def mean_trip_obs(self):
        """全局平均行程拥堵系数（报告用）。"""
        c = sum(self.cum_trip_cnt)
        return (sum(self.cum_trip_sum) / c) if c else 1.0

    def snapshot(self):
        return tuple(self.tau)
