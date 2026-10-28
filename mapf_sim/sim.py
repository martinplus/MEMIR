"""MAPD 在线仿真（lifelong），支持带容量 agent（MT-MAPD）与 open-loop 任务到达。
零第三方依赖。

时序（每个 timestep）：
    drift -> 信息素窗口更新 -> 执行一步 -> 批次/任务完成检测
          -> 任务到达 -> 分配与规划 -> 空闲 agent 预定位决策 -> 冲突检测

冲突模型：vertex + edge(swap) 禁止，following 允许。

三个阶段共用本文件：
    capacity == 1 且 arrival_mode == "closed"  -> Phase 1 行为，完全复现旧结果
    capacity  > 1                              -> Phase 2 行为（MT-MAPD 批次）
    arrival_mode == "open"                     -> Phase 3（open-loop + 预定位）
"""
import math
import random
import time

from planner import (ReservationTable, PlanFailure, st_astar, plan_task,
                     dwell_safe, dwell_violation)
from pheromone import PheromoneField          # 区域级拥堵系数（Phase 1，已证无效）
from seq import TransitionPheromone, build_sequence, _seq_length  # 区域间转移偏好（Phase 2）
from demand import DemandField                # 区域级到达强度（Phase 3）
from rest import RestPolicy, rest_cells_for   # 空闲 agent 预定位策略（Phase 3）
from assign import assign


def _poisson(rng, lam):
    """Knuth 采样。λ 较小时高效；本环境 λ<5，足够。"""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


class SimConfig:
    def __init__(self, **kw):
        self.n_agents = 24
        self.T = 2000
        self.assign_mode = "hungarian"   # hungarian | greedy
        # ---- Phase 1（区域级拥堵系数，已证无效，默认关闭）----
        self.ph_mode = "none"            # none | ema | window | oracle
        self.tau_phi = 1000.0
        self.modulate = "none"           # none | mean | target
        self.prio = "none"
        # ---- Phase 2（序列级转移偏好）----
        self.capacity = 1                # agent 一次携带的任务数（1 = ST-MAPD）
        self.seq_mode = "nn"             # nn | ph | oracle | random
        self.gamma = 1.0                 # 信息素重要性指数
        self.ph2_mode = "ema"            # ema | window | oracle | none
        self.tau_phi2 = 500.0
        self.ph2_oracle = None           # 事后真实偏好场（oracle 模式用）
        self.xi = 0.0                    # 局部更新强度（多样性机制，ACS local update）
        # ---- 环境 ----
        self.zipf_alpha = 1.0            # 空间不均匀度（0 = 均匀）
        self.drift_mode = "static"       # static | periodic | shift
        self.drift_param = 2000
        self.pending_factor = 2.0
        self.window = 10
        self.tau_max = 3.0
        self.seed = 0
        self.oracle_tau = None
        self.warmup = 300
        self.series_step = 100
        self.record_tau = False
        self.n_options = 1          # 每个 task 的候选 pickup endpoint 数量
        self.n_deliveries = 1       # 每个 task 的候选 delivery endpoint 数量
        # ---- Phase 3：open-loop 到达 + 空闲 agent 预定位 ----
        self.arrival_mode = "closed"  # closed（维持队列深度）| open（泊松到达）
        self.lmbda = 0.30             # open 模式：每 timestep 的到达率 λ
        self.rest_policy = "nearest"  # stay | nearest | random | exp | oracle | softmax
        self.rest_beta = 0.0          # 改位代价系数（避免为微小收益长途跋涉）
        self.rest_temp = 0.0          # softmax 温度（0 = 贪心）
        self.rest_xi = 0.0            # ACS 局部衰减：生态位分化
        self.rest_deplete = 0.0       # 负信息素强度 η（覆盖耗减，对抗 1-中位数诅咒）
        self.svc_radius = 4.0         # 服务半径 R（耗减核 κ(d)=exp(-d/R) 的尺度）
        self.flow_cost = 0.0          # 驻留干扰避让：停在主干搬运通道上的代价
        # flow 场的来源：struct = 离线拓扑先验（rest.build_haul_load）
        #                 emp   = 在线观测的经验载货车流 EMA（内生源）
        #                 both  = 两者相加
        self.flow_mode = "struct"
        self.flow_tau = 200.0         # 经验车流 EMA 的时间常数
        self.meta_mode = False        # RL 控蒸发率：双时间尺度自适应
        self.tau_fast = 100.0         # 快估计时间常数
        self.meta_ratio = 8.0         # τ_slow / τ_fast
        self.meta_scale = 0.30        # 漂移强度归一化尺度
        self.rl_mode = "none"         # none | adv（REINFORCE with baseline）
        self.rl_eta = 0.05            # 价值乘子学习率
        self.elig = 0.0               # 资格迹衰减因子 λ_TD（延迟信用分配）
        self.reeval_interval = 40     # 已停泊 agent 的重新评估周期
        # 正在"驶向泊位"途中的空闲 agent 是否仍可被派单。
        #   0（默认，全部既有实验的语义）= 不可派单：必须走完改位路径才计入空闲。
        #     这在 |A| 很大、空闲率高时会让"改位"付出派单可用性的代价，
        #     是 O_scale 中 B1/|A|>=128 反转的直接原因（见 _reversal_diag）。
        #   1 = 可派单：现实中 AGV 在移库途中完全可以被重新下任务。
        # 默认 0 以保证与既有 4000+ runs 逐位一致；仅 P_avail 实验开启。
        self.assign_moving = 0
        self.move_thresh = 0.0        # 重新评估时的移动门槛（迟滞）
        self.demand_mode = "ema"      # ema | window | none | oracle
        self.tau_phi3 = 500.0         # 到达强度记忆的时间常数
        self.oracle_lam = None        # 事后真实到达场（oracle 策略用）
        # ---- liveness 守护（修 |A| 较大时的规划失败级联）----
        self.rest_set = "park"        # park（严格 well-formed）| any（含通道）
        self.dwell_safety = False     # TER-Safety：永久占位不得切断 task endpoint 互通
        self.dwell_probe = 8          # 每条 decision 最多检查几个候选
        # 无候选通过 TER 检查时的兜底：
        #   min  -> 取窗口内违规数最小者（可能 >0，只是启发式；既有全部实验）
        #   park -> 退到最近的 non-task endpoint（A1 下**保证** v=0，见 Lemma 3）
        self.dwell_fallback = "min"
        self.preempt = 0              # 让行：规划失败时最多驱逐几个阻塞的空闲 agent
        # ---- 现实性压力（Reviewer 2 Q6；默认 0 = 关闭，与既有全部实验一致）----
        self.comm_lag = 0             # 决策层读到的是 L 个 timestep 之前的到达场
        self.obs_noise = 0.0          # 到达事件被误记到随机区域的概率（定位不确定性）
        self.dyn_obs = 0              # 动态障碍物（人/叉车）个数，轨迹全程预预约
        self.het_lat = 0              # 半数 agent 的额外发车延迟（异质节拍）拍数
        self.demand_feedback = 0.0    # 到达区域按当前 idle 分布抽样的概率（策略依赖需求）
        self.flow_def = "single"      # flow 先验的定义：single|rand|any|multi
        self.flow_seed = 0            # rand 定义的 tie-break 随机种子（不动任务流）
        self.haul_probe = 0           # 只统计实际载货交通（用于算相关性），不写回
        self.tp_dummy = False         # Token Passing 的 dummy path（任务一次规划到泊位）
        # ---- 规划模式：tp（Ma et al. 2017 全时域）/ rhcr（Li et al. 2021 窗口重规划）----
        self.planner_mode = "tp"      # tp | rhcr
        self.rhcr_h = 10              # RHCR 重规划周期 h
        self.rhcr_w = 20              # RHCR 规划窗口 w
        # ---- 学习类 repositioning 基线（rest_policy="learn"）----
        # 对照目的：不告诉模型任何结构，让它从回报里学「空闲时站在哪一格」，
        # 动作空间就是候选 rest cell 本身，与 MEMIR 完全一致。详见 learn.py。
        self.learn_kind = "lin"       # lin | tab | attn
        self.learn_mode = "rl"        # rl（REINFORCE with baseline）| bc（模仿 oracle）
        self.learn_lr = 0.05
        self.learn_temp = 0.5         # 策略 softmax 温度
        self.learn_eps = 0.10         # tab 基线的 ε-greedy
        self.learn_anchor = 16        # tab 基线的锚点格数量
        self.learn_topk = 20          # lin/attn 的候选预筛规模
        self.learn_greedy = False     # 执行时取 argmax（关闭探索）
        self.learn_hidden = 8         # attn 基线的隐层宽度（容量旋钮）
        self.pop_flow = 0.0           # popsample 基线上的流载折减强度
        self.__dict__.update(kw)

    def key(self):
        return dict(n_agents=self.n_agents, T=self.T, assign_mode=self.assign_mode,
                    ph_mode=self.ph_mode, tau_phi=self.tau_phi, modulate=self.modulate,
                    zipf_alpha=self.zipf_alpha, drift_mode=self.drift_mode,
                    drift_param=self.drift_param, pending_factor=self.pending_factor,
                    capacity=self.capacity, seq_mode=self.seq_mode, gamma=self.gamma,
                    ph2_mode=self.ph2_mode, tau_phi2=self.tau_phi2, seed=self.seed,
                    arrival_mode=self.arrival_mode, lmbda=self.lmbda,
                    rest_policy=self.rest_policy, rest_beta=self.rest_beta,
                    demand_mode=self.demand_mode, tau_phi3=self.tau_phi3)


class MAPDSim:
    def __init__(self, wh, cfg):
        self.wh = wh
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.rt = ReservationTable()
        self.nb = wh.n_block_regions

        # ---- 任务分布：Zipf over 货架块区域 ----
        self.order = list(range(self.nb))
        self._rebuild_weights(cfg.zipf_alpha)

        mode = cfg.ph_mode if cfg.ph_mode in ("ema", "window", "oracle") else "window"
        self.ph = PheromoneField(wh.n_regions, tau_phi=cfg.tau_phi, window=cfg.window,
                                 tau_max=cfg.tau_max, mode=mode,
                                 fixed_tau=cfg.oracle_tau)
        self.ph.record = cfg.record_tau

        self.ph2 = TransitionPheromone(wh.n_regions, tau_phi=cfg.tau_phi2,
                                       window=cfg.window, mode=cfg.ph2_mode,
                                       fixed_tau=cfg.ph2_oracle, xi=cfg.xi)
        self.ph2.record = cfg.record_tau

        # ---- agents 初始停在 non-task endpoint ----
        parks = list(wh.parking_cells)
        self.rng.shuffle(parks)
        if cfg.n_agents > len(parks):
            raise ValueError(f"well-formed violated: {cfg.n_agents} agents > "
                             f"{len(parks)} non-task endpoints")
        self.agents = []
        for i in range(cfg.n_agents):
            # 异质节拍：半数 agent 发车前多等 het_lat 拍（慢车/大载荷）
            lat = int(cfg.het_lat) if (i % 2 == 0) else 0
            self.agents.append(dict(id=i, pos=parks[i], prev=parks[i], path=None,
                                    t0=0, idx=0, task=None, state="idle", stuck=0,
                                    batch=None, lat=lat))

        # ---- Phase 3：到达强度场与预定位器 ----
        self.df = DemandField(wh.n_regions, tau_phi=cfg.tau_phi3,
                              window=cfg.window, mode=cfg.demand_mode,
                              xi=cfg.rest_xi, fixed=cfg.oracle_lam,
                              meta=cfg.meta_mode, tau_fast=cfg.tau_fast,
                              ratio=cfg.meta_ratio, s_scale=cfg.meta_scale)
        self.df.record = cfg.record_tau
        self.df.use_mult = (cfg.rl_mode == "adv")
        # 到达强度场的历史快照（通信延迟 comm_lag 用）
        self._prob_hist = []
        self._fb_rng = random.Random(cfg.seed * 15485863 + 7)
        # 归属噪声用独立 RNG：不得扰动主任务流的随机序列（否则破坏 CRN 配对）
        self._nz_rng = random.Random(cfg.seed * 32452843 + 17)
        self._obs_pos = []            # 动态障碍物的当前（前沿）位置
        self._obs_end = []            # 各障碍物已预约到的时刻
        self._obs_H = 30              # 障碍物的滚动预测视野
        # 候选休息位置：'park' 严格保证 well-formed；'any' 允许停在通道，
        # 大幅扩展定位自由度（初始配置仍在 non-task endpoint，故不破坏 well-formed）
        if cfg.rest_set == "any":
            base = getattr(wh, "rest_candidates", None)
            if base is not None:
                cells = list(base)
            else:
                pk = set(wh.pickup_cells)
                cells = [c for c in wh.free_cells
                         if c[1] < wh.station_top and c not in pk]
        else:
            cells = list(wh.parking_cells)
        self.rp = RestPolicy(wh, cells, policy=cfg.rest_policy,
                             beta=cfg.rest_beta, temperature=cfg.rest_temp,
                             seed=cfg.seed, n_agents=cfg.n_agents,
                             deplete=cfg.rest_deplete,
                             svc_radius=cfg.svc_radius,
                             flow_cost=cfg.flow_cost,
                             learn_kind=cfg.learn_kind,
                             learn_lr=cfg.learn_lr, learn_temp=cfg.learn_temp,
                             learn_eps=cfg.learn_eps,
                             learn_anchor=cfg.learn_anchor,
                             learn_topk=cfg.learn_topk,
                             learn_mode=cfg.learn_mode,
                             learn_greedy=cfg.learn_greedy,
                             pop_flow=cfg.pop_flow,
                             flow_def=cfg.flow_def, flow_seed=cfg.flow_seed,
                             learn_hidden=cfg.learn_hidden)
        # TER-Safety 静态需求集：全部 task endpoint（pickup ∪ delivery）
        self._req_static = set(wh.pickup_cells) | set(wh.station_cells)
        # 可选的逐 timestep 追踪回调（用于绘制驻留分布图），默认关闭、零开销
        self._trace_cb = None
        # ---- 经验载货交通场（flow_mode=emp / both）----
        # 注意：与 flow_mode=struct 的离线先验不同，这个场是**内生的** ——
        # 它会受到本策略自身行为的影响（Braverman et al. 意义下的自我否定回路）。
        # 保留它是为了直接检验「载体必须外生」这一命题。
        self._haul_ema = {}
        self._struct_flow = getattr(self.rp, "flow", None)
        if cfg.flow_mode in ("emp", "both"):
            self.rp.flow = dict.fromkeys(cells, 0.0)

        # open-loop：整条任务流**预先生成**，使不同方法共享完全相同的任务
        # （common random numbers），配对比较的方差大幅下降。
        self._stream_rows = []
        self._stream_i = 0
        if cfg.arrival_mode == "open":
            self._build_stream()

        self.pending = []
        self.next_tid = 0
        self.stats = dict(completed=0, completed_post=0, service=[], service_post=[],
                          conflicts=0, plan_fail=0, steps=0, waits=0, series=[],
                          tau_hist=[],
                          # liveness 守护（Phase 4）
                          plan_attempt_fail=0, preempt_calls=0, preempt_yields=0,
                          dwell_checks=0, dwell_blocked=0, dwell_viol=0,
                          dwell_fb=0, dwell_fb_fail=0, dyn_obs_conf=0,
                          learn_curve=[],
                          decide_cnt=0, decide_ns=0,
                          # MT 专属统计
                          batches=0, batch_sizes=[], margin=[], travel=0, seq_geom=0,
                          empty_travel=0, seq_len=0, first_leg=0,
                          # Phase 3 专属统计
                          pending_sum=0, idle_steps=0, arrivals=0, rest_moves=0,
                          resp_dist=0, assign_cnt=0,
                          n_leg=0, leg_assign=0, leg_approach=0, leg_haul=0,
                          wait_assign=0, no_idle=0)

    # ---------------- 现实性压力：动态障碍物 ----------------
    def _build_dyn_obs(self):
        """初始化动态障碍物（人/叉车）：随机选起点，轨迹**在线滚动生成**。

        为什么不能一次性预约全程：驻留判定用的是 future_clear（该格此后
        **任何时刻**都不得有他人预约），若把 4000 步轨迹一次性预约，障碍物
        走过的每一个格子都会被永久划为禁停区，代价被放大到不真实的程度。
        因此改为滚动预测视野 H：每个 tick 把轨迹向前推进到 t+H 并预约，
        agent 看到的是"有限前瞻"的障碍物预测 —— 与真实系统一致。
        """
        cfg = self.cfg
        n = int(cfg.dyn_obs)
        if n <= 0:
            return
        rng = random.Random(cfg.seed * 104729 + 991)
        start = set(a["pos"] for a in self.agents)
        pool = [c for c in self.wh.free_cells if c not in start]
        if not pool:
            return
        self._obs_pos = [rng.choice(pool) for _ in range(n)]
        self._obs_end = [0] * n
        self._obs_H = int(getattr(cfg, "dyn_obs_h", 30))
        for k, c in enumerate(self._obs_pos):
            self.rt.cell[(c[0], c[1], 0)] = -1001 - k
            self.rt.by_cell.setdefault(c, set()).add(0)

    def _obs_next(self, cur, tt, oid):
        """障碍物在时刻 tt 的下一个位置；被占则原地不动（人会给 AGV 让路）。"""
        cand = [cur]
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            q = (cur[0] + dx, cur[1] + dy)
            if not (0 <= q[0] < self.wh.W and 0 <= q[1] < self.wh.H):
                continue
            if self.wh.grid[q[1]][q[0]] != 0:
                continue
            # 已有任何时空预约（别的 agent / 别的障碍物）-> 不可进入
            if self.rt.cell.get((q[0], q[1], tt)) is not None:
                continue
            # 已停驻的 agent（permanent）-> 不可进入。这条保证了"障碍物
            # 绝不撞上停着的 agent"，也正是允许滚动视野仍然安全的原因。
            pm = self.rt.permanent.get(q)
            if pm is not None and pm[1] <= tt:
                continue
            cand.append(q)
        return self.rng.choice(cand)

    def _dyn_obs_step(self, t):
        """把每个障碍物的预约向前推进到 t+H。"""
        if not self._obs_pos:
            return
        H = self._obs_H
        for k in range(len(self._obs_pos)):
            oid = -1001 - k
            cur = self._obs_pos[k]
            tt = self._obs_end[k]
            while tt < t + H:
                nxt = self._obs_next(cur, tt + 1, oid)
                if nxt != cur:
                    # 边预约（反向）：否则 agent 与障碍物可以"对穿"（swap）
                    self.rt.edge[(nxt[0], nxt[1], cur[0], cur[1], tt)] = oid
                cur = nxt
                tt += 1
                key = (cur[0], cur[1], tt)
                if self.rt.cell.get(key) is None:
                    self.rt.cell[key] = oid
                    self.rt.by_cell.setdefault(cur, set()).add(tt)
            self._obs_pos[k] = cur
            self._obs_end[k] = tt
        self.rt.max_t = max(self.rt.max_t, t + H)

    # ---------------- open-loop 任务流 ----------------
    def _build_stream(self):
        """用独立 RNG 预生成全部到达事件，并用与主循环**相同的漂移调度**。

        返回 rows = [(release_t, region, [pickup_opts], [delivery_opts]), ...]
        """
        cfg = self.cfg
        rng = random.Random(cfg.seed * 7919 + 13)
        order = list(range(self.nb))
        rows = []
        for t in range(cfg.T + 2):
            # 与主循环同构的漂移调度
            p = cfg.drift_param
            if cfg.drift_mode != "static" and p > 0 and t > 0 and t % p == 0:
                if cfg.drift_mode == "periodic":
                    order = order[1:] + order[:1]
                elif cfg.drift_mode == "shift":
                    rng.shuffle(order)
            n = _poisson(rng, cfg.lmbda)
            if n <= 0:
                continue
            w = self._weights_of(order, cfg.zipf_alpha)
            for _ in range(n):
                r = self._pick_region(rng, order, w)
                cells = self.wh.pickup_by_region.get(r) or []
                if not cells:
                    cells = self.wh.pickup_cells
                opts = [rng.choice(cells)]
                regs = [r]
                for _ in range(max(0, int(cfg.n_options) - 1)):
                    rr = rng.randrange(self.nb)
                    if rr in regs:
                        continue
                    cc = self.wh.pickup_by_region.get(rr) or []
                    if not cc:
                        continue
                    regs.append(rr)
                    opts.append(rng.choice(cc))
                st = self.wh.station_cells
                nd = max(1, int(cfg.n_deliveries))
                dls = rng.sample(st, min(nd, len(st)))
                rows.append((t, r, opts, dls))
        self._stream_rows = rows
        self._stream_i = 0

    @staticmethod
    def _weights_of(order, alpha):
        return [1.0 / ((i + 1) ** alpha) for i in range(len(order))]

    @staticmethod
    def _pick_region(rng, order, w):
        tot = sum(w)
        x = rng.random() * tot
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if x <= acc:
                return order[i]
        return order[-1]

    # ---------------- 任务流 ----------------
    def _rebuild_weights(self, alpha):
        n = self.nb
        self.weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]

    def _maybe_drift(self, t):
        p = self.cfg.drift_param
        if p <= 0 or t == 0 or t % p != 0:
            return
        if self.cfg.drift_mode == "periodic":
            self.order = self.order[1:] + self.order[:1]
        elif self.cfg.drift_mode == "shift":
            self.rng.shuffle(self.order)
        self._rebuild_weights(self.cfg.zipf_alpha)

    def _sample_region(self):
        fb = float(self.cfg.demand_feedback)
        if fb > 0 and self._fb_rng.random() < fb:
            # 策略依赖的需求：以概率 fb 把这次到达放在**当前某个空闲 agent
            # 所在的区域**。这让需求反过来追着驻留策略跑 —— 一条正反馈回路，
            # 直接检验"载体外生"这条前提被破坏时会发生什么。
            # 注意这里用独立 RNG，尽量不动主任务流的随机序列。
            idle = [a["pos"] for a in self.agents
                    if a["state"] in ("idle", "park")]
            if idle:
                return self.wh.region(*self._fb_rng.choice(idle))
        tot = sum(self.weights)
        x = self.rng.random() * tot
        acc = 0.0
        for i, w in enumerate(self.weights):
            acc += w
            if x <= acc:
                return self.order[i]
        return self.order[-1]

    def _gen_task(self, t):
        """closed-loop 即时生成（偿还队列）。返回 task dict。"""
        k = max(1, int(self.cfg.n_options))
        regs, opts = [], []
        r = self._sample_region()
        cells = self.wh.pickup_by_region.get(r) or []
        if cells:
            regs.append(r)
            opts.append(self.rng.choice(cells))
        for _ in range(k - 1):
            r = self.rng.randrange(self.nb)
            if r in regs:
                continue
            cells = self.wh.pickup_by_region.get(r) or []
            if not cells:
                continue
            regs.append(r)
            opts.append(self.rng.choice(cells))
        if not opts:
            opts = [self.rng.choice(self.wh.pickup_cells)]
        nd = max(1, int(self.cfg.n_deliveries))
        st = self.wh.station_cells
        dls = self.rng.sample(st, min(nd, len(st))) if nd < len(st) else list(st)
        anchor = regs[0] if regs else -1
        tk = dict(id=self.next_tid, pickups=opts, pickup=None,
                  deliveries=dls, delivery=None,
                  release=t, agent=None, region=anchor)
        self.next_tid += 1
        # 到达事件即信息素观测：此为**外生**量，不受 act 策略影响。
        # obs_noise>0：以该概率把这次到达误记到一个随机区域（定位不确定性的
        # 信息侧后果 —— 位置没错，错的是"这条需求落在哪个区"的归属）。
        if anchor >= 0:
            self.df.observe(self._maybe_misattribute(anchor))
        self.stats["arrivals"] += 1
        return tk

    def _maybe_misattribute(self, r):
        p = float(self.cfg.obs_noise)
        if p <= 0 or self._nz_rng.random() >= p:
            return r
        return self._nz_rng.randrange(self.wh.n_regions)

    def _replenish(self, t):
        if self.cfg.arrival_mode == "open":
            # 释放早已预生成的到达事件 —— 不同方法共享同一条任务流
            rows = self._stream_rows
            n = len(rows)
            i = self._stream_i
            while i < n and rows[i][0] <= t:
                rt_, rr, opts, dls = rows[i]
                # 策略依赖的需求：以概率 ρ 把这次到达的**取货端点**改到"某个
                # 空闲 agent 当前所在区域"里随机一个 pickup endpoint。释放时刻
                # 仍由预生成流决定（保持可比），只有**位置**开始追随策略 ——
                # 这就在 open-loop 到达下也造出了"需求追着驻留策略跑"的回路。
                rho = float(self.cfg.demand_feedback)
                if rho > 0 and self._fb_rng.random() < rho:
                    idle = [a["pos"] for a in self.agents
                            if a["state"] in ("idle", "park")]
                    if idle:
                        pos = self._fb_rng.choice(idle)
                        rr2 = self.wh.region(*pos)
                        cc = self.wh.pickup_by_region.get(rr2) or []
                        if cc:
                            rr = rr2
                            opts = [self._fb_rng.choice(cc)]
                self.pending.append(dict(id=self.next_tid, pickups=opts, pickup=None,
                                         deliveries=dls, delivery=None,
                                         release=rt_, agent=None, region=rr))
                self.next_tid += 1
                if rr >= 0:
                    self.df.observe(self._maybe_misattribute(rr))
                self.stats["arrivals"] += 1
                i += 1
            self._stream_i = i
            return
        target = int(self.cfg.pending_factor * self.cfg.n_agents)
        while len(self.pending) < target:
            self.pending.append(self._gen_task(t))

    # ---------------- 规划 ----------------
    def _plan_to_parking(self, a, t):
        """兜底安置：沿按距离排序的候选逐个尝试，直到找到**可永久停留**的位置。

        关键点：**必须**让每个空闲 agent 最终获得 permanent reservation。
        若只留下短时 try_hold（滚动词），其它 agent 在更晚时刻就看不到它，
        从而规划出穿过它的路径 —— 这正是 rest_set='any' 下冲突的唯一来源。
        """
        cands = sorted(self.rp.cells,
                       key=lambda c: self.wh.dist(a["pos"], c))
        # dwell_fallback="park" 时，兜底安置也**必须**落在安全格上，否则
        # Lemma 3 的归纳前提会被这条旁路破坏（实测 viol 就来自这里）。
        chk = self.cfg.dwell_safety and \
            self.cfg.dwell_fallback in ("park", "free")
        occ = self._occupied_cells(exclude=a["id"]) if chk else None
        for c in cands[:80]:
            if self.rt.permanent.get(c) is not None:
                continue
            if chk and self._dwell_cost(set(occ) | {c}) > 0:
                continue
            try:
                p = st_astar(self.wh, self.rt, a["pos"], c, t, a["id"])
                t_arr = t + len(p) - 1
                if not self.rt.future_clear(c[0], c[1], t_arr, a["id"]):
                    continue
                p = self._reg(a, p, t, True)
                a.update(path=p, t0=t, idx=0, state="park", task=None, batch=None,
                         rest_target=c, picked=False)
                self._mark_trace(a, c)
                return True
            except PlanFailure:
                continue
        # 极端情况：原地永久停下（宁可牺牲少量吞吐，也不能留下无预约的 agent）
        self._reg(a, [a["pos"]], t, True)
        a.update(path=[a["pos"]], t0=t, idx=0, state="park", task=None, batch=None,
                 rest_target=a["pos"] if a["pos"] in self.rp.cells else None,
                 picked=False)
        self.stats["rest_forced"] = self.stats.get("rest_forced", 0) + 1
        return True

    def _assign_and_plan(self, t):
        if self.cfg.assign_moving:
            # 移库途中的空闲 agent 也允许被派单（现实中 AGV 可中途改任务）。
            # 派单时 _plan_task -> _reg 会先 remove_agent 清掉旧预约，语义一致。
            idle = [a for a in self.agents if a["state"] in ("idle", "park")]
            self.stats["moving_idle"] = self.stats.get("moving_idle", 0) + sum(
                1 for a in idle
                if a["path"] is not None and a["idx"] < len(a["path"]) - 1)
        else:
            idle = [a for a in self.agents if a["state"] in ("idle", "park")
                    and (a["path"] is None or a["idx"] >= len(a["path"]) - 1)]
        unassigned = [tk for tk in self.pending if tk["agent"] is None]
        if not idle:
            # 没有任何处于「路径已走完」状态的空闲 agent —— 记录一次"无从调度"
            if unassigned:
                self.stats["no_idle"] += 1
            return
        # 为完全无 reservation 的 idle agent 先占位
        for a in idle:
            if a["path"] is None:
                self.rt.remove_agent(a["id"])
                if not self.rt.try_hold(a["id"], a["pos"], t, dur=3):
                    self._plan_to_parking(a, t)

        # L1 段：release -> 派单。只要还排在 pending 里，每次 tick 都累计一次
        for tk in unassigned:
            self.stats["wait_assign"] += 1

        tau = self.ph.tau if self.cfg.modulate != "none" else None
        pairs = []
        if unassigned:
            pairs = assign(self.wh, idle, unassigned, tau,
                           mode=self.cfg.assign_mode, modulate=self.cfg.modulate)
        assigned = set()
        if self.cfg.prio == "ph" and tau is not None:
            order = sorted(range(len(pairs)),
                           key=lambda k: -tau[self.wh.region(*idle[pairs[k][0]]["pos"])])
        else:
            order = range(len(pairs))

        cap = max(1, int(self.cfg.capacity))
        for k in order:
            ai, ti, ch = pairs[k]
            a, tk = idle[ai], unassigned[ti]
            if ch is None:
                continue
            pk, dl = ch
            if cap <= 1:
                if self._plan_task(a, tk, t, pk, dl):
                    assigned.add(a["id"])
                continue
            # ---- MT：以 Hungarian 分配的种子任务为起点，扩展成一批 ----
            seq = [(tk, pk, dl)]
            tk["agent"] = a["id"]
            rest = [x for x in unassigned if x["agent"] is None]
            extra = build_sequence(self.wh, pk, rest, self.ph2,
                                   mode=self.cfg.seq_mode, cap=cap,
                                   gamma=self.cfg.gamma, rng=self.rng,
                                   seed=seq[0])
            for it in extra[1:]:
                it[0]["agent"] = a["id"]
                seq.append(it)
            if self._plan_batch(a, seq, t):
                assigned.add(a["id"])
            else:
                for it in seq:
                    it[0]["agent"] = None
                # 规划失败则退化为单任务批次，保证 agent 不被卡死
                if len(seq) > 1 and self._plan_batch(a, seq[:1], t):
                    seq[0][0]["agent"] = a["id"]
                    assigned.add(a["id"])

        for a in idle:
            if a["id"] in assigned:
                continue
            moving = a["path"] is not None and a["idx"] < len(a["path"]) - 1
            if moving:
                continue
            if a["state"] == "park" \
                    and a.get("rest_t", -10 ** 9) > t - self.cfg.reeval_interval:
                continue
            self._rest_decide(a, t)
            a["rest_t"] = t

    # ---------------- Phase 3：空闲 agent 预定位 ----------------
    def _rest_prob(self):
        """当前用于加权中位的目标分布 p̂。"""
        cfg = self.cfg
        if cfg.demand_mode == "oracle" and cfg.oracle_lam is not None:
            s = sum(cfg.oracle_lam)
            if s <= 1e-12:
                return [1.0 / self.wh.n_regions] * self.wh.n_regions
            return [x / s for x in cfg.oracle_lam]
        if cfg.demand_mode == "none":
            return [1.0 / self.wh.n_regions] * self.wh.n_regions
        if cfg.comm_lag > 0 and self._prob_hist:
            # 通信延迟：决策层只能读到 L 个 timestep 之前的到达强度场。
            # 这是最贴近现实的「信息侧」降级：记忆本身没错，只是迟到了。
            i = max(0, len(self._prob_hist) - 1 - int(cfg.comm_lag))
            return self._prob_hist[i]
        return self.df.prob()

    def _occupied_cells(self, exclude=None):
        """被他人「长期占据」的格子集合。

        tp   模式：permanent reservation（Token Passing 语义）
        rhcr 模式：没有 permanent，改为各空闲 agent 的**目标泊位**，
                   使 rest 策略不会把两个 agent 派到同一格。
        """
        if self.cfg.planner_mode == "rhcr":
            occ = set()
            for b in self.agents:
                if b["id"] == exclude or b["state"] != "park":
                    continue
                occ.add(b.get("rest_target") or b["pos"])
            return occ
        occ = set()
        for c, own in self.rt.permanent.items():
            if own[0] != exclude:
                occ.add(c)
        return occ

    def _idle_cells(self, exclude=None):
        """其它「空闲/已停泊」agent 当前占用的格子（学习类基线的竞争特征）。"""
        out = []
        for b in self.agents:
            if b["id"] == exclude:
                continue
            if b["state"] in ("idle", "park"):
                out.append(b["pos"])
        return out

    def _reg(self, a, path, t, permanent):
        """统一注册：rhcr 模式下把路径截断到窗口长度 w，且**不**永久占位。"""
        if self.cfg.planner_mode == "rhcr":
            w = int(self.cfg.rhcr_w)
            h = max(1, int(self.cfg.rhcr_h))
            if w > 0 and len(path) > w + 1:
                path = path[:w + 1]
            # 注意：这里的补齐长度**不能**取到整个规划窗口 w —— 实验表明那会让
            # 空闲 agent 在通道上长期占位，系统彻底死锁（|A|≥8 即 0 吞吐）。
            # 因此 RHCR 变体暂不作为论文基线：idle agent 的驻留语义不属于
            # Li et al. (2021) 原始规范，需要单独的碰撞消解设计。本文基线为
            # Token Passing（Ma et al. 2017）的 tp / tp+dummy 两种标准变体。
            cover = h + 1
            if len(path) < cover:
                path = list(path) + [path[-1]] * (cover - len(path))
            permanent = False
        self.rt.remove_agent(a["id"])
        self.rt.add_path(a["id"], path, t, permanent=permanent)
        return path

    # ---------- RHCR：窗口化滚动重规划（Li et al. AAAI 2021）----------
    def _rhcr_tick(self, t):
        """每个重规划周期 h，为所有 agent 重新规划其所在窗口内的路径。"""
        cfg = self.cfg
        h = max(1, int(cfg.rhcr_h))
        for a in self.agents:
            remaining = 0 if a["path"] is None else \
                len(a["path"]) - 1 - a["idx"]
            global_tick = (t % h == 0)
            due = (remaining <= 0) or global_tick
            if a["state"] == "busy":
                tk = a["task"]
                if tk is None:
                    continue
                # 严格遵循 Li et al. (2021)：带任务的 agent **只在全局 tick** 重规划。
                # 窗口用尽时由 _rhcr_seal 把当前格续约一整个周期（原地等待），
                # 提前重规划会让他人已规划好的路径突然失效 -> 真实冲突。
                if not global_tick:
                    continue
                self._rhcr_replan_task(a, tk, t)
            else:
                if not due and a.get("rest_t", -10 ** 9) > t - cfg.reeval_interval:
                    continue
                self._rest_decide(a, t)
                a["rest_t"] = t

    def _rhcr_replan_task(self, a, tk, t):
        cfg = self.cfg
        try:
            if a.get("picked"):
                pth = st_astar(self.wh, self.rt, a["pos"], tk["delivery"], t,
                               a["id"])
            else:
                pth = plan_task(self.wh, self.rt, a["pos"], tk["pickup"],
                                tk["delivery"], t, a["id"])[0]
        except PlanFailure:
            self.stats["plan_attempt_fail"] += 1
            # RHCR 标准兜底：保持原路径不动（下个周期再试）
            return
        pth = self._reg(a, pth, t, False)
        a.update(path=pth, t0=t, idx=0)

    # ---------- Liveness 守护 ----------
    def _dwell_required(self):
        """TER-Safety 的「必须互通」节点集：全部 task endpoint + 全部 agent 当前位置。

        只要求这些节点在残余图里同属一个连通分量：没有任何 endpoint 的空口袋
        被切断不影响规划，因此允许 —— 这比「全图连通」弱得多，也就不会因为
        历史累积的既成违规而否决一切候选、把空闲 agent 集体冻死。
        """
        req = set(self._req_static)
        for a in self.agents:
            req.add(a["pos"])
        return req

    def _dwell_cost(self, blocked):
        """TER 违规数（不可达的 required 节点数），0 = 安全。"""
        self.stats["dwell_checks"] += 1
        v = dwell_violation(self.wh, blocked, self._dwell_required())
        if v:
            self.stats["dwell_viol"] += 1
        return v

    def _dwell_pick(self, occ, cands, pos=None):
        """在偏好序列里取 TER 违规最小者（平手取偏好序靠前者）。

        与「硬约束 + 逐个否决」不同，这里永远不会返回 None：最坏情况下退回
        偏好序首位，因此空闲 agent 不会被冻死；残余的少量违规交给 preempt 让行
        兜底。occ 为**他人**的永久停驻格集合（已排除本 agent 自己的格子）。

        dwell_fallback="park"：若窗口内**没有** v=0 的候选，则退到最近的
        non-task endpoint（parking endpoint）。在 A1（well-formed instance）
        下该集合中的格一定满足 v=0，于是返回值**恒为安全**，与探测窗口
        dwell_probe 的大小无关 —— 这正是 Lemma 3 的可执行版本。
        """
        if not cands:
            return None, None
        if not self.cfg.dwell_safety:
            return cands[0], 0
        best, bv = cands[0], None
        for c in cands[:max(1, int(self.cfg.dwell_probe))]:
            v = self._dwell_cost(set(occ) | {c})
            if bv is None or v < bv:
                best, bv = c, v
                if v == 0:
                    break
        if bv and self.cfg.dwell_fallback in ("park", "free") and pos is not None:
            alt = self._dwell_park_fallback(pos, occ)
            if alt is not None:
                self.stats["dwell_fb"] += 1
                return alt, 0
            self.stats["dwell_fb_fail"] += 1
        if bv:
            self.stats["dwell_blocked"] += 1
        return best, bv

    def _dwell_park_fallback(self, pos, occ):
        """第一个 TER 违规为 0 的候选格；找不到则返回 None。

        park：只扫 non-task endpoint（最多 64 个最近的）—— 保 vid-safe，
              但"不存在安全的 non-task endpoint"并不能推出"不存在安全格"，
              因此 Lemma 只是**单向**的（可函数名 legacy 保留）。
        free：先按距离扫完全部 non-task endpoint，再按距离扫其余候选格，
              即在**整个候选集**上做完备搜索。于是"扫描失败"等价于
              "候选集中不存在 v=0 的格"，Lemma 4 的逆否命题也随之成立。
        """
        limited = self.cfg.dwell_fallback != "free"
        seen = set()
        pk = [c for c in self.wh.parking_cells if c not in occ and c != pos]
        pk.sort(key=lambda c: self.wh.dist(pos, c))
        order = []
        for c in pk:
            if c in self.rp._cellset:
                order.append(c)
                seen.add(c)
        if not limited:
            rest = [c for c in self.rp.cells
                    if c not in occ and c != pos and c not in seen]
            rest.sort(key=lambda c: self.wh.dist(pos, c))
            order.extend(rest)
        else:
            order = order[:64]
        for c in order:
            if self._dwell_cost(set(occ) | {c}) == 0:
                return c
        return None

    def _yield_dwell(self, va, t):
        """让行：把一个阻塞他人的空闲 agent 迁到另一个安全泊位。

        对应 MAPFUA 的设定——unassigned agents 被允许移动以给 assigned agents 让路。
        """
        pos = va["pos"]
        occ = self._occupied_cells(exclude=va["id"])
        cands = self.rp.cells if self.cfg.dwell_safety else None
        if cands is not None:
            # TER-Safety 优先：近邻里取违规最小者（同样必须是-agent 的 requirement）
            ranked = sorted([c for c in self.rp.cells
                             if c not in occ and c != pos],
                            key=lambda c: self.wh.dist(pos, c))[:10]
            best, bv = None, None
            for c in ranked:
                v = self._dwell_cost(set(occ) | {c})
                if bv is None or v < bv:
                    best, bv = c, v
                    if v == 0:
                        break
            cands = [best] if best is not None else []
        else:
            cands = sorted([c for c in self.rp.cells
                            if c not in occ and c != pos],
                           key=lambda c: self.wh.dist(pos, c))[:10]
        for c in cands:
            try:
                pth = st_astar(self.wh, self.rt, pos, c, t, va["id"])
            except PlanFailure:
                continue
            t_arr = t + len(pth) - 1
            if not self.rt.future_clear(c[0], c[1], t_arr, va["id"]):
                continue
            self.rt.remove_agent(va["id"])
            self.rt.add_path(va["id"], pth, t)
            va.update(path=pth, t0=t, idx=0, state="park", task=None,
                      batch=None, rest_t=t)
            return True
        return False

    # ---------------- 经验载货交通场 ----------------
    def _update_haul_field(self):
        """维护「已进入送货段（picked）的 agent」占用格子的 EMA，归一化后写回 rp.flow。

        每 tick 只对现存条目做衰减 + 对当 tick 出现的格子做上升，
        规模 ~O(被占用格数)，与地图尺寸无关。
        """
        a = 1.0 / max(1.0, float(self.cfg.flow_tau))
        ema = self._haul_ema
        seen = set()
        for ag in self.agents:
            if ag["state"] == "busy" and ag.get("picked"):
                p = ag["pos"]
                seen.add(p)
                cur = ema.get(p, 0.0)
                ema[p] = cur + a * (1.0 - cur)
        dead = []
        for c, v in ema.items():
            if c not in seen:
                v *= (1.0 - a)
                if v < 1e-6:
                    dead.append(c)
                else:
                    ema[c] = v
        for c in dead:
            ema.pop(c, None)
        # haul_probe：只做**统计**，不写回 rp.flow（写回会把离线结构先验冲掉）。
        # 打开它可以把"实际载货交通"落盘，用来算各版 flow 定义与真实流量的相关性。
        if self.cfg.flow_mode not in ("emp", "both"):
            return
        f = self.rp.flow
        if f is None:
            return
        mx = max(ema.values()) if ema else 0.0
        struct = self._struct_flow or {}
        for c in f:
            f[c] = 0.0
        if mx <= 0:
            return
        mode = self.cfg.flow_mode
        for c, v in ema.items():
            if c not in f:
                continue
            f[c] = v / mx if mode == "emp" else v / mx + struct.get(c, 0.0)

    def _rest_decide(self, a, t):
        """包装：计时（用于报告 runtime per decision）。"""
        t0 = time.perf_counter()
        r = self._rest_decide_inner(a, t)
        s = self.stats
        s["decide_cnt"] += 1
        s["decide_ns"] += int((time.perf_counter() - t0) * 1e9)
        return r

    def _rest_decide_inner(self, a, t):
        cfg = self.cfg
        pos = a["pos"]
        occ = self._occupied_cells(exclude=a["id"])
        p = self._rest_prob()
        others = self._idle_cells(exclude=a["id"]) \
            if self.rp.lp is not None else None
        if cfg.dwell_safety:
            probe = max(3, int(cfg.dwell_probe))
            ranked = self.rp.rank(pos, p, occ, agent_id=a["id"],
                                  exclude_self=pos, limit=probe,
                                  others=others)
            target, vnew = self._dwell_pick(occ, ranked, pos=pos)
            cost_pos = None
            if target is not None and pos in self.rp.cells:
                cost_pos = self._dwell_cost(set(occ) | {pos})
        else:
            target = self.rp.choose(pos, p, occ, agent_id=a["id"],
                                    exclude_self=pos, others=others)
            cost_pos = None

        stay = (target is None or target == pos)
        if not stay and self.cfg.move_thresh > 0 and pos in self.rp.cells:
            # 迟滞：收益不足以补偿切换成本时维持原位
            cur = self.rp.score(pos, pos, p)
            new = self.rp.score(target, pos, p)
            if new >= cur - self.cfg.move_thresh:
                stay = True
        if stay and cost_pos is not None and target is not None \
                and cost_pos > 0 and cost_pos > (vnew or 0):
            # 原地会让 TER 不变量恶化，而候选更优 —— 此时不得赖着不走
            stay = False
        if stay:
            target = pos if pos in self.rp.cells else None
            if target is not None:
                self._reg(a, [target], t, True)
                a.update(path=[target], t0=t, idx=0, state="park",
                         task=None, batch=None, rest_target=target,
                         picked=False)
                self._mark_trace(a, target)
                return True
            return self._plan_to_parking(a, t)

        # ACO 局部更新（ACS）：生态位分化，避免所有空闲 agent 挤到同一热点
        if self.df.xi > 0:
            r_target = self.wh.region(*target)
            self.df.local_decay(r_target)

        return self._plan_to_cell(a, target, t)

    # ---------- RL 信用分配 ----------
    def _mark_trace(self, a, target):
        """预定位决策建立资格迹：回报要等下一次该 agent 被派单才兑现，
        且延迟是随机的，故用 TD(λ) 资格迹回溯credit。"""
        if self.cfg.rl_mode != "adv":
            return
        r = self.wh.region(*target)
        a["elig"] = self.df.new_trace(r)

    def _record_response(self, a, d0):
        """任务派给该 agent 时结算回报：(基线响应时间 − 实际响应时间)。

        这是 REINFORCE with baseline：只在"比平均好"时强化。
        负值会削弱对应区域，从而实现多 agent 的竞争性信用分配。
        """
        cfg = self.cfg
        st = self.stats
        # 学习类基线：驻留决策的回报在这里兑现（响应距离越小越好）
        if self.rp.lp is not None:
            self.rp.learn_observe(a["id"], d0)
        if cfg.rl_mode != "adv":
            return
        if st.get("resp_ema", -1.0) < 0:
            st["resp_ema"] = float(d0)
        base = st["resp_ema"]
        adv = (base - d0) / max(base, 1.0)      # 无量纲优势
        st["resp_ema"] += 0.02 * (d0 - base)
        st["resp_cnt"] = st.get("resp_cnt", 0) + 1
        el = a.get("elig")
        if el:
            self.df.reinforce(el, adv, cfg.rl_eta)
            a["elig"] = None

    def _plan_to_cell(self, a, cell, t):
        try:
            pth = st_astar(self.wh, self.rt, a["pos"], cell, t, a["id"])
        except PlanFailure:
            # 占位失败：退化为最近可用 non-task endpoint
            return self._plan_to_parking(a, t)
        # 永久停留必须保证该格在抵达之后没有任何他人的预约
        t_arr = t + len(pth) - 1
        if not self.rt.future_clear(cell[0], cell[1], t_arr, a["id"]):
            self.stats["rest_blocked"] = self.stats.get("rest_blocked", 0) + 1
            return self._plan_to_parking(a, t)
        pth = self._reg(a, pth, t, True)
        a.update(path=pth, t0=t, idx=0, state="park", task=None, batch=None,
                 rest_target=cell, picked=False)
        self.stats["rest_moves"] += 1
        self._mark_trace(a, cell)
        return True

    def _plan_task(self, a, tk, t, pickup=None, delivery=None):
        """规划一个任务，失败时启用 liveness 让行机制（若开启）。"""
        if self.cfg.preempt <= 0:
            return self._plan_task_once(a, tk, t, pickup, delivery)
        self.stats["preempt_calls"] += 1
        for _ in range(self.cfg.preempt + 1):
            if self._plan_task_once(a, tk, t, pickup, delivery):
                return True
            blk = getattr(self, "_last_blockers", None) or {}
            if not blk:
                self.stats["plan_fail"] += 1
                a["stuck"] += 1
                return False
            acted = False
            for vid, _n in sorted(blk.items(), key=lambda kv: -kv[1]):
                if vid == a["id"] or not (0 <= vid < len(self.agents)):
                    continue
                va = self.agents[vid]
                if va["state"] in ("idle", "park") and self._yield_dwell(va, t):
                    self.stats["preempt_yields"] += 1
                    acted = True
                    break
            if not acted:
                break
        self.stats["plan_fail"] += 1
        a["stuck"] += 1
        return False

    def _plan_task_once(self, a, tk, t, pickup=None, delivery=None):
        pk = pickup or tk.get("pickup") or tk["pickups"][0]
        dl = delivery or tk.get("delivery") or tk["deliveries"][0]
        self._last_blockers = {}
        # 异质节拍：慢车先在原地多停 w 拍再出发。原地这段时间被本 agent 自己的
        # permanent reservation 覆盖（泊位格从 t_start 起归它所有），因此**不会**
        # 引入任何冲突；真正改变的是响应时间分布，也正是要测量的量。
        w = int(a.get("lat", 0) or 0)
        try:
            path, t_pick = plan_task(self.wh, self.rt, a["pos"], pk, dl,
                                     t + w, a["id"], blockers=self._last_blockers)
        except PlanFailure:
            self.stats["plan_attempt_fail"] += 1
            return False
        if w > 0:
            path = [a["pos"]] * w + list(path)

        # ---- Token Passing 的 dummy path（Ma et al. 2017）----
        # 标准 TP 要求一次规划出 start→pickup→delivery→non-task endpoint 的
        # 完整路径，且末端 permanent reservation，以此保证 well-formed。
        permanent = False
        del_idx = None
        if self.cfg.tp_dummy:
            t_dl = t + len(path) - 1
            occ = self._occupied_cells(exclude=a["id"])
            cells = sorted([c for c in self.wh.parking_cells if c not in occ],
                           key=lambda c: self.wh.dist(dl, c))[:6]
            for pc in cells:
                try:
                    p3 = st_astar(self.wh, self.rt, dl, pc, t_dl, a["id"])
                except PlanFailure:
                    continue
                if not self.rt.future_clear(pc[0], pc[1],
                                            t_dl + len(p3) - 1, a["id"]):
                    continue
                del_idx = len(path) - 1
                path = path + p3[1:]
                permanent = True
                break

        path = self._reg(a, path, t, permanent)
        tk["pickup"] = pk
        tk["delivery"] = dl
        free = self.wh.dist(a["pos"], pk)
        a.update(path=path, t0=t, idx=0, state="busy", task=tk, stuck=0, batch=None,
                 assign_t=t, assign_r=self.wh.region(*a["pos"]),
                 pick_t=t_pick, pick_free=free if free > 0 else 1,
                 del_idx=del_idx, picked=False, rest_target=None)
        tk["agent"] = a["id"]
        tk["assign_t"] = t          # L1 段终点
        tk["pick_t"] = t_pick       # L2 段终点（实际抵达时会被覆盖）
        self.stats["resp_dist"] += max(free, 0)
        self.stats["assign_cnt"] += 1
        self._record_response(a, free if free > 0 else 0)
        return True

    def _plan_batch(self, a, seq, t):
        """多任务批次规划：start → p_1 → … → p_c → d_1 → … → d_c。

        逐段时空 A* 串联。任一段失败即整体回滚（由调用方降级处理）。
        """
        picks = [it[1] for it in seq]
        dels = [it[2] for it in seq]
        self.rt.remove_agent(a["id"])
        cur = a["pos"]
        tt = t
        path = [cur]
        pick_marks, del_marks = [], []
        try:
            for p in picks:
                seg = st_astar(self.wh, self.rt, cur, p, tt, a["id"])
                tt += len(seg) - 1
                path += seg[1:]
                pick_marks.append(len(path) - 1)
                cur = p
            for d in dels:
                seg = st_astar(self.wh, self.rt, cur, d, tt, a["id"])
                tt += len(seg) - 1
                path += seg[1:]
                del_marks.append(len(path) - 1)
                cur = d
        except PlanFailure:
            self.rt.remove_agent(a["id"])
            self.stats["plan_fail"] += 1
            return False

        self.rt.remove_agent(a["id"])
        self.rt.add_path(a["id"], path, t, permanent=False)
        # 首个任务「单独执行」的自由流估计，用于计算边际行程
        ff = self.wh.dist(a["pos"], picks[0])
        fd = self.wh.dist(picks[0], dels[0])
        first_leg = (ff if ff > 0 else 0) + (fd if fd > 0 else 0)
        geom = _seq_length(self.wh, a["pos"], picks, dels)
        if geom > 0:
            self.stats["seq_geom"] += geom
        r_seq = [self.wh.region(*a["pos"])]
        r_seq += [self.wh.region(*p) for p in picks]
        r_seq += [self.wh.region(*d) for d in dels]
        a.update(path=path, t0=t, idx=0, state="busy", task=seq[0][0], stuck=0,
                 batch=dict(tasks=[it[0] for it in seq], picks=picks, dels=dels,
                            pick_marks=pick_marks, del_marks=del_marks,
                            n_picked=0, n_delivered=0,
                            start_t=t, start_pos=a["pos"], region_seq=r_seq,
                            first_leg=first_leg))
        for it in seq:
            it[0]["pickup"] = it[1]
            it[0]["delivery"] = it[2]
        return True

    def _observe_trips(self, t):
        """（Phase 1）到达 pickup 时产生一次拥堵系数观测。"""
        for a in self.agents:
            if a["state"] != "busy" or a.get("pick_t") is None:
                continue
            if a["pick_t"] != t:
                continue
            o = (t - a["assign_t"]) / max(a["pick_free"], 1)
            r_end = self.wh.region(*a["pos"])
            self.ph.observe_trip([a["assign_r"], r_end], o)
            a["pick_t"] = None

    # ---------------- 执行 ----------------
    def _execute(self):
        for a in self.agents:
            a["prev"] = a["pos"]
            if a["path"] is None:
                continue
            if a["idx"] < len(a["path"]) - 1:
                a["idx"] += 1
                a["pos"] = a["path"][a["idx"]]
                waited = a["path"][a["idx"]] == a["path"][a["idx"] - 1]
                r = self.wh.region(*a["pos"])
                self.ph.observe_wait(r, waited)
                self.stats["steps"] += 1
                if waited:
                    self.stats["waits"] += 1

    def _rhcr_seal(self, t):
        """RHCR 下为每个「当前格在 t+1 没有预约」的 agent 补一个周期的驻留预约。

        RHCR 没有 permanent reservation，因此一旦某个 agent 的窗口用尽（任务
        完成、泊位抵达、路径被 _check_completion 折叠），它的格子就不再受保护，
        后续规划的 agent 会被允许穿过去 —— 执行时就产生真实碰撞。本方法在
        每个 tick 末尾补齐，等价于 Li et al. (2021) 中「窗口必须始终覆盖下一
        步」的不变量，从而保证 RHCR 基线也是 collision-free 的。
        """
        if self.cfg.planner_mode != "rhcr":
            return
        for a in self.agents:
            pos = a["pos"]
            if pos is None:
                continue
            p = a.get("path")
            if p is not None and a.get("idx", 0) < len(p) - 1:
                continue            # 仍有后续步 -> 未来预约已存在，不可覆盖
            owner = self.rt.cell.get((pos[0], pos[1], t + 1))
            if owner == a["id"]:
                continue            # 窗口中已覆盖下一步
            if owner is not None:
                continue            # 已被他人预约（极罕见），本 tick 不抢占
            a["path"] = self._reg(a, [pos], t, False)
            a["idx"] = 0

    def _check_completion(self, t):
        # ---- pickup 到位标记（RHCR 窗口重规划需要区分"去取货"还是"去送货"）----
        for a in self.agents:
            if a["state"] != "busy" or a["batch"] is not None:
                continue
            tk = a["task"]
            if tk is not None and not a.get("picked"):
                if tk.get("pickup") is not None and a["pos"] == tk["pickup"]:
                    a["picked"] = True
                    tk["pick_t"] = t            # L2 段终点（实际抵达）
        # ---- dummy path 尾段：任务已在 N 步前完成，agent 仍在走向泊位 ----
        # Ma et al. (2017) 的 dummy path 语义：任务完成 ≠ 路径结束。若不在这里
        # 把它转回 idle，agent 会永远停留在 busy 而无法再被派单 -> 全系统饿死。
        for a in self.agents:
            if a["state"] != "busy" or a["batch"] is not None \
                    or a["task"] is not None or a["path"] is None:
                continue
            if a["idx"] >= len(a["path"]) - 1:
                a.update(state="idle", del_idx=None, stuck=0)

        # ---- Phase 1 单任务路径 ----
        for a in self.agents:
            if a["path"] is None or a["idx"] < len(a["path"]) - 1:
                # 走在 dummy path 上（尚未抵达泊位）时也可能刚好完成 delivery
                if a["path"] is None or a.get("del_idx") is None:
                    continue
            if a["state"] == "busy" and a["task"] is not None and a["batch"] is None:
                tk = a["task"]
                di = a.get("del_idx")
                at_end = a["idx"] >= len(a["path"]) - 1
                done = ((a["idx"] >= di) if di is not None
                        else (at_end and a["pos"] == tk["delivery"]))
                if done:
                    self._finish_task(a, tk, t)
                    if at_end:
                        self.rt.remove_agent(a["id"])
                        a.update(path=None, task=None, state="idle", stuck=0,
                                 del_idx=None)
                    else:
                        a.update(task=None)   # 继续走 dummy path 到泊位
            elif a["state"] == "park":
                a["path"] = [a["pos"]]
                a["idx"] = 0

        # ---- Phase 2 批次路径 ----
        for a in self.agents:
            b = a.get("batch")
            if b is None or a["path"] is None:
                continue
            while b["n_delivered"] < len(b["tasks"]) \
                    and a["idx"] >= b["del_marks"][b["n_delivered"]]:
                tk = b["tasks"][b["n_delivered"]]
                self._finish_task(a, tk, t)
                b["n_delivered"] += 1
            if b["n_delivered"] >= len(b["tasks"]):
                self._finish_batch(a, b, t)
                self.rt.remove_agent(a["id"])
                a.update(path=None, task=None, state="idle", stuck=0, batch=None)

    def _finish_task(self, a, tk, t):
        self.stats["completed"] += 1
        self.stats["service"].append(t - tk["release"])
        if t >= self.cfg.warmup:
            self.stats["completed_post"] += 1
            self.stats["service_post"].append(t - tk["release"])
            self._legs(tk, t)
        try:
            self.pending.remove(tk)
        except ValueError:
            pass

    def _legs(self, tk, t):
        """把一个任务的驻留时间拆成三段，用于定位延迟到底花在哪。

            L1 assign   release -> 被派给某个 agent（等待 + 无可用 agent）
            L2 approach 派出 -> 抵达 pickup（即 resp_dist 对应的实际行程）
            L3 haul     pickup -> delivery
        """
        ta = tk.get("assign_t")
        tp = tk.get("pick_t")
        if ta is None:
            return
        s = self.stats
        s["n_leg"] += 1
        s["leg_assign"] += ta - tk["release"]
        s["leg_approach"] += (tp if tp is not None else t) - ta
        s["leg_haul"] += t - (tp if tp is not None else t)

    def _finish_batch(self, a, b, t):
        """批次完成：计算边际行程并沉积到转移信息素场。"""
        c = len(b["tasks"])
        L = t - b["start_t"]
        margin = L - b["first_leg"]
        if margin < 0:
            margin = 0
        per_task = margin / max(c - 1, 1) if c > 1 else margin
        self.stats["batches"] += 1
        self.stats["batch_sizes"].append(c)
        self.stats["travel"] += L
        if c > 1:
            self.stats["margin"].append(per_task)
            self.stats["seq_len"] += L
            self.stats["first_leg"] += b["first_leg"]
        self.ph2.observe_batch(b["region_seq"], per_task)

    def _detect_conflicts(self):
        seen = {}
        n = 0
        for a in self.agents:
            p = a["pos"]
            if p in seen:
                n += 1
            seen[p] = a["id"]
        moves = {}
        for a in self.agents:
            moves[a["prev"]] = a["pos"]
        sw = 0
        for a in self.agents:
            p, q = a["prev"], a["pos"]
            if p != q and moves.get(q) == p:
                sw += 1
        self.stats["conflicts"] += n + sw // 2

    # ---------------- 主循环 ----------------
    def run(self, verbose=False):
        cfg = self.cfg
        self._t_start = time.time()
        self._build_dyn_obs()
        self._replenish(0)
        self._assign_and_plan(0)
        for t in range(cfg.T):
            self._maybe_drift(t)
            if t % cfg.window == 0:
                self.ph.tick(t)
                self.ph2.tick(t)
                self.df.tick(t)
            if cfg.comm_lag > 0:
                self._prob_hist.append(self.df.prob())
            if self._obs_pos:
                self._dyn_obs_step(t)
            if cfg.flow_mode in ("emp", "both") or cfg.haul_probe:
                self._update_haul_field()
            if cfg.rl_mode == "adv" and cfg.elig > 0:
                for a in self.agents:
                    if a.get("elig"):
                        self.df.decay_traces(a["elig"], cfg.elig ** cfg.window)
            self._execute()
            tc = t + 1
            self._observe_trips(tc)
            self._check_completion(tc)
            self._replenish(tc)
            self._rhcr_seal(tc)     # 补驻留预约：保证 RHCR 窗口始终覆盖下一步
            self._assign_and_plan(tc)
            if cfg.planner_mode == "rhcr":
                self._rhcr_tick(tc)
                self._rhcr_seal(tc)  # 规划可能释放了他人预约，末尾再补一次
            self._cur_t = tc
            self._detect_conflicts()
            # ---- Phase 3 过程统计 ----
            self.stats["pending_sum"] += len(self.pending)
            self.stats["idle_steps"] += sum(
                1 for a in self.agents if a["state"] in ("idle", "park"))
            if (t + 1) % cfg.series_step == 0:
                self.stats["series"].append((tc, self.stats["completed"]))
            if self._trace_cb is not None:
                self._trace_cb(self, tc)
            if self.rp.lp is not None:
                q = cfg.T // 4
                if q > 0 and (t + 1) % q == 0:
                    # 学习基线的收敛轨迹：回报基线 = 已实现的平均响应距离
                    self.stats["learn_curve"].append(
                        self.rp.lp.base if self.rp.lp.base is not None else -1.0)

        s = self.stats
        post = max(cfg.T - cfg.warmup, 1)
        tp_all = s["completed"] / max(cfg.T, 1) * 1000.0
        tp_post = s["completed_post"] / post * 1000.0
        mg = s["margin"]
        return dict(
            throughput=tp_all,
            throughput_post=tp_post,
            completed=s["completed"],
            completed_post=s["completed_post"],
            service_time=(sum(s["service"]) / len(s["service"])) if s["service"] else -1.0,
            service_post=(sum(s["service_post"]) / len(s["service_post"]))
            if s["service_post"] else -1.0,
            wait_ratio=(s["waits"] / s["steps"]) if s["steps"] else 0.0,
            leg_assign=(s["leg_assign"] / s["n_leg"]) if s["n_leg"] else -1.0,
            leg_approach=(s["leg_approach"] / s["n_leg"]) if s["n_leg"] else -1.0,
            leg_haul=(s["leg_haul"] / s["n_leg"]) if s["n_leg"] else -1.0,
            no_idle=s["no_idle"],
            moving_idle=s.get("moving_idle", 0),
            assign_moving=int(cfg.assign_moving),
            detour=self.ph.mean_trip_obs(),
            conflicts=s["conflicts"],
            plan_fail=s["plan_fail"],
            plan_attempt_fail=s["plan_attempt_fail"],
            preempt_calls=s["preempt_calls"],
            preempt_yields=s["preempt_yields"],
            dwell_blocked=s["dwell_blocked"],
            dwell_checks=s["dwell_checks"],
            dwell_viol=s["dwell_viol"],
            dwell_fb=s["dwell_fb"],
            dwell_fb_fail=s["dwell_fb_fail"],
            learn_upd=(self.rp.lp.m.n_upd if self.rp.lp is not None else 0),
            learn_nfb=(self.rp.lp.n_fb if self.rp.lp is not None else 0),
            learn_curve="|".join("%.3f" % x for x in s["learn_curve"]),
            haul_load=";".join("%d_%d_%.4f" % (c[0], c[1], v)
                               for c, v in sorted(self._haul_ema.items())),
            decide_cnt=s["decide_cnt"],
            decide_ns=s["decide_ns"],
            # 空驶里程：派单瞬间 agent 到 pickup endpoint 的最短距离
            resp_dist=(s["resp_dist"] / s["assign_cnt"]) if s["assign_cnt"] else -1.0,
            idle_ratio=(s["idle_steps"] / max(cfg.T * cfg.n_agents, 1)),
            wall_sec=time.time() - self._t_start,
            series=s["series"],            tau_hist=self.ph.history,
            tau_final=self.ph.snapshot(),
            oracle_tau=self.ph.oracle_tau(),
            # MT 指标
            batches=s["batches"],
            avg_batch=(sum(s["batch_sizes"]) / len(s["batch_sizes"]))
            if s["batch_sizes"] else 0.0,
            margin_per_task=(sum(mg) / len(mg)) if mg else -1.0,
            travel_per_task=(s["travel"] / s["completed"]) if s["completed"] else -1.0,
            steps_per_task=(s["steps"] / s["completed"]) if s["completed"] else -1.0,
            seq_eff=(s["first_leg"] / s["seq_len"]) if s["seq_len"] else -1.0,
            geom_per_batch=(s["seq_geom"] / s["batches"]) if s["batches"] else -1.0,
            tau2_spread=self.ph2.spread(),
            tau2_final=self.ph2.snapshot(),
            tau2_oracle=self.ph2.oracle_tau(),
            mean_pending=s["pending_sum"] / max(cfg.T, 1),
            rest_moves=s["rest_moves"],
            arrivals=s["arrivals"],
            demand_spread=self.df.spread(),
            demand_final=self.df.snapshot(),
            demand_prob=tuple(self.df.prob()),
            demand_mult=tuple(self.df.mult),
            demand_oracle=tuple(self.df.oracle_prob()),
            demand_cum=tuple(self.df.cum),
        )
