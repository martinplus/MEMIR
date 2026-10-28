"""论文用统计量：配对比较的 bootstrap 置信区间、Wilcoxon 符号秩、Holm 校正。

零第三方依赖（scipy / numpy 均不可用），全部自己实现：
  - 配对 bootstrap：对同一 seed 的两个方法做差值，重采样 ->
    不假设正态分布，也不需要查表
  - Wilcoxon 符号秩：n>=10 时用正态近似（含并列修正与连续性校正）
  - Holm-Bonferroni：多重比较校正
"""
import math
import random


def mean_ci(x, n_boot=10000, seed=0):
    """单样本均值的 bootstrap 95% CI。"""
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    m = sum(x) / n
    bs = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += x[rng.randrange(n)]
        bs.append(s / n)
    bs.sort()
    lo = bs[int(0.025 * n_boot)]
    hi = bs[min(int(0.975 * n_boot), n_boot - 1)]
    return m, lo, hi


def paired_diff(x, y):
    """配对差值序列 d_i = x_i - y_i（x = 待检验方法，y = 基线）。"""
    return [a - b for a, b in zip(x, y)]


def paired_bootstrap(d, n_boot=10000, seed=0):
    """配对差值的均值点估计与 95% CI + P(d<0) 的经验概率。"""
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = random.Random(seed)
    m = sum(d) / n
    bs = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        bs.append(s / n)
    bs.sort()
    lo = bs[int(0.025 * n_boot)]
    hi = bs[min(int(0.975 * n_boot), n_boot - 1)]
    pneg = sum(1 for b in bs if b >= 0) / float(n_boot)
    return m, lo, hi, min(pneg, 1.0 - pneg) * 2.0


def _norm_sf(z):
    """标准正态上尾概率（Abramowitz-Stegun 7.1.26 近似，|err|<7.5e-8）。"""
    if abs(z) >= 8.0:
        return 0.0 if z > 0 else 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
             - 0.356563782) * t + 0.319381530) * t
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    p = pdf * poly
    return p if z > 0 else 1.0 - p


def _signed_ranks(nz):
    """并列秩（同绝对值取平均秩），返回按原顺序排列的秩向量。"""
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks, order


def wilcoxon_exact(d, max_n=16):
    """Wilcoxon 符号秩的**精确**置换检验（双侧）。

    正态近似在小样本下的粒度是致命的：n=10 时即使十个差值同号，连续性校正
    后的近似 p 也只能到 0.0059，于是任何 m>8 的 Holm 校正族都机械地不可
    能显著。精确检验在 n=10 时的最小双侧 p 是 2/2^10 = 0.00195（n=3 时
    为 0.25，这才是"三 seed 判不出显著"的真正来源）。n<=max_n 时走精确
    枚举（2^16 次，毫秒级），以上退回正态近似。
    """
    nz = [v for v in d if v != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0, 0.0, 1.0
    if n > max_n:
        return None
    ranks, _order = _signed_ranks(nz)
    obs = sum(r for r, x in zip(ranks, nz) if x > 0)
    total = n * (n + 1) / 2.0
    lo = min(obs, total - obs)
    cnt = 0
    tot = 1 << n
    for mask in range(tot):
        s = 0.0
        mm = mask
        for i in range(n):
            if mm & 1:
                s += ranks[i]
            mm >>= 1
        if s <= lo + 1e-9 or s >= total - lo - 1e-9:
            cnt += 1
    p = cnt / float(tot)
    mu = total / 2.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (obs - mu) / math.sqrt(var) if var > 0 else 0.0
    return obs, z, min(1.0, p)


def wilcoxon_signed_rank(d):
    """Wilcoxon 符号秩检验（双侧）。返回 (W+, z, p)。

    n<=16 用精确置换分布；更大的样本用带并列修正与连续性校正的正态近似。
    """
    ex = wilcoxon_exact(d)
    if ex is not None:
        return ex
    nz = [v for v in d if v != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0, 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    # 并列秩：同绝对值者取平均秩
    ranks = [0.0] * n
    i = 0
    r = 1
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        if t > 1:
            tie_term += (t ** 3 - t)
        i = j + 1
        r = j + 2
    wp = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return wp, 0.0, 1.0
    cc = -0.5 if (wp - mu) > 0 else 0.5
    z = (wp - mu + cc) / math.sqrt(var)
    return wp, z, min(1.0, 2.0 * _norm_sf(abs(z)))


def _betacf(a, b, x, itmax=300, eps=3e-16):
    """连分数展开（Lentz 算法），Numerical Recipes 的 betacf。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def student_t_sf2(t, df):
    """Student t 的双侧 p 值：P(|T_df| >= |t|)。"""
    if df <= 0:
        return 1.0
    if t == 0:
        return 1.0
    x = df / (df + t * t)
    return betainc(0.5 * df, 0.5, x)


def paired_t(d):
    """配对 t 检验（双侧）。返回 (t, df, p, mean, sd)。

    ★ 为什么需要它：Wilcoxon 符号秩在 n 很小时**数学上不可能**显著 ——
      精确检验下 n=3 的最小双侧 p = 2/2^3 = 0.25，n=5 时 = 2/32 = 0.0625，
      均 > 0.05；本文实现用正态近似，n=3 时最小 p ≈ 0.18，同样不可能。
      因此凡是 n_pairs < 6（如尺度实验 O_scale 的 3 seeds）必须改用配对 t，
      并在表注中说明用的是哪个统计量。
    """
    n = len(d)
    if n < 2:
        return 0.0, 0, 1.0, (d[0] if n else 0.0), 0.0
    m = sum(d) / n
    v = sum((x - m) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(v)
    if sd == 0:
        return (float("inf") if m != 0 else 0.0), n - 1, 0.0, m, 0.0
    t = m / (sd / math.sqrt(n))
    return t, n - 1, min(1.0, student_t_sf2(t, n - 1)), m, sd


def holm(pvals):
    """Holm-Bonferroni 逐步校正，返回与输入同序的调整后 p 值。"""
    m = len(pvals)
    idx = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    prev = 0.0
    for k, i in enumerate(idx):
        v = (m - k) * pvals[i]
        prev = max(prev, v)
        out[i] = min(1.0, prev)
    return out


def cohens_dz(d):
    """配对效应量 Cohen's d_z（差值均值 / 差值标准差）。"""
    n = len(d)
    if n < 2:
        return 0.0
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / (n - 1)
    sd = math.sqrt(var)
    return m / sd if sd > 1e-12 else 0.0


def pct(a, b):
    """相对基线的百分比变化（负 = 改善）。"""
    if b == 0:
        return float("nan")
    return (a - b) / b * 100.0
