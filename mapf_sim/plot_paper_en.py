# -*- coding: utf-8 -*-
"""Generate all figures for the MEMIR manuscript as vector PDF (+ SVG).

All data are read from the aggregated result tables in results_final/tables.md
and from results_final/figs/fig5_dwell_W1_24.json (dwell traces).
Every label is English. Fonts are embedded in the PDF, so the output is a true
vector figure usable directly by pdflatex.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style ----
CAND = ["Arial", "Helvetica", "DejaVu Sans"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
FAM = next((c for c in CAND if c in _avail), "DejaVu Sans")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [FAM]
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.size"] = 8.0
rcParams["axes.labelsize"] = 8.5
rcParams["axes.titlesize"] = 8.5
rcParams["xtick.labelsize"] = 7.5
rcParams["ytick.labelsize"] = 7.5
rcParams["legend.fontsize"] = 7.0
rcParams["axes.linewidth"] = 0.6
rcParams["lines.linewidth"] = 1.3
rcParams["grid.linewidth"] = 0.4
rcParams["grid.color"] = "#d7dce3"
rcParams["axes.edgecolor"] = "#5b6570"
rcParams["xtick.major.width"] = 0.6
rcParams["ytick.major.width"] = 0.6
rcParams["svg.fonttype"] = "none"      # keep text editable / searchable
rcParams["pdf.fonttype"] = 42          # embed TrueType

# colour-blind safe categorical palette (Okabe-Ito)
COL = {
    "nearest": "#999999",
    "random": "#bbbbbb",
    "static": "#0072B2",
    "memir0": "#D55E00",
    "memir": "#009E73",
    "oracle": "#CC79A7",
    "oracle_f": "#E69F00",
    "pmedian": "#56B4E9",
    "popsample": "#F0E442",
    "learn": "#882255",
}
HATCH_SIG = None
HATCH_NS = "///"
CMAP = plt.get_cmap("magma")

# Fleet size is encoded by one sequential ramp everywhere in the paper, so a
# given |A| keeps the same colour in every figure and the legends of adjacent
# panels can be read against one another.
FLEETCOL = {4: "#cfe3f2", 6: "#b3d3ea", 8: "#9dc6de", 10: "#7fb2d4",
            12: "#5f9dc9", 16: "#2b7bba", 24: "#1a4f7a", 32: "#0b3d6b",
            40: "#062647"}

SC = 2.66      # half text width (inches); MDPI \textwidth = 394pt = 5.48in
DC = 5.48      # full text width (inches)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def style(ax, ygrid=True, xgrid=False):
    ax.set_axisbelow(True)            # gridlines stay behind bars and points
    ax.grid(axis="y" if ygrid else "both", which="major", alpha=1.0)
    if xgrid:
        ax.grid(axis="x", which="major", alpha=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# =============================================================== Fig 1 ====
# Table A: fleet sweep on W1 (lambda = 0.08, Zipf alpha = 2.0, 10 seeds)
FLEET = [4, 6, 8, 10, 12, 16, 24, 32, 40]
TAB_A = {
    "nearest": [155.91, 66.49, 56.87, 55.46, 55.10, 54.89, 54.60, 54.25, 53.71],
    "random": [155.02, 68.75, 55.71, 51.32, 48.08, 44.32, 40.52, 38.72, 37.50],
    "static": [153.94, 68.53, 49.10, 46.07, 47.20, 40.19, 38.57, 35.94, 36.11],
    "memir0": [152.49, 51.81, 38.93, 36.41, 35.76, 36.04, 36.64, 37.12, 37.27],
    "memir": [152.39, 51.30, 38.44, 35.79, 34.84, 34.29, 34.01, 33.84, 33.66],
    "oracle": [152.49, 52.03, 39.11, 36.61, 36.36, 37.18, 37.48, 37.11, 36.34],
}
IDLE_MEMIR = [2.5, 42.1, 63.0, 72.0, 77.2, 83.2, 88.9, 91.7, 93.4]
IDLE_STATIC = [1.7, 27.7, 53.6, 64.3, 69.3, 80.4, 87.4, 91.2, 92.9]


def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, 2.55))
    ax = ax1
    for k in ["nearest", "random", "static", "memir0", "memir", "oracle"]:
        ls = "--" if k in ("memir0", "oracle_f") else "-"
        mk = "o" if k == "memir" else ("s" if k == "static" else None)
        if k == "memir":
            continue
        ax.plot(FLEET, TAB_A[k], ls, color=COL[k], marker=mk, ms=3.2,
                label=k)
    ax.plot(FLEET, TAB_A["memir"], "-", color=COL["memir"], marker="o", ms=3.6,
            lw=1.7, label="MEMIR (proposed)", zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(FLEET)
    ax.set_xticklabels([str(v) for v in FLEET])
    ax.set_xlabel("fleet size  $|A|$")
    ax.set_ylabel("service time  (timesteps, log scale)")
    style(ax)
    leg = ax.legend(frameon=False, ncol=2, loc="upper right",
                    columnspacing=0.8, handlelength=1.6)
    leg.get_texts()[-1].set_weight("bold")
    ax.axvspan(3.2, 4.8, color="#f2f4f7", zorder=0)
    ax.annotate("saturated", xy=(4, 200), ha="center", va="top",
                fontsize=6.5, color="#6b7280")
    ax.set_title("(a)  all policies, full sweep", loc="left", fontsize=8)

    ax = ax2
    gain = [(m / s - 1) * 100 for m, s in zip(TAB_A["memir"], TAB_A["static"])]
    gain0 = [(m / s - 1) * 100 for m, s in zip(TAB_A["memir0"],
                                               TAB_A["static"])]
    ax.plot(IDLE_STATIC, gain0, "--", color=COL["memir0"], marker="s", ms=3.4,
            label="MEMIR without $\\psi$")
    ax.plot(IDLE_STATIC, gain, "-", color=COL["memir"], marker="o", ms=3.6,
            lw=1.7, label="MEMIR", zorder=5)
    seq = [(x, y, l) for x, y, l in zip(IDLE_STATIC, gain, FLEET) if l != 32]
    for i, (x, y, lab) in enumerate(seq):
        dy = -10 if i % 2 == 0 else 9
        dx = 4
        if lab == 16:
            dx = -4
        elif lab == 40:
            dx = 4
        ax.annotate("$|A|{=}%d$" % lab, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6, color="#4b5563")
    ax.axhline(0, color="#9ca3af", lw=0.6)
    ax.set_xlabel("idle fraction of the fleet under MEMIR  (%)")
    ax.set_ylabel("change in service time\nvs. uniform-static  (%)")
    style(ax)
    ax.legend(frameon=False, loc="upper center", fontsize=6.8, ncol=2,
              handlelength=1.5, columnspacing=1.0)
    ax.set_title("(b)  gain vs. idle slack", loc="left", fontsize=8)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_fleet")


# =============================================================== Fig 2 ====
# Table A: three-leg decomposition  L1 assign / L2 approach / L3 haul
LEGS = {
    8: {"nearest": (1.17, 25.99, 29.72), "static": (1.53, 17.84, 29.73),
        "memir0": (0.65, 8.37, 29.92), "memir": (0.58, 8.11, 29.74)},
    16: {"nearest": (0.00, 25.18, 29.71), "static": (0.00, 10.24, 29.95),
         "memir0": (0.00, 4.38, 31.67), "memir": (0.00, 4.46, 29.83)},
    32: {"nearest": (0.00, 24.52, 29.73), "static": (0.00, 5.87, 30.06),
         "memir0": (0.00, 3.29, 33.83), "memir": (0.00, 3.62, 30.22)},
}
LEG_COL = ["#4477AA", "#EE6677", "#228833"]


def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(DC, 2.25), sharey=True)
    keys = ["nearest", "static", "memir0", "memir"]
    labels = ["nearest", "static", "MEMIR\nno $\\psi$", "MEMIR"]
    for ax, n in zip(axes, sorted(LEGS)):
        bottom = np.zeros(len(keys))
        for j, comp in enumerate(range(3)):
            vals = np.array([LEGS[n][k][comp] for k in keys])
            ax.bar(range(len(keys)), vals, bottom=bottom, width=0.62,
                   color=LEG_COL[j], edgecolor="white", linewidth=0.5,
                   label=["$T_1$ assign", "$T_2$ approach",
                          "$T_3$ haul"][j] if n == 8 else None)
            bottom += vals
        tot = bottom
        for i, v in enumerate(tot):
            ax.text(i, v + 0.7, "%.1f" % v, ha="center", fontsize=6.2,
                    color="#374151")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_title("(a)" if n == 8 else "(b)" if n == 16 else "(c)",
                     loc="left", fontsize=8)
        ax.text(0.98, 0.96, "$|A|=%d$" % n, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#374151")
        style(ax)
    axes[0].set_ylabel("service time  (timesteps)")
    fig.legend(frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), fontsize=7.5)
    axes[1].set_ylim(0, 62)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_legs")


# =============================================================== Fig 3 ====
# Table H: sensitivity to the haul-traffic prior weight psi
PSI = [0, 10, 20, 30, 40, 80, 160]
HPSI = {
    8: {"static": 48.65, "svc": [38.36, 37.93, 38.01, 37.92, 37.99, 40.62,
                                 40.61],
        "L3": [29.61, 29.50, 29.47, 29.45, 29.45, 29.49, 29.46]},
    16: {"static": 39.89, "svc": [35.59, 34.35, 34.17, 33.95, 34.08, 35.75,
                                  36.75],
         "L3": [31.33, 30.21, 29.95, 29.54, 29.50, 29.46, 29.50]},
    32: {"static": 35.66, "svc": [36.70, 34.64, 33.70, 33.60, 33.44, 34.27,
                                  35.62],
         "L3": [33.46, 31.29, 30.30, 29.98, 29.76, 29.56, 29.61]},
}


def fig3():
    """Sensitivity to the haul-traffic prior weight psi (Table H).

    Top row: service time. Bottom row: the haul leg T3, which falls
    monotonically and thereby trades against the rising response leg.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(DC, 3.55))
    xs = np.arange(len(PSI))
    psilab = ["%g" % v for v in PSI]

    # shared vertical ranges per row, with headroom so markers stay clear
    top = []
    bot = []
    for n in HPSI:
        d = HPSI[n]
        top += [min(d["svc"]), max(d["svc"]), d["static"]]
        bot += [min(d["L3"]), max(d["L3"])]
    pad = 0.06
    y0 = (min(top) - pad * (max(top) - min(top)),
          max(top) + pad * (max(top) - min(top)))
    y1 = (min(bot) - 0.22 * (max(bot) - min(bot)),
          max(bot) + 0.14 * (max(bot) - min(bot)))

    for j, n in enumerate(sorted(HPSI)):
        d = HPSI[n]

        ax = axes[0][j]
        ax.axhline(d["static"], color=COL["static"], ls=(0, (3.2, 2.0)),
                   lw=1.0, zorder=1)
        ax.plot(xs, d["svc"], "-o", color=COL["memir"], ms=4.0, lw=1.5,
                mfc="white", zorder=4)
        k = int(np.argmin(d["svc"]))
        ax.plot([xs[k]], [d["svc"][k]], marker="*", ms=13, color="#b8072b",
                zorder=5)
        style(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(psilab, fontsize=6.6)
        ax.set_xlim(-0.45, len(PSI) - 0.55)
        ax.set_ylim(*y0)
        ax.set_title("(%s)  service time,  $|A|=%d$" % ("abc"[j], n),
                     loc="left", fontsize=7.4, pad=4)

        ax = axes[1][j]
        ax.axhline(d["L3"][0], color="#9ca3af", ls=(0, (3.2, 2.0)), lw=0.9,
                   zorder=1)
        ax.plot(xs, d["L3"], "-s", color=LEG_COL[2], ms=3.8, lw=1.4,
                mfc="white", zorder=4)
        style(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(psilab, fontsize=6.6)
        ax.set_xlim(-0.45, len(PSI) - 0.55)
        ax.set_ylim(*y1)
        ax.set_title("(%s)  haul leg,  $|A|=%d$" % ("def"[j], n),
                     loc="left", fontsize=7.4, pad=4)

    axes[0][0].set_ylabel("service time")
    axes[1][0].set_ylabel("haul leg  $T_3$")

    h = [Line2D([], [], color=COL["memir"], marker="o", ms=4.0, mfc="white",
                lw=1.5, label="MEMIR, service time"),
         Line2D([], [], color=COL["static"], ls=(0, (3.2, 2.0)), lw=1.0,
                label="uniform-static reference"),
         Line2D([], [], color="#b8072b", marker="*", ms=11, lw=0,
                label="interior optimum"),
         Line2D([], [], color=LEG_COL[2], marker="s", ms=3.8, mfc="white",
                lw=1.4, label="haul leg $T_3$"),
         Line2D([], [], color="#9ca3af", ls=(0, (3.2, 2.0)), lw=0.9,
                label="$T_3$ at $\\psi=0$")]
    fig.legend(handles=h, frameon=False, ncol=3, fontsize=6.8,
               loc="lower center", bbox_to_anchor=(0.5, 0.165),
               handlelength=1.9, columnspacing=1.6)
    fig.supxlabel("traffic-avoidance weight  $\\psi$", fontsize=8.5, y=0.085)
    fig.subplots_adjust(left=0.072, right=0.985, top=0.905, bottom=0.335,
                        hspace=0.62, wspace=0.26)
    save(fig, "fig3_psi")


# =============================================================== Fig 4 ====
# Table E (spatial skew) and Table F (non-stationary demand)
ALPHA = [0.0, 1.0, 2.0, 3.0]
E_GAIN = {8: [-8.26, -12.88, -22.05, -27.88],
          16: [4.67, -1.80, -14.90, -21.37],
          16.1: [12.82, 0.41, -10.77, -13.84]}    # MEMIR without psi, |A|=16
REGIMES = ["stationary", "periodic drift", "abrupt shift"]
FDRIFT = {   # |A| = 16, service time
    "MEMIR (meta-adapt.)": [34.24, 31.04, 30.65],
    "MEMIR ($\\tau{=}100$)": [33.95, 30.95, 30.59],
    "MEMIR ($\\tau{=}2000$)": [34.00, 32.07, 34.58],
    "hindsight oracle": [34.04, 31.98, 35.85],
    "nearest (no memory)": [54.48, 48.14, 42.66],
}


def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, 2.55))
    ax = ax1
    ax.plot(ALPHA, E_GAIN[8], "-o", color=COL["memir"], ms=3.4, lw=1.6,
            label="MEMIR, $|A|=8$")
    ax.plot(ALPHA, E_GAIN[16], "-s", color=COL["static"], ms=3.4, lw=1.4,
            label="MEMIR, $|A|=16$")
    ax.plot(ALPHA, E_GAIN[16.1], "--^", color=COL["memir0"], ms=3.2, lw=1.1,
            label="MEMIR no $\\psi$, $|A|=16$")
    ax.axhline(0, color="#9ca3af", lw=0.6)
    for x in ALPHA:
        ax.axvline(x, color="#eef1f5", lw=0.6, zorder=0)
    ax.set_xlabel("spatial skew of demand  $\\alpha$  (Zipf exponent)")
    ax.set_ylabel("change in service time  vs. uniform-static  (%)")
    ax.set_xticks(ALPHA)
    style(ax)
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_title("(a)  arrival-field heterogeneity", loc="left",
                 fontsize=8)

    ax = ax2
    x = np.arange(len(REGIMES))
    cols = [COL["memir"], "#66c2a5", "#01665e", COL["oracle"], COL["nearest"]]
    mks = ["o", "s", "^", "D", "v"]
    for i, (k, v) in enumerate(FDRIFT.items()):
        ax.plot(x, v, "-", color=cols[i], marker=mks[i], ms=3.6, lw=1.4,
                label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES, fontsize=7.2)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("service time  (timesteps)")
    style(ax)
    ax.set_ylim(28.5, 60)
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left",
              columnspacing=0.8, handlelength=1.8)
    ax.set_title("(b)  non-stationary demand, $|A|=16$", loc="left",
                 fontsize=8)
    ax.annotate("", xy=(2.0, 35.85), xytext=(2.0, 30.65),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#b8072b"))
    ax.text(2.06, 33.2, "hindsight average\nis no longer an\nupper bound",
            fontsize=6.2, color="#b8072b", va="center")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_cond")


# =============================================================== Fig 5 ====
# Mechanism traces on W1, |A| = 24 (6 seeds)
def fig5():
    """Mechanism evidence on W1 at |A| = 24 (6 seeds).

    Top row (a)-(c) shares one map geometry and one pixel size, so the three
    panels can be read against each other cell by cell.  Bottom row (d) is the
    cross-term, the only statistic that ranks all four policies consistently.
    """
    with open(os.path.join(ROOT, "results_final", "figs",
                           "fig5_dwell_W1_24.json"), encoding="utf-8") as fh:
        D = json.load(fh)
    grid = np.array(D["layout"]["grid"], dtype=float)
    H, W = grid.shape

    def to_arr(rows):
        a = np.zeros((H, W))
        for x, y, v in rows:
            if 0 <= y < H and 0 <= x < W:
                a[y, x] = v
        return a

    haul = to_arr(D["haul"])
    dw = {k: to_arr(v) for k, v in D["dwell"].items()}
    stat = D["stat"]
    free = grid == 0.0                      # True where a cell is traversable

    # ------------------------------------------------------------ geometry --
    FW, FH = DC, 3.55
    uL, uR = 0.36, DC - 0.22
    uw = uR - uL
    gap = 0.30
    mw = (uw - 2 * gap) / 3.0               # width of one map panel
    ih = mw * H / W                         # keeps square pixels
    img_b = 2.34                            # bottom of every map image
    cb_off = 0.20                           # gap between image and its colourbar
    bar_b, bar_h = 0.44, 1.28

    def frac(x_in, y_in, w_in, h_in):
        return [x_in / FW, y_in / FH, w_in / FW, h_in / FH]

    fig = plt.figure(figsize=(FW, FH))

    panels = [("(a)  prior $\\mathrm{flow}(c)$",
               np.where(free, haul, np.nan), "struct. haul load", None),
              ("(b)  dwell, no $\\psi$",
               np.where(free, dw["memir0"], np.nan), "idle-time share", None),
              ("(c)  dwell, with $\\psi$",
               np.where(free, dw["memir"], np.nan), "idle-time share", None)]
    vmax = max(np.nanmax(p[1]) for p in panels[1:])
    for i, (title, arr, cblab, _) in enumerate(panels):
        x0 = uL + i * (mw + gap)
        ax = fig.add_axes(frac(x0, img_b, mw, ih))
        ax.imshow(np.where(free, -1.0, np.nan), cmap="Greys", vmin=-1, vmax=1,
                  interpolation="nearest", aspect="equal")
        vmx = np.nanmax(arr) if i == 0 else vmax
        im = ax.imshow(arr, cmap=CMAP, interpolation="nearest",
                       aspect="equal", vmin=0, vmax=max(vmx, 1e-9))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_linewidth(0.5)
        ax.set_title(title, loc="center", fontsize=7.8, pad=4)
        cb = fig.add_axes(frac(x0 + 0.02, img_b - cb_off, mw - 0.04, 0.075))
        cbar = fig.colorbar(im, cax=cb, orientation="horizontal")
        cbar.ax.tick_params(labelsize=5.8, length=2.0, pad=1.0)
        cbar.ax.set_xlabel(cblab, fontsize=6.2, labelpad=1.0)
        cb.set_axisbelow(True)

    # ---------------------------------------------------------- panel (d) --
    axb = fig.add_axes(frac(uL, bar_b, uw, bar_h))
    pols = [k for k in ["nearest", "static", "memir0", "memir"]
            if k in stat]
    names = {"nearest": "nearest", "static": "uniform-static",
             "memir0": "MEMIR\nno $\\psi$", "memir": "MEMIR"}
    ov = [stat[p]["overlap"] for p in pols]
    cols = [COL.get(p, "#666666") for p in pols]
    axb.bar(range(len(pols)), ov, width=0.52, color=cols,
            edgecolor="white", linewidth=0.5)
    for i, v in enumerate(ov):
        axb.text(i, v + 0.035, "%.2f" % v, ha="center", fontsize=7,
                 color="#374151")
    axb.set_xticks(range(len(pols)))
    axb.set_xticklabels([names[p] for p in pols], fontsize=7.5)
    axb.set_ylabel("dwell mass on haul corridors\n"
                   "$\\sum_c \\,\\mathrm{dwell}(c)\\cdot\\mathrm{haul}(c)$",
                   fontsize=8)
    style(axb)
    axb.set_ylim(0, 1.75)
    axb.set_title("(d)  dwell mass left on haul corridors",
                  loc="left", fontsize=7.8, pad=5)
    save(fig, "fig5_map")


# =============================================================== Fig 6 ====
# Table J (non-learning baselines) and Table K (learning baselines)
JDATA = {
    16: {"nearest": 36.56, "pmedian0": -8.32, "pmedian": -12.92,
         "popsample": -13.18, "popsample_f": -12.98, "memir0": -10.32,
         "memir": -14.68},
    32: {"nearest": 50.97, "pmedian0": 9.03, "pmedian": -5.31,
         "popsample": 0.74, "popsample_f": -1.19, "memir0": 3.30,
         "memir": -5.82},
}
JORDER = ["pmedian0", "pmedian", "popsample", "popsample_f", "memir0",
          "memir"]
JLABEL = ["p-median\n(no $\\psi$)", "p-median", "demand\nprop.",
          "demand\nprop. $+ \\psi$", "MEMIR\n(no $\\psi$)", "MEMIR"]
JNS = {32: {"popsample"}}
KDATA = {   # gap to MEMIR (positive = learner is worse)
    8: {"REINFORCE linear": 3.73, "contextual bandit": 31.03,
        "attention pointer": 3.73, "distilled 1-median": 2.75,
        "REINFORCE (long warm-up)": 1.84, "attention (long warm-up)": 1.50,
        "distilled (long warm-up)": -0.37},
    16: {"REINFORCE linear": 2.53, "contextual bandit": 14.89,
         "attention pointer": 3.22, "distilled 1-median": 6.77,
         "REINFORCE (long warm-up)": 3.09, "attention (long warm-up)": 3.77,
         "distilled (long warm-up)": 7.03},
    24: {"REINFORCE linear": 5.81, "contextual bandit": 7.92,
         "attention pointer": 5.19, "distilled 1-median": 10.75,
         "REINFORCE (long warm-up)": None, "attention (long warm-up)": None,
         "distilled (long warm-up)": None},
}
KNS = {8: {"REINFORCE (long warm-up)", "attention (long warm-up)",
           "distilled (long warm-up)"},
       16: {"REINFORCE (long warm-up)"}}


def fig6():
    """Baselines.  In both panels colour encodes fleet size and hatching
    encodes non-significance, so the two encodings are read the same way in
    every comparison.  The MEMIR bar group carries a dark outline."""
    from matplotlib.patches import Patch, Rectangle

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DC, 4.05))

    # ------------------------------------------------ (a) tree-based policies
    ax = ax1
    pols = JORDER
    names = ["p-median\n(no $\\psi$)", "p-median", "demand-prop.",
             "demand-prop.\n$+\\ \\psi$", "MEMIR\n(no $\\psi$)", "MEMIR"]
    x = np.arange(len(pols))
    w = 0.36
    fl = sorted(JDATA)
    for i, n in enumerate(fl):
        for j, k in enumerate(pols):
            v = JDATA[n][k]
            ns = k in JNS.get(n, set())
            ax.bar(x[j] + (i - 0.5) * w, v, w, color=FLEETCOL[n],
                   edgecolor="#25303a" if ns else "#ffffff",
                   linewidth=0.45 if ns else 0.5,
                   hatch=HATCH_NS if ns else HATCH_SIG,
                   zorder=3)
    for j, k in enumerate(pols):                     # outline the MEMIR group
        if k not in ("memir0", "memir"):
            continue
        ax.add_patch(Rectangle((x[j] - w - 0.02, -19), 2 * w + 0.04, 32.5,
                               fill=False, edgecolor="#111827", linewidth=0.7,
                               zorder=5))
    ax.axhline(0, color="#5b6570", lw=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.9)
    ax.set_ylabel("change in service time\nvs. uniform-static  (%)")
    style(ax)
    ax.set_ylim(-19, 13.5)
    h1 = [Patch(facecolor=FLEETCOL[n], edgecolor="white",
                label="fleet  $|A|=%d$" % n) for n in fl]
    h1 += [Patch(facecolor="#c6ced6", edgecolor="#25303a", hatch=HATCH_NS,
                 linewidth=0.45, label="not significant  ($p\\geq0.05$)")]
    leg = ax.legend(handles=h1, frameon=False, fontsize=6.6, ncol=3,
                    loc="upper center", bbox_to_anchor=(0.5, 1.005),
                    columnspacing=1.6, handlelength=1.5)
    leg.get_frame().set_linewidth(0)
    ax.set_title("(a)  non-learning policies, identical inputs", loc="left", fontsize=7.8, pad=4)

    # ------------------------------------------------ (b) learned policies ---
    ax = ax2
    keys = list(KDATA[8])
    short = ["REINFORCE\nlinear", "contextual\nbandit", "attention\npointer",
             "distilled\n1-median", "REINFORCE\nlinear$^\\dagger$",
             "attention\npointer$^\\dagger$",
             "distilled\n1-median$^\\dagger$"]
    x = np.arange(len(keys))
    w = 0.27
    fl2 = sorted(KDATA)
    for i, n in enumerate(fl2):
        for j, k in enumerate(keys):
            v = KDATA[n][k]
            if v is None:
                continue
            ns = k in KNS.get(n, set())
            ax.bar(x[j] + (i - 1) * w, v, w, color=FLEETCOL[n],
                   edgecolor="#25303a" if ns else "#ffffff",
                   linewidth=0.45 if ns else 0.5,
                   hatch=HATCH_NS if ns else HATCH_SIG, zorder=3)
    ax.axhline(0, color="#5b6570", lw=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.3)
    ax.set_ylabel("gap to MEMIR  (%)\n(positive = worse)")
    style(ax)
    ax.set_ylim(-4.5, 36)
    h2 = [Patch(facecolor=FLEETCOL[n], edgecolor="white",
                label="fleet  $|A|=%d$" % n) for n in fl2]
    h2 += [Patch(facecolor="#c6ced6", edgecolor="#25303a", hatch=HATCH_NS,
                 linewidth=0.45, label="not significant  ($p\\geq0.05$)")]
    leg = ax.legend(handles=h2, frameon=False, fontsize=6.6, ncol=4,
                    loc="upper right", columnspacing=1.6, handlelength=1.5)
    leg.get_frame().set_linewidth(0)
    ax.set_title("(b)  learned policies, more information; positive = worse", loc="left",
                 fontsize=7.8, pad=4)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.075,
                        hspace=0.52)
    save(fig, "fig6_baselines")


# =============================================================== Fig 7 ====
GDATA = [("W3  (213)", 8, -6.83), ("W3  (213)", 16, -2.35),
         ("W1  (711)", 8, -21.71), ("W1  (711)", 16, -14.68),
         ("W2  (1230)", 8, -16.70), ("W2  (1230)", 16, -11.71),
         ("B1 warehouse-10-20-10-2-1  (5699)", 16, -23.67),
         ("B1 warehouse-10-20-10-2-1  (5699)", 32, -19.58),
         ("B2 warehouse-10-20-10-2-2  (9776)", 16, -19.83),
         ("B2 warehouse-10-20-10-2-2  (9776)", 32, -25.76)]
NLEG = {   # standard maps, static -> MEMIR
    "B1 $|A|{=}16$": (72.92, 44.01, 77.51, 77.60),
    "B1 $|A|{=}32$": (51.94, 22.10, 77.93, 78.36),
    "B2 $|A|{=}16$": (80.63, 57.46, 91.64, 90.62),
    "B2 $|A|{=}32$": (66.39, 26.33, 91.97, 91.33),
}


def fig7():
    from matplotlib.patches import Patch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, 2.95),
                                   gridspec_kw=dict(width_ratios=[1.14, 1.0]))

    # ------------------------------------- (a) gains across five layouts -----
    ax = ax1
    maps = [("W3", "213"), ("W1", "711"), ("W2", "1230"),
            ("B1", "5699"), ("B2", "9776")]       # ordered by extent
    x = np.arange(len(maps))
    w = 0.26
    for i, n in enumerate([8, 16, 32]):
        xs, ys = [], []
        for j, (code, cells) in enumerate(maps):
            hit = [g for mm, nn, g in GDATA
                   if mm.split()[0] == code and nn == n]
            if hit:
                xs.append(j + (i - 1) * w)
                ys.append(hit[0])
        if ys:
            ax.bar(xs, ys, width=w, color=FLEETCOL[n], edgecolor="white",
                   linewidth=0.4, label="$|A|=%d$" % n)
    ax.axhline(0, color="#5b6570", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (code, cells) for code, cells in maps],
                       fontsize=7)
    ax.set_ylabel("change in service time\nvs. uniform-static  (%)")
    style(ax)
    ax.set_ylim(-33, 15)
    leg = ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper center",
                    bbox_to_anchor=(0.5, 1.005), handlelength=1.3,
                    columnspacing=1.5, handletextpad=0.5)
    leg.get_frame().set_linewidth(0)
    ax.set_title("(a)  transfer to unseen layouts", loc="left",
                 fontsize=7.8, pad=4)

    # --------------------------- (b) where the gain sits on MovingAI maps ----
    ax = ax2
    labels = list(NLEG)
    xs = np.arange(len(labels)) * 3.1
    w = 0.62
    col = {"s2": "#f4c07a", "m2": "#d55e00", "s3": "#a9cdec",
           "m3": "#0072b2"}
    for j, k in enumerate(labels):
        s2, m2, s3, m3 = NLEG[k]
        quad = [(0.00, s2, col["s2"]), (0.72, m2, col["m2"]),
                (1.62, s3, col["s3"]), (2.34, m3, col["m3"])]
        for off, v, c in quad:
            ax.bar(xs[j] + off, v, width=w, color=c, edgecolor="white",
                   linewidth=0.45)
    style(ax)
    ax.set_xticks([xs[j] + 1.17 for j in range(len(labels))])
    ax.set_xticklabels(["B1\n$|A|{=}16$", "B1\n$|A|{=}32$", "B2\n$|A|{=}16$",
                        "B2\n$|A|{=}32$"], fontsize=7.2)
    ax.set_ylabel("leg duration  (timesteps)")
    ax.set_ylim(0, 138)
    h = [Patch(facecolor=col["s2"], edgecolor="white",
               label="$T_2$ approach $-$ uniform-static"),
         Patch(facecolor=col["m2"], edgecolor="white",
               label="$T_2$ approach $-$ MEMIR"),
         Patch(facecolor=col["s3"], edgecolor="white",
               label="$T_3$ haul $-$ uniform-static"),
         Patch(facecolor=col["m3"], edgecolor="white",
               label="$T_3$ haul $-$ MEMIR")]
    leg = ax.legend(handles=h, frameon=False, fontsize=6.2, ncol=1,
                    loc="upper left", bbox_to_anchor=(0.015, 1.005),
                    handlelength=1.3, labelspacing=0.32,
                    title="leg and policy", title_fontsize=6.4,
                    borderpad=0.1)
    leg.get_frame().set_linewidth(0)
    ax.set_title("(b)  where the gain sits (benchmark maps)",
                 loc="left", fontsize=7.8, pad=4)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.155,
                        wspace=0.30)
    save(fig, "fig7_transfer")


# =============================================================== Fig 8 ====
# Table O (24 cells) and Table P (attribution of the large-fleet reversal)
OSCALE = [
    ("B1", 5699, 64, 0.16, -12.73, 75.1), ("B1", 5699, 64, 0.32, -22.03, 43.2),
    ("B1", 5699, 128, 0.32, 1.56, 76.1), ("B1", 5699, 128, 0.64, -14.37, 39.9),
    ("B1", 5699, 256, 0.64, 24.32, 74.0), ("B1", 5699, 256, 1.28, -5.35, 35.2),
    ("B3", 22599, 64, 0.16, -22.54, 38.5), ("B3", 22599, 64, 0.32, -1.46, 8.6),
    ("B3", 22599, 128, 0.32, -16.33, 37.9),
    ("B3", 22599, 128, 0.64, -1.80, 8.6),
    ("W4", 2551, 32, 0.08, -14.35, 82.0), ("W4", 2551, 32, 0.16, -27.17, 62.0),
    ("W4", 2551, 64, 0.16, -11.46, 81.1), ("W4", 2551, 64, 0.32, -26.04, 62.1),
    ("W4", 2551, 128, 0.32, -13.96, 80.9),
    ("W4", 2551, 128, 0.64, -25.60, 62.3),
    ("W5", 5511, 64, 0.16, -23.81, 72.0), ("W5", 5511, 64, 0.32, -10.84, 15.9),
    ("W5", 5511, 128, 0.32, -25.57, 72.2),
    ("W5", 5511, 128, 0.64, -10.46, 18.9),
    ("W6", 9591, 128, 0.32, -31.72, 62.1), ("W6", 9591, 128, 0.64, -3.12, 6.9),
    ("W6", 9591, 256, 0.64, -16.90, 57.2),
    ("W6", 9591, 256, 1.28, -1.40, 6.2),
]
MAPC = {"W4": "#66c2a5", "W5": "#3288bd", "W6": "#5e4fa2",
        "B1": "#d53e4f", "B3": "#f46d43"}
MAPLAB = {"W4": "W4  (2 551 cells)", "W5": "W5  (5 511)",
          "W6": "W6  (9 591)", "B1": "B1  (5 699)", "B3": "B3  (22 599)"}
PAF = [307, 871, 1494, 5094, 10454, 22873]
PAFG = [-22.03, -14.37, -12.73, -5.35, 1.56, 24.32]


def fig8():
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DC, 2.95))
    ax = ax1
    for m in ["W4", "W5", "W6", "B1", "B3"]:
        pts = [p for p in OSCALE if p[0] == m]
        ax.scatter([p[5] for p in pts], [p[4] for p in pts],
                   s=[12 + 26 * (np.log10(p[2]) - 0.4) for p in pts],
                   facecolors="none" if m in ("B1", "B3") else MAPC[m],
                   edgecolors=MAPC[m], linewidths=1.1, alpha=0.9,
                   label=MAPLAB[m])
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.axvspan(0, 15, color="#f6f7f9", zorder=0)
    ax.text(7.5, 21, "saturated:\nnothing to schedule", ha="center",
            va="top", fontsize=6.2, color="#6b7280")
    ax.set_xlabel("idle fraction of the fleet under MEMIR  (%)")
    ax.set_ylabel("change in service time\nvs. uniform-static  (%)")
    ax.set_xlim(0, 90)
    style(ax)
    ax.set_title("(a)  gain against idle slack (24 cells)",
                 loc="left", fontsize=8)

    ax = ax2
    ax.plot(np.log10(PAF), PAFG, "-o", color="#b2182b", ms=4.5, lw=1.5)
    ax.axhline(0, color="#9ca3af", lw=0.6)
    for x, y in zip(np.log10(PAF), PAFG):
        ax.annotate("", xy=(x, y), xytext=(x, 0), fontsize=6,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#cbd5e1"))
    ax.set_xticks(np.log10(PAF))
    ax.set_xticklabels(["%d" % v for v in PAF], rotation=35, ha="right",
                       fontsize=6.2)
    ax.set_xlabel("failed planning attempts under MEMIR  (log axis)")
    ax.set_ylabel("change in service time\nvs. uniform-static  (%)")
    style(ax)
    ax.set_title("(b)  attribution of the B1 reversal",
                 loc="left", fontsize=8)

    # one shared legend, below both panels (same convention as Figs. 5, 6, 9)
    h = [Line2D([], [], marker="o", ms=5.2, lw=0,
                markerfacecolor=MAPC[m] if m not in ("B1", "B3") else "none",
                markeredgecolor=MAPC[m], markeredgewidth=1.1,
                label=MAPLAB[m]) for m in ["W4", "W5", "W6", "B1", "B3"]]
    leg = fig.legend(handles=h, frameon=False, ncol=3, fontsize=6.4,
                     loc="lower center", bbox_to_anchor=(0.5, 0.035),
                     handletextpad=0.5, columnspacing=1.4)
    leg.get_frame().set_linewidth(0)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.295,
                        wspace=0.34)
    save(fig, "fig8_scale")


def fig9():
    """Decomposition of the MEMIR gain into two mechanisms.

    Panel (a): four variants that differ ONLY in which of the two components
    is switched on (Table 2).  Panel (b): the marginal contribution of each
    component, measured in percentage points of the uniform-static reference.
    Hatched bars are not significant (Wilcoxon signed-rank, n = 10 seeds).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fleet = [4, 6, 8, 10, 12, 16, 24, 32, 40]
    win0 = [0.28, -17.72, -12.75, -15.36, -20.93, -10.62, -8.20, -0.00, 1.31]
    win = [-1.36, -15.98, -12.12, -15.50, -21.59, -12.21, -11.33, -5.73, -6.09]
    mem0 = [-0.94, -24.39, -20.71, -20.96, -24.23, -10.32, -5.01, 3.30, 3.22]
    mem = [-1.01, -25.14, -21.71, -22.31, -26.18, -14.68, -11.84, -5.82, -6.79]

    d_mem = [b - a for a, b in zip(win0, mem0)]      # window -> EMA, no psi
    d_psi = [b - a for a, b in zip(win0, win)]       # window only, psi on
    ns_mem = {0, 5}                                  # |A| = 4, 16
    ns_psi = {0, 2, 3, 4}                            # |A| = 4, 8, 10, 12

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DC, 3.15),
        gridspec_kw=dict(width_ratios=[1.06, 1.0]))

    # ---------------- (a)  four variants ----------------
    series = [("memory $+$ $\\psi$  (MEMIR)", mem, "#009E73", "D", "-"),
              ("memory only  (no $\\psi$)", mem0, "#D55E00", "^", "--"),
              ("window $+$ $\\psi$", win, "#0072B2", "s", "-."),
              ("window only  (no memory)", win0, "#9ad0f0", "o", ":")]
    hy = []
    for lab, ys, c, mk, ls in series:
        ln, = ax1.plot(fleet, ys, ls, marker=mk, ms=3.8, lw=1.3, color=c,
                       mfc="white" if mk in ("o", "s") else c, label=lab)
        hy.append(ln)
    ax1.axhline(0, lw=0.7, color="#5b6570")
    ax1.set_xscale("log")
    ax1.xaxis.set_major_locator(FixedLocator(fleet))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "%d" % v))
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlim(3.4, 46)
    ax1.set_ylim(-31, 7)
    ax1.set_xlabel("fleet size  $|A|$  (log axis)")
    ax1.set_ylabel("change in service time\nvs. uniform-static  (%)")
    style(ax1)
    ax1.set_title("(a)  the factorial ablation",
                  loc="left", fontsize=7.8, pad=4)

    # ---------------- (b)  marginal contributions ----------------
    x = np.arange(len(fleet))
    w = 0.40
    b1 = ax2.bar(x - w / 2, d_mem, w, color="#D55E00", lw=0.5,
                 edgecolor="white")
    b2 = ax2.bar(x + w / 2, d_psi, w, color="#0072B2", lw=0.5,
                 edgecolor="white")
    for i, (ra, rb) in enumerate(zip(b1, b2)):
        for r, nsset in ((ra, ns_mem), (rb, ns_psi)):
            if i in nsset:
                r.set_hatch(HATCH_NS)
                r.set_edgecolor("#25303a")
    ax2.axhline(0, lw=0.7, color="#5b6570")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["%d" % v for v in fleet], fontsize=7)
    ax2.set_xlabel("fleet size  $|A|$")
    ax2.set_ylabel("marginal contribution\n(percentage points)")
    ax2.set_ylim(-14.5, 9.5)
    style(ax2)
    ax2.set_title("(b)  marginal contributions", loc="left",
                  fontsize=7.8, pad=4)
    ax2.annotate("memory turns harmful once the fleet\n"
                 "can block itself",
                 xy=(7.0, 3.6), xytext=(0.30, 8.4), fontsize=6.1,
                 color="#8a4a00", ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="#8a4a00",
                                 shrinkA=1, shrinkB=2))

    # one legend for the whole figure, so neither panel is covered by text
    hb = [Patch(facecolor="#D55E00", edgecolor="white",
                label="effect of memory  (EMA $-$ window)"),
          Patch(facecolor="#0072B2", edgecolor="white",
                label="effect of the prior $\\psi$"),
          Patch(facecolor="#c6ced6", edgecolor="#25303a", hatch=HATCH_NS,
                linewidth=0.45, label="not significant  ($p\\geq0.05$)")]
    leg = fig.legend(handles=hy + hb, loc="lower center",
                     bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=6.5,
                     frameon=False, columnspacing=1.5, handlelength=1.9,
                     handletextpad=0.6)
    leg.get_frame().set_linewidth(0)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.30,
                        wspace=0.26)
    save(fig, "fig9_decomp")


def main():
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print("font family:", FAM)


if __name__ == "__main__":
    main()
