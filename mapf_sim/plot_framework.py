"""Figure 1 of the paper: where the proposed layer sits in the MAPD stack.

Vector output (PDF with embedded TrueType + SVG with editable text).
All strings are English; width matches the MDPI \\textwidth (394 pt = 5.48 in).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

CAND = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
FAM = next((c for c in CAND if c in _avail), "DejaVu Sans")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [FAM]
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["font.size"] = 8.0
rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figs")
os.makedirs(OUT, exist_ok=True)

W, H = 5.48, 3.95

GREY = "#e8ebef"
GREY_E = "#c9d1da"
BLUE = "#dbe9f4"
BLUE_E = "#4a90c2"
GREEN = "#dff0e6"
GREEN_E = "#1c7a46"
ORANGE = "#fbe6d2"
ORANGE_E = "#b06a1e"
IN = "#ffffff"
TXT = "#1f2933"
SUB = "#5b6570"


def box(ax, x, y, w, h, fc, ec, lw=0.8, style="round,pad=0.02,rounding_size=0.06",
        dash=None, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                       edgecolor=ec, facecolor=fc, zorder=z,
                       linestyle=(0, dash) if dash is not None else "solid",
                       mutation_scale=1.0)
    ax.add_patch(p)
    return p


def label(ax, x, y, s, size=8.0, color=TXT, weight="normal", ha="center",
          va="center", z=3):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha,
            va=va, zorder=z)


def arrow(ax, p1, p2, color="#6b7280", lw=0.9, dash=None, rad=0.0,
          style="-|>", ms=7, z=4):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                        linewidth=lw, color=color, zorder=z,
                        linestyle=(0, dash) if dash is not None else "solid",
                        connectionstyle="arc3,rad=%f" % rad,
                        shrinkA=0.5, shrinkB=0.5)
    ax.add_patch(a)
    return a


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ------------------------------------------------ sources (top) ------
    box(ax, 0.05, 3.50, 1.72, 0.42, ORANGE, ORANGE_E)
    label(ax, 0.91, 3.71, "task release stream", 8)
    box(ax, 3.66, 3.50, 1.77, 0.42, ORANGE, ORANGE_E)
    label(ax, 4.55, 3.71, "map $G$ and endpoint sets", 8)

    # ------------------------------------------------ decision stack -----
    stack = [
        (2.72, "L1   task assignment", "which agent takes which task", GREY, GREY_E),
        (2.02, "L2   space-time path", "collision-free trajectory", GREY, GREY_E),
        (1.32, "L3   idle repositioning", "where an unloaded agent waits", GREEN, GREEN_E),
    ]
    for y, t, s, fc, ec in stack:
        box(ax, 0.05, y, 1.72, 0.56, fc, ec)
        label(ax, 0.91, y + 0.35, t, 8, weight="bold")
        label(ax, 0.91, y + 0.15, s, 6.6, color=SUB)
    box(ax, 0.05, 0.60, 1.72, 0.42, GREY, GREY_E)
    label(ax, 0.91, 0.81, "execution (token passing)", 7.6)

    arrow(ax, (0.55, 2.72), (0.55, 2.58))
    arrow(ax, (0.55, 2.02), (0.55, 1.88))
    arrow(ax, (0.55, 1.32), (0.55, 1.02))
    label(ax, 0.62, 2.65, "assigned task", 6.2, color=SUB, ha="left")
    label(ax, 0.62, 1.95, "after delivery", 6.2, color=SUB, ha="left")
    label(ax, 0.62, 1.15, "reservation", 6.2, color=SUB, ha="left")

    # ------------------------------------------------ MEMIR panel --------
    px, py, pw, ph = 2.02, 0.42, 3.41, 2.86
    box(ax, px, py, pw, ph, "#fbfcfd", GREEN_E, lw=1.0, dash=(4, 2), z=1)
    label(ax, px + 0.06, py + ph - 0.15, "MEMIR:  idle repositioning layer",
          8.2, color=GREEN_E, weight="bold", ha="left")

    iw = (pw - 0.30) / 2.0
    # two memory / prior boxes
    box(ax, px + 0.10, 2.42, iw, 0.62, IN, BLUE_E)
    label(ax, px + 0.10 + iw / 2, 2.75, "arrival intensity field", 7.6)
    label(ax, px + 0.10 + iw / 2, 2.55, "$\\hat\\lambda_r(t)$", 8.4,
          color=BLUE_E, weight="bold")

    box(ax, px + 0.20 + iw, 2.42, iw, 0.62, IN, ORANGE_E)
    label(ax, px + 0.20 + iw + iw / 2, 2.75, "haul-flow prior", 7.6)
    label(ax, px + 0.20 + iw + iw / 2, 2.55, "$\\mathrm{flow}(c)$", 8.4,
          color=ORANGE_E, weight="bold")

    # objective
    box(ax, px + 0.10, 1.72, pw - 0.20, 0.58, GREEN, GREEN_E)
    label(ax, px + pw / 2, 2.10,
          "$J(c)\\,=\\,\\sum_r \\hat p_r\\,\\bar d(c,r)\\;+\\;"
          "\\beta\\,d(\\mathrm{pos},c)\\;+\\;\\psi\\,\\mathrm{flow}(c)$",
          8.6, color="#14532d")

    # candidate minimisation
    box(ax, px + 0.10, 1.16, pw - 0.20, 0.42, IN, "#5b6570")
    label(ax, px + pw / 2, 1.37, "minimise over candidate cells $C$", 7.8)

    # safety filter
    box(ax, px + 0.10, 0.66, pw - 0.20, 0.42, IN, "#8a4a00")
    label(ax, px + pw / 2, 0.87, "TER-Safety reachability filter", 7.8)

    # output
    box(ax, px + 0.10, 0.10, pw - 0.20, 0.40, GREEN, GREEN_E)
    label(ax, px + pw / 2, 0.30, "resting cell  $c^\\star$", 8.6,
          color=GREEN_E, weight="bold")

    arrow(ax, (px + 0.10 + iw / 2, 2.42), (px + 0.10 + iw / 2, 2.30))
    arrow(ax, (px + 0.20 + iw + iw / 2, 2.42),
          (px + 0.20 + iw + iw / 2, 2.30))
    arrow(ax, (px + pw / 2, 1.72), (px + pw / 2, 1.58))
    arrow(ax, (px + pw / 2, 1.16), (px + pw / 2, 1.08), style="-|>", ms=6)
    arrow(ax, (px + pw / 2, 0.66), (px + pw / 2, 0.50))

    # ------------------------------------------------ couplings ----------
    arrow(ax, (1.77, 3.71), (px + 0.10 + iw / 2, 3.71), color=ORANGE_E,
          rad=0.0)
    arrow(ax, (px + 0.10 + iw / 2, 3.71), (px + 0.10 + iw / 2, 3.06),
          color=ORANGE_E)
    label(ax, px + 0.10 + iw / 2 + 0.05, 3.42, "observed releases",
          6.4, color=ORANGE_E, ha="left")

    arrow(ax, (4.55, 3.50), (4.55, 3.06), color=ORANGE_E)
    label(ax, 4.60, 3.30, "computed once, offline", 6.4, color=ORANGE_E,
          ha="left")

    arrow(ax, (px, 0.30), (1.77, 1.55), color=GREEN_E, dash=(4, 2))
    label(ax, 1.90, 0.62, "target cell", 6.4, color=GREEN_E, ha="left")

    label(ax, 0.91, 0.30, "layers L1 and L2 untouched", 6.6, color=SUB)

    fig.savefig(os.path.join(OUT, "fig0_framework.pdf"), bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(os.path.join(OUT, "fig0_framework.svg"), bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("wrote fig0_framework   (font:", FAM + ")")


if __name__ == "__main__":
    main()
