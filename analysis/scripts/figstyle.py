#!/usr/bin/env python
# =============================================================================
# figstyle.py — ONE shared visual system for every figure in this paper.
#
# Rewritten 2026-08-03 after two figure audits. Three things were wrong and are fixed here:
#   * canvases were 305-345 mm wide against a 180 mm double-column maximum, so the type ladder
#     collapsed below the legibility floor at reproduction scale. Sizes are now in millimetres.
#   * the palette was isoluminant — top leaf Y=40 against bottom leaf Y=44 on a 0-255 scale — so
#     it vanished in greyscale. Every family is now spaced on luminance, not hue.
#   * MODEL shared hex values with TREAT, so teal meant "control" in one panel of a figure and
#     "both traits" in the next. The three families are now disjoint, and that is enforced by a test.
#
# Nothing may be encoded by colour alone: every categorical contrast also carries a marker or a
# hatch. Panel letters are positioned in points so they do not drift between figures.
# =============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---- canvas: journal column widths, in inches ------------------------------
MM = 1.0 / 25.4
SINGLE = 85 * MM          # 3.35 in
ONE_HALF = 130 * MM       # 5.12 in
DOUBLE = 180 * MM         # 7.09 in — the hard maximum for a two-column figure

# ---- palette: spaced on Rec.709 luminance so it survives greyscale ---------
# Y values in comments are on 0-255 after linearisation.
TREAT = {"Control": "#0f5f59", "NStress": "#e8912e"}          # Y 23 vs 96
TREAT_LABEL = {"Control": "control", "NStress": "low nitrogen"}
TREAT_MARKER = {"Control": "o", "NStress": "^"}               # never colour alone
TREAT_HATCH = {"Control": "", "NStress": "///"}

LEAFPOS = {"top": "#2b4a7d", "bottom": "#c25f2a", "ratio": "#9b8ec4"}   # Y 18, 51, 78
LEAFPOS_LABEL = {"top": "top leaf", "bottom": "bottom leaf", "ratio": "bottom : top"}
LEAFPOS_MARKER = {"top": "o", "bottom": "s", "ratio": "^"}

# A rule is named by the trait it uses, so it carries that trait's colour: no hex means two
# things, and the reader sees at a glance which reading each rule rests on. "both" is a dark
# neutral because it belongs to neither trait alone.
MODEL = {"rCCI_30": LEAFPOS["ratio"], "QY_BL_30": LEAFPOS["bottom"], "both": "#1f2933"}
MODEL_LABEL = {"rCCI_30": "rCCI$_{30}$ alone", "QY_BL_30": "F$_v$/F$_m$ alone",
               "both": "both traits"}
MODEL_MARKER = {"rCCI_30": "s", "QY_BL_30": "^", "both": "o"}

TIME = {"30 DAT": "#37678f", "60 DAT": "#a9c4dc"}          # Y 44 vs 137
TIME_HATCH = {"30 DAT": "", "60 DAT": "///"}

# Greys carry no category. They are for quantities that are not a category at all: a single curve,
# a reference range, a bar whose identity is already given by its own printed label. Nothing that
# needs to be told apart from something else may be drawn in a grey.
BLACK = "0.10"            # every number printed inside a plot
NEUTRAL = "0.55"          # a bar or a curve that encodes nothing by its colour
NEUTRAL_LIGHT = "0.85"

VARCOMP = {"pct_var_T": "#3a3a3a", "pct_var_G": "#7d7d7d",
           "pct_var_GxT": "#b4b4b4", "pct_var_e": "#e0e0e0"}
VARCOMP_LABEL = {"pct_var_T": "nitrogen", "pct_var_G": "genotype",
                 "pct_var_GxT": "genotype × nitrogen", "pct_var_e": "residual"}
VARCOMP_HATCH = {"pct_var_T": "", "pct_var_G": "///", "pct_var_GxT": "...", "pct_var_e": ""}

DIVERGING = LinearSegmentedColormap.from_list(
    "topbottom", ["#2b4a7d", "#9fb3d1", "#f4f4f2", "#e7b48f", "#c25f2a"])
CORR = mpl.colormaps["RdBu_r"]


def _luma(hex_colour):
    h = hex_colour.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 255 * (0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b))


def check_palette(min_gap=18):
    """Fail loudly if any family becomes isoluminant again, or if two families collide."""
    problems = []
    for name, fam in (("TREAT", TREAT), ("LEAFPOS", LEAFPOS), ("MODEL", MODEL)):
        vals = sorted((_luma(c), k) for k, c in fam.items())
        for (y1, k1), (y2, k2) in zip(vals, vals[1:]):
            if y2 - y1 < min_gap:
                problems.append("%s: %s and %s differ by only %.0f of 255" % (name, k1, k2, y2 - y1))
    # MODEL deliberately reuses LEAFPOS: a rule is named by its trait. TREAT must stay disjoint
    # from both, because a treatment group and a measurement are different kinds of thing.
    shared = (set(TREAT.values()) & set(MODEL.values())) | \
             (set(TREAT.values()) & set(LEAFPOS.values()))
    if shared:
        problems.append("treatment colours collide with a measurement family: %s" % sorted(shared))
    return problems


# ---- one naming table for the fifteen genotypes -----------------------------
# 05_figures.py, 05b_figures_extended.py and 05c_figures_supp.py all label the same plants. They
# used to each hold their own copy, or none, so the supplementary figures printed BAM812 where the
# main figures printed BAM 812. One table, read by all three.
ALIAS = {"MOROBEREKAN": "MOROBERAKAN", "RPW9SS1": "RPW94SS1"}
DISPLAY = {"MOROBERAKAN": "Moroberakan", "IR83388BB1083": "IR83388-B-B108-3",
           "IR77298141210": "IR77298-14-1-2-10", "PUSA1121": "Pusa 1121",
           "BAM8315": "BAM 8315", "BAM812": "BAM 812", "MALCHI": "Malchi",
           "BAM3690": "BAM 3690", "BAM4138": "BAM 4138", "BAM4521": "BAM 4521",
           "BLACKGORA": "Black Gora", "SUWEON": "Suweon", "KUNJUKUNJU": "Kunjukunju",
           "CAUVERY": "Cauvery", "RPW94SS1": "RPW-9-4 (SS1)"}


def key(x):
    k = "".join(str(x).upper().split()).replace("-", "").replace("(", "").replace(")", "")
    return ALIAS.get(k, k)


def disp(x):
    return DISPLAY.get(key(x), str(x))


def sub(t):
    """Trait label with the leaf-position suffix set as a subscript. The underscore glyph sits on
    the baseline and strikes through the row beneath it on a dense categorical axis."""
    return str(t).replace("_TL", "$_{TL}$").replace("_BL", "$_{BL}$")


# ---- typography: sized for the printed page, nothing below FS_MIN ----------
FS_TITLE, FS_LABEL, FS_TICK = 8.0, 7.5, 6.5
FS_ANNOT, FS_LEGEND, FS_PANEL, FS_MIN = 6.5, 6.5, 9.0, 6.0


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 7.5, "axes.titlesize": FS_TITLE, "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "lines.linewidth": 1.0, "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.0,
        "figure.dpi": 300, "savefig.dpi": 600,
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42, "svg.fonttype": "none",     # editable vector text
        "mathtext.default": "regular",
    })


def panel_letter(ax, letter, dx=-24, dy=5, size=None):
    """Panel letter at a fixed offset in POINTS from the top-left of the axes, so the position
    does not drift with axes size the way an axes-fraction offset does."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=size or FS_PANEL, fontweight="bold", va="bottom", ha="left")


def half_violin(ax, x, values, side, colour, width=0.38, alpha=0.55):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        ax.scatter(np.full(len(v), x), v, s=6, color=colour, zorder=3, linewidths=0)
        return
    parts = ax.violinplot([v], positions=[x], widths=width * 2, showextrema=False, showmedians=False)
    for b in parts["bodies"]:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        if side == "left":
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_facecolor(colour); b.set_alpha(alpha)
        b.set_edgecolor(colour); b.set_linewidth(0.5)
    off = -0.05 if side == "left" else 0.05
    ax.plot([x + off], [np.median(v)], marker="_", ms=5, mew=1.1, color="0.1", zorder=4)


def strip(ax, x, values, colour, jitter=0.05, size=5, seed=0, marker="o"):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    xs = np.random.default_rng(seed).normal(x, jitter, len(v))
    ax.scatter(xs, v, s=size, color=colour, zorder=3, linewidths=0, alpha=0.9, marker=marker)


def reaction_norms(ax, control, stress, colour, lw=0.5, alpha=0.55):
    for c, s in zip(control, stress):
        ax.plot([0, 1], [c, s], color=colour, lw=lw, alpha=alpha, zorder=2)
    ax.plot([0, 1], [np.nanmean(control), np.nanmean(stress)], color="0.1", lw=1.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["control", "low N"], fontsize=FS_TICK)
    ax.set_xlim(-0.22, 1.22)


def star(p):
    """Marker from a real tail probability. Above 0.05 nothing is drawn, so a blank never
    reads as an unreported test — the caption states the convention."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def save(fig, path):
    """PNG for review and PDF for the journal, from one call, so the two cannot drift."""
    fig.savefig(path)
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf")
