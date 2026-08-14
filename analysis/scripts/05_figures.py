#!/usr/bin/env python
# =============================================================================
# 05_figures.py — main Figures 1 to 5, at journal print size.
#
#   Fig 1  how it was done                     design and soils, timeline, leaf positions
#   Fig 2  where the nitrogen signal sits      distributions with reaction norms, F landscape, variance
#   Fig 3  why the ratio transfers             bottom-vs-top with the 1:1 line, per-genotype rCCI
#   Fig 4  does it classify an unseen genotype complementarity, decision space, scores, accuracy
#   Fig 5  can a practitioner use it           threshold, sensitivity/specificity, benchmark
#
# Rebuilt 2026-08-03 after two figure audits. The canvas is now 180 mm (S.DOUBLE) rather than the
# 345 mm used before, so the type ladder in figstyle.py is legible at reproduction scale. Panels the
# audits judged redundant were removed rather than shrunk: the script-flow panel of Figure 1 (internal
# build documentation), the two rows of Figure 2 merged into one, the slope chart and the 30-vs-60 DAT
# panel of Figure 3 (both carried by Figure 6), and the ROC panel of Figure 5 (an L-shape carrying no
# information once a trait separates perfectly). No claim sentence is printed inside any image, every
# categorical contrast carries a marker as well as a colour, and Figure 4 now leads with the
# complementarity rather than with a resubstitution fit.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.stats import f as _fdist
import figstyle as S

S.set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
MAIN = os.path.join(ROOT, "analysis", "results", "figures", "main")
os.makedirs(MAIN, exist_ok=True)

FOCAL = [("CCI_TL", "CCI, top leaf", "top"), ("CCI_BL", "CCI, bottom leaf", "bottom"),
         ("QY_TL", "F$_v$/F$_m$, top leaf", "top"), ("QY_BL", "F$_v$/F$_m$, bottom leaf", "bottom")]

# One canonical display name per genotype, so the same plant is never labelled two ways. The table
# now lives in figstyle.py, so the supplementary figures read the same one.
key, disp = S.key, S.disp
BLACK = S.BLACK


def save(fig, name):
    S.save(fig, os.path.join(MAIN, name))
    plt.close(fig)
    print("  wrote", name)


def load():
    plants = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
    gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
    gm["treatment"] = gm["treatment"].replace({"N stress": "NStress"})
    Fa = pd.read_csv(os.path.join(DERIVED, "anova_F_authors_table1.csv"))
    return plants, gm, Fa


def gv(gm, trait, treat, tp="30 DAT", col="value"):
    d = gm[(gm["trait"] == trait) & (gm["treatment"] == treat) & (gm["timepoint"] == tp)]
    return d.sort_values("genotype_code")[col].to_numpy(float)


def gnames(gm, trait, treat, tp="30 DAT"):
    d = gm[(gm["trait"] == trait) & (gm["treatment"] == treat) & (gm["timepoint"] == tp)]
    return [disp(n) for n in d.sort_values("genotype_code")["genotype_name"]]


# =========================================================== FIGURE 1
def fig1():
    fig = plt.figure(figsize=(S.DOUBLE, 3.35))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.20, 0.72], wspace=0.30,
                  left=0.015, right=0.99, top=0.90, bottom=0.03)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    S.panel_letter(ax, "a", dx=0, dy=2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 9.6, "15 genotypes × 2 nitrogen regimes × 3 pots = 90 plants",
            fontsize=S.FS_ANNOT, fontweight="bold")
    x0 = 0.05
    # Box width used to encode n. That made "japonica (2)" a 1.04-unit label in a 1.04-unit box, so
    # two of the four labels ran into their neighbours. The count is printed inside the box, so the
    # width was carrying nothing the reader could not already read.
    for nm, k in (("indica", 6), ("basmati", 5), ("japonica", 2), ("aus", 2)):
        w = 2.30
        ax.add_patch(FancyBboxPatch((x0, 8.2), w, 0.62, boxstyle="round,pad=0.03",
                                    fc="#eef1f5", ec="#9aa7b5", lw=0.5))
        ax.text(x0 + w / 2, 8.51, "%s (%d)" % (nm, k), ha="center", va="center", fontsize=S.FS_MIN)
        x0 += w + 0.10
    ax.text(0, 7.3, "soil before transplanting, kg ha$^{-1}$",
            fontsize=S.FS_ANNOT, fontweight="bold")
    for i, (nm, c, n) in enumerate((("available N", 220.62, 155.17),
                                    ("available P$_2$O$_5$", 96.06, 67.19),
                                    ("available K$_2$O", 162.42, 113.15))):
        y = 6.3 - i * 1.30
        ax.text(0, y - 0.20, nm, fontsize=S.FS_MIN, va="center")
        for j, (val, t) in enumerate(((c, "Control"), (n, "NStress"))):
            yy = y - j * 0.44
            ax.barh(yy, val / 250 * 2.9, left=4.1, height=0.34, color=S.TREAT[t],
                    hatch=S.TREAT_HATCH[t], edgecolor="white", lw=0.4)
            ax.text(4.15 + val / 250 * 2.9, yy, " %.1f" % val, fontsize=S.FS_MIN, va="center")
    ax.text(0, 2.05, "applied, kg ha$^{-1}$", fontsize=S.FS_ANNOT, fontweight="bold")
    ax.text(0, 1.35, "control   120 N : 80 P$_2$O$_5$ : 60 K$_2$O", fontsize=S.FS_MIN,
            color=S.TREAT["Control"])
    ax.text(0, 0.70, "low N        0 N : 80 P$_2$O$_5$ : 60 K$_2$O", fontsize=S.FS_MIN,
            color=S.TREAT["NStress"])
    ax.legend(handles=[Line2D([], [], color=S.TREAT[t], lw=3, label=S.TREAT_LABEL[t])
                       for t in ("Control", "NStress")],
              loc="lower right", fontsize=S.FS_MIN, ncol=1)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    S.panel_letter(ax, "b", dx=0, dy=2)
    ax.set_xlim(-9, 100); ax.set_ylim(0, 10)
    ax.annotate("", xy=(97, 4.6), xytext=(-6, 4.6),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.3"))
    for x, lab, sub in ((0, "0", "transplanting"), (30, "30", "max. tillering"),
                        (60, "60", "reproductive"), (90, "harvest", "")):
        ax.plot([x], [4.6], marker="|", ms=6, color="0.3", mew=0.9)
        ax.text(x, 3.95, lab, ha="center", va="top", fontsize=S.FS_MIN, fontweight="bold")
        if sub:
            ax.text(x, 3.30, sub, ha="center", va="top", fontsize=S.FS_MIN, color="0.4")
    ax.text(45, 2.25, "days after transplanting", ha="center", fontsize=S.FS_MIN, color="0.35")
    for x, lab in ((-2, "basal"), (20, "split 2"), (45, "split 3")):
        ax.annotate("", xy=(x, 4.8), xytext=(x, 6.2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=S.TREAT["Control"]))
        ax.text(x, 6.4, lab, ha="center", fontsize=S.FS_MIN, color=S.TREAT["Control"])
    ax.text(21, 7.35, "nitrogen applied (control only)", fontsize=S.FS_MIN, color=S.TREAT["Control"])
    ax.scatter([0, 30, 60], [4.6] * 3, s=20, facecolor="white", edgecolor="0.25",
               zorder=5, linewidths=0.7)
    ax.plot([0, 60], [8.7, 8.7], lw=0.6, color="0.45")
    for x in (0, 30, 60):
        ax.plot([x, x], [8.55, 8.85], lw=0.6, color="0.45")
    ax.text(30, 9.05, "measurement dates", ha="center", fontsize=S.FS_MIN, color="0.4")
    ax.plot([30, 45], [1.25, 1.25], lw=1.0, color=S.TREAT["NStress"])
    for x in (30, 45):
        ax.plot([x, x], [1.08, 1.42], lw=1.0, color=S.TREAT["NStress"])
    ax.text(37.5, 0.55, "15-day correction window", ha="center", fontsize=S.FS_MIN,
            color=S.TREAT["NStress"])

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    S.panel_letter(ax, "c", dx=0, dy=2)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.plot([4.6, 4.6], [2.9, 8.8], lw=1.5, color="#6f8f4f", solid_capstyle="round")
    for i, (y, ln, col, tag) in enumerate(((8.2, 2.3, S.LEAFPOS["top"], "top leaf"),
                                           (6.8, 2.5, "#9bbf6a", None),
                                           (5.4, 2.5, "#9bbf6a", None),
                                           (3.9, 2.4, S.LEAFPOS["bottom"], "bottom leaf"))):
        side = 1 if i % 2 == 0 else -1
        ax.plot([4.6, 4.6 + side * ln], [y, y + 0.55], lw=2.0, color=col,
                solid_capstyle="round", alpha=1.0 if tag else 0.5)
        if tag:
            ax.text(4.6 + side * (ln + 0.2), y + 0.30, tag, fontsize=S.FS_MIN, color=col,
                    ha="left" if side > 0 else "right", va="center", fontweight="bold")
    ax.text(0.1, 2.15, "at each position\nCCI (MC-100)\nF$_v$/F$_m$ (FluorPen)",
            fontsize=S.FS_MIN, color="0.25", va="top")
    # The definition is a sentence, not a category, so it is black. It was set in the ratio purple
    # at the 6.0 pt floor and was the faintest text in the figure.
    ax.text(0.1, 0.50, "rCCI = CCI$_{bottom}$ / CCI$_{top}$", fontsize=S.FS_MIN,
            color=BLACK, fontweight="bold")
    save(fig, "Fig01_design_workflow.png")


# =========================================================== FIGURE 2
def fig2(gm, Fa):
    fig = plt.figure(figsize=(S.DOUBLE, 4.5))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.02], hspace=0.52, wspace=0.55,
                  left=0.085, right=0.985, top=0.90, bottom=0.09)
    pF = {r["trait"]: r for _, r in Fa.iterrows()}

    for j, (tr, lab, pos) in enumerate(FOCAL):
        ax = fig.add_subplot(gs[0, j])
        c, n = gv(gm, tr, "Control"), gv(gm, tr, "NStress")
        S.reaction_norms(ax, c, n, S.LEAFPOS[pos])
        S.half_violin(ax, -0.30, c, "left", S.TREAT["Control"], width=0.18)
        S.half_violin(ax, 1.30, n, "right", S.TREAT["NStress"], width=0.18)
        ax.set_xlim(-0.60, 1.60)
        ax.set_title(lab, color=S.LEAFPOS[pos], pad=3, fontsize=S.FS_ANNOT)
        f = pF[tr]["F_treatment"]
        pval = float(_fdist.sf(f, 1, 60))
        ax.text(0.5, -0.24, "$F$ = %s %s" % (("%.2f" % f) if f < 100 else "%.0f" % f, S.star(pval)),
                transform=ax.transAxes, ha="center", fontsize=S.FS_MIN, color=BLACK)
        if j == 0:
            ax.set_ylabel("genotype mean\n($n$ = 15 per regime)", fontsize=S.FS_MIN)
            S.panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1, 0:2])
    # The "ratio" family of this panel holds the six bottom:top ratios AND the two top-minus-bottom
    # differences dCCI and dQY. "bottom : top" would be wrong for two of the eight points, so the
    # key names the family by what it is, and the caption says which members it holds.
    _key = {"top": S.LEAFPOS_LABEL["top"], "bottom": S.LEAFPOS_LABEL["bottom"],
            "ratio": "within-plant contrast"}
    for pos in ("top", "bottom", "ratio"):
        d = Fa[Fa["position"] == pos]
        ax.scatter(d["F_genotype"], d["F_treatment"], s=14, marker=S.LEAFPOS_MARKER[pos],
                   color=S.LEAFPOS[pos], label=_key[pos],
                   edgecolor="white", linewidth=0.3, zorder=3)
    for tr, dx, dy in (("CCI_TL", -30, 7), ("CCI_BL", -16, 7), ("QY_BL", -12, 6),
                       ("rCCI", 10, -13), ("QY_TL", -26, 5)):
        r = pF[tr]
        # Two purple triangles sit at a similar height in this region, so colour cannot say which
        # point "rCCI" names. That one label gets a leader; the rest resolve by colour.
        ax.annotate(tr, (r["F_genotype"], r["F_treatment"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=S.FS_MIN, color=S.LEAFPOS[r["position"]],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.65", shrinkA=0, shrinkB=3)
                    if tr == "rCCI" else None)
    ax.axhline(4.0, ls=":", lw=0.6, color="0.55")
    ax.axvline(1.86, ls=":", lw=0.6, color="0.55")
    ax.text(0.45, 5.8, "$P$ = 0.05", fontsize=S.FS_MIN, color=BLACK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("genotype $F$"); ax.set_ylabel("nitrogen $F$")
    ax.set_xlim(0.4, 130); ax.set_ylim(0.05, 5000)
    ax.legend(loc="lower left", ncol=1, fontsize=S.FS_MIN)
    S.panel_letter(ax, "b")

    ax = fig.add_subplot(gs[1, 2:4])
    vc = pd.read_csv(os.path.join(TABLES, "variance_components.csv"))
    ypos = np.arange(len(vc))[::-1]
    left = np.zeros(len(vc))
    for k in ("pct_var_T", "pct_var_G", "pct_var_GxT", "pct_var_e"):
        ax.barh(ypos, vc[k], left=left, height=0.40, color=S.VARCOMP[k],
                hatch=S.VARCOMP_HATCH[k], label=S.VARCOMP_LABEL[k], edgecolor="white", lw=0.5)
        for yy, val, l0 in zip(ypos, vc[k], left):
            if val >= 8:
                ax.text(l0 + val / 2, yy, "%.0f" % val, ha="center", va="center",
                        fontsize=S.FS_MIN, color="white" if k == "pct_var_T" else BLACK)
            elif val >= 1.0:
                # Three thin segments sit within a few per cent of each other, so alternate the
                # callout height; a single offset made them overprint.
                _h = 10 + 9 * (list(("pct_var_T", "pct_var_G", "pct_var_GxT",
                                     "pct_var_e")).index(k) % 2)
                ax.annotate("%.1f" % val, (l0 + val / 2, yy), textcoords="offset points",
                            xytext=(0, _h), ha="center", fontsize=S.FS_MIN, color=BLACK,
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.65"))
        left += vc[k].to_numpy()
    ax.set_yticks(ypos)
    ax.set_yticklabels(["rCCI$_{30}$", "F$_v$/F$_m$\nbottom"], fontsize=S.FS_MIN)
    ax.set_xlabel("share of phenotypic variance (%), $n$ = 90 plants")
    ax.set_xlim(0, 100)
    ax.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=S.FS_MIN,
              handlelength=1.1, columnspacing=0.8, handletextpad=0.4)
    S.panel_letter(ax, "c")
    save(fig, "Fig02_signal_location.png")


# =========================================================== FIGURE 3
def fig3(gm):
    fig = plt.figure(figsize=(S.DOUBLE, 3.15))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.30], wspace=0.32,
                  left=0.085, right=0.985, top=0.88, bottom=0.30)

    ax = fig.add_subplot(gs[0, 0])
    for t in ("Control", "NStress"):
        ax.scatter(gv(gm, "CCI_TL", t), gv(gm, "CCI_BL", t), s=14, color=S.TREAT[t],
                   marker=S.TREAT_MARKER[t], label=S.TREAT_LABEL[t],
                   edgecolor="white", linewidth=0.3, zorder=3)
    lim = [90, 460]
    ax.plot(lim, lim, ls="--", lw=0.6, color="0.45", zorder=2)
    ax.text(400, 420, "1:1", fontsize=S.FS_MIN, color="0.45")
    for f in (0.9, 0.6):
        ax.plot(lim, [f * l for l in lim], ls=":", lw=0.5, color="0.7", zorder=1)
        ax.text(452, 452 * f + 10, "%.1f" % f, fontsize=S.FS_MIN, color="0.6", va="bottom")
    ax.set_xlim(lim); ax.set_ylim([60, 460])
    ax.set_xlabel("CCI, top leaf")
    ax.set_ylabel("CCI, bottom leaf")
    ax.legend(loc="upper left", fontsize=S.FS_MIN)
    S.panel_letter(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    plants = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
    plants["k"] = plants["genotype"].map(key)
    agg = plants.groupby(["k", "treatment"])["rCCI_30"].agg(["mean", "std", "count"])
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    names = gnames(gm, "rCCI", "Control")
    ks = [key(n) for n in names]
    c = np.array([agg.loc[(k, "Control"), "mean"] for k in ks])
    n = np.array([agg.loc[(k, "NStress"), "mean"] for k in ks])
    ce = np.array([agg.loc[(k, "Control"), "se"] for k in ks])
    ne = np.array([agg.loc[(k, "NStress"), "se"] for k in ks])
    o = np.argsort(c)[::-1]
    x = np.arange(len(o))
    for xi in x:
        ax.plot([xi, xi], [c[o][xi], n[o][xi]], lw=0.5, color="0.78", zorder=1)
    ax.errorbar(x, c[o], yerr=ce[o], fmt=S.TREAT_MARKER["Control"], ms=2.8, lw=0.6, capsize=1.2,
                color=S.TREAT["Control"], label=S.TREAT_LABEL["Control"])
    ax.errorbar(x, n[o], yerr=ne[o], fmt=S.TREAT_MARKER["NStress"], ms=2.8, lw=0.6, capsize=1.2,
                color=S.TREAT["NStress"], label=S.TREAT_LABEL["NStress"])
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in o], rotation=55, ha="right", fontsize=S.FS_MIN)
    ax.set_ylabel("rCCI$_{30}$  (mean ± s.e., $n$ = 3)")
    # An errorbar handle reproduces the marker with its caps, so a key inside the data field reads
    # as a datum. Marker-only proxies, placed above the axes.
    ax.legend(handles=[Line2D([], [], ls="none", marker=S.TREAT_MARKER[t], ms=2.8,
                              color=S.TREAT[t], label=S.TREAT_LABEL[t])
                       for t in ("Control", "NStress")],
              ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=S.FS_MIN)
    S.panel_letter(ax, "b")
    save(fig, "Fig03_ratio_mechanism.png")


# =========================================================== FIGURE 4
def fig4(plants):
    mdl = pd.read_csv(os.path.join(TABLES, "lda_model.csv"))
    b = {r["trait"]: r["coef"] for _, r in mdl.iterrows()}
    b0 = b.pop("(constant)")
    perf = pd.read_csv(os.path.join(TABLES, "Table3_performance.csv")).set_index("model")
    per = pd.read_csv(os.path.join(TABLES, "logocv_per_genotype.csv"))

    fig = plt.figure(figsize=(S.DOUBLE, 4.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 0.85, 0.85], hspace=0.70, wspace=0.52,
                  left=0.155, right=0.985, top=0.92, bottom=0.15)

    # The hero panel is the complementarity, which is the evidence the paper rests on.
    ax = fig.add_subplot(gs[0:2, 0])
    order = ["rCCI_30", "QY_BL_30", "both"]
    g = [disp(x) for x in per["genotype"]]
    ypos = np.arange(len(g))[::-1]
    # Offset in y, not x. Fourteen of the fifteen genotypes score 6 on Fv/Fm, and a 0.075-unit
    # x offset on a 6.8-unit axis is 2.1 pt between centres for markers 3.7 pt across: the orange
    # triangle disappeared behind the black circle. One row is 16.3 pt, so 0.28 rows clears them.
    for i, m in enumerate(order):
        ax.scatter(per[m], ypos + (1 - i) * 0.28, s=14, color=S.MODEL[m],
                   marker=S.MODEL_MARKER[m], label=S.MODEL_LABEL[m], zorder=3,
                   edgecolor="white", linewidth=0.3)
    ax.axvline(6, ls=":", lw=0.6, color="0.6")
    ax.set_yticks(ypos); ax.set_yticklabels(g, fontsize=S.FS_MIN)
    ax.set_xlabel("plants correct when that genotype\nis held out (of 6)")
    ax.set_xlim(-0.3, 6.5); ax.set_xticks([0, 2, 4, 6])
    ax.legend(loc="lower left", ncol=1, fontsize=S.FS_MIN)
    S.panel_letter(ax, "a", dx=-52)

    ax = fig.add_subplot(gs[0, 1])
    for t in ("Control", "NStress"):
        s = plants[plants["treatment"] == t]
        ax.scatter(s["rCCI_30"], s["QY_BL_30"], s=8, color=S.TREAT[t], marker=S.TREAT_MARKER[t],
                   label=S.TREAT_LABEL[t], edgecolor="white", linewidth=0.2, zorder=3)
    xs = np.linspace(0.36, 1.12, 200)
    ax.plot(xs, -(b0 + b["rCCI_30"] * xs) / b["QY_BL_30"], lw=0.8, color="0.2", zorder=2)
    ax.set_xlim(0.36, 1.12); ax.set_ylim(0.55, 0.89)
    ax.set_xlabel("rCCI$_{30}$"); ax.set_ylabel("F$_v$/F$_m$, bottom leaf")
    # The key must not sit anywhere a plant could sit: inside the axes, the teal control marker
    # landed in the low-nitrogen cloud below the boundary and read as a misclassified plant. There
    # is no empty corner large enough at this panel size, so the key goes outside.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=S.FS_MIN,
              handletextpad=0.3, columnspacing=1.0)
    ax.text(0.03, 0.97, "resubstitution fit, $n$ = 90", transform=ax.transAxes, va="top",
            fontsize=S.FS_MIN, color=BLACK)
    S.panel_letter(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    sc = plants["rCCI_30"] * b["rCCI_30"] + plants["QY_BL_30"] * b["QY_BL_30"] + b0
    y = (plants["treatment"] == "NStress").to_numpy()
    bins = np.linspace(sc.min() - 1, sc.max() + 1, 22)
    ax.hist(sc[~y], bins=bins, color=S.TREAT["Control"], alpha=0.95,
            label=S.TREAT_LABEL["Control"])
    ax.hist(sc[y], bins=bins, color=S.TREAT["NStress"], alpha=0.95,
            hatch=S.TREAT_HATCH["NStress"], label=S.TREAT_LABEL["NStress"])
    ax.axvline(0, lw=0.8, color="0.2")
    ax.set_xlabel("discriminant score $D$"); ax.set_ylabel("plants")
    ax.legend(loc="upper center", fontsize=S.FS_MIN)
    S.panel_letter(ax, "c")

    ax = fig.add_subplot(gs[1, 1:3])
    x = np.arange(3)
    acc = perf.loc[order, "accuracy"].to_numpy()
    lo = perf.loc[order, "acc_lo"].to_numpy(); hi = perf.loc[order, "acc_hi"].to_numpy()
    ax.errorbar(x, acc, yerr=[acc - lo, hi - acc], fmt="none", lw=0.8, capsize=2, color="0.35")
    for i, m in enumerate(order):
        ax.scatter([x[i]], [acc[i]], s=22, color=S.MODEL[m], marker=S.MODEL_MARKER[m],
                   zorder=3, edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, ls="--", lw=0.6, color="0.55")
    ax.text(2.42, 0.522, "majority class", fontsize=S.FS_MIN, color=BLACK, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([S.MODEL_LABEL[m] for m in order], fontsize=S.FS_MIN)
    ax.set_ylabel("pooled accuracy\n(90 held-out plants)")
    ax.set_ylim(0.42, 1.06); ax.set_xlim(-0.5, 2.5)
    ax.text(0.5, 0.05, "permutation null $P$ = 0.0005", transform=ax.transAxes, ha="center",
            fontsize=S.FS_MIN, color="0.10")
    S.panel_letter(ax, "d")
    save(fig, "Fig04_discrimination_validation.png")


# =========================================================== FIGURE 5
def fig5(plants):
    diag = pd.read_csv(os.path.join(TABLES, "diagnostics.csv")).iloc[0]
    q = plants["QY_BL_30"].to_numpy(float)
    r = plants["rCCI_30"].to_numpy(float)
    y = (plants["treatment"] == "NStress").to_numpy().astype(int)

    fig = plt.figure(figsize=(S.DOUBLE, 3.0))
    # Panel c's row labels now name the index, so they are long; the gutter has to hold them or
    # they run back into panel b's plotting region.
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.9, 1.10], wspace=0.68,
                  left=0.070, right=0.985, top=0.90, bottom=0.22)

    ax = fig.add_subplot(gs[0, 0])
    # Fv/Fm is recorded on a 0.01 grid, so bin EDGES on that grid put every value in the bin to its
    # left: the lowest control plant, 0.80, was drawn from 0.79 to 0.80, inside the shaded band the
    # caption defines as separating. Edges at the half-step centre each bar on its own value.
    bins = np.round(np.arange(0.555, 0.885, 0.01), 3)
    ax.hist(q[y == 0], bins=bins, color=S.TREAT["Control"], alpha=0.95,
            label=S.TREAT_LABEL["Control"])
    ax.hist(q[y == 1], bins=bins, color=S.TREAT["NStress"], alpha=0.95,
            hatch=S.TREAT_HATCH["NStress"], label=S.TREAT_LABEL["NStress"])
    ax.axvspan(diag["qybl_gap_lo"], diag["qybl_gap_hi"], color="0.90", zorder=0)
    ax.axvline(diag["qybl_threshold"], lw=0.9, color="0.15")
    ax.annotate("threshold %.3f\n95%% CI [%.3f, %.3f]"
                % (diag["qybl_threshold"], diag["qybl_thr_lo"], diag["qybl_thr_hi"]),
                xy=(diag["qybl_threshold"], 9.0), xytext=(0.578, 8.6), fontsize=S.FS_MIN,
                color=BLACK,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3"))
    ax.set_xlabel("F$_v$/F$_m$, bottom leaf, 30 DAT")
    ax.set_ylabel("number of plants ($n$ = 90)")
    ax.legend(loc="upper left", fontsize=S.FS_MIN)
    S.panel_letter(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    thr = np.linspace(0.55, 0.90, 400)
    sens = np.array([(q[y == 1] < t).mean() for t in thr])
    spec = np.array([(q[y == 0] >= t).mean() for t in thr])
    ax.plot(thr, sens, lw=1.0, color=S.TREAT["NStress"], label="sensitivity")
    ax.plot(thr, spec, lw=1.0, ls="--", color=S.TREAT["Control"], label="specificity")
    ok = thr[(sens == 1) & (spec == 1)]
    if len(ok):
        ax.axvspan(ok.min(), ok.max(), color="0.90", zorder=0)
    ax.set_xlabel("threshold on F$_v$/F$_m$")
    ax.set_ylabel("sensitivity / specificity")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(loc="lower left", fontsize=S.FS_MIN)
    S.panel_letter(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    imported = (r < 0.909).astype(int)
    grid = np.arange(0.40, 1.10, 0.005)
    best = grid[int(np.argmax([(((r < t).astype(int)) == y).mean() for t in grid]))]
    refit = (r < best).astype(int)
    # Name the index in every row. Unlabelled, the refitted 0.770 sat two columns from panel a's
    # "threshold 0.760" and read as a competing cut on Fv/Fm; it is a cut on rCCI.
    labels = ["Wu et al. 2024 rCCI$_{30}$\ncut-off 0.909,\nas published",
              "rCCI$_{30}$, cut-off\nrefitted here (%.3f)" % best,
              "this study, two traits,\ngenotype held out"]
    vals = [(imported == y).sum() / 90, (refit == y).sum() / 90, 1.0]
    ax.barh(np.arange(3)[::-1], vals, height=0.48,
            color=["0.78", "0.52", S.MODEL["both"]], edgecolor="white", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.015, 2 - i, "%.3f" % v, va="center", fontsize=S.FS_MIN, color=BLACK)
    ax.set_yticks(np.arange(3)[::-1]); ax.set_yticklabels(labels, fontsize=S.FS_MIN)
    ax.set_xlabel("accuracy on these 90 plants")
    ax.set_xlim(0, 1.20)
    # dx=-74 pt put this letter 4.5 mm inside panel b's frame, 24 mm from the panel it names. A
    # small offset keeps it against its own panel; the row labels start well below the axes top.
    S.panel_letter(ax, "c", dx=-10)
    save(fig, "Fig05_threshold_benchmark.png")


def main():
    bad = S.check_palette()
    if bad:
        raise SystemExit("palette self-test failed:\n  " + "\n  ".join(bad))
    plants, gm, Fa = load()
    print("figures ->", MAIN)
    for f in os.listdir(MAIN):
        if f[:5] in ("Fig01", "Fig02", "Fig03", "Fig04", "Fig05"):
            os.remove(os.path.join(MAIN, f))
    fig1()
    fig2(gm, Fa)
    fig3(gm)
    fig4(plants)
    fig5(plants)


if __name__ == "__main__":
    main()
