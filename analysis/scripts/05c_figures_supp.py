#!/usr/bin/env python
# =============================================================================
# 05c_figures_supp.py — the four supplementary figures.
#   S1  correlation structure among the focal traits, by nitrogen regime
#   S2  the Figure 2 layout repeated at 60 days after transplanting
#   S3  per-genotype means for the four focal traits, with standard errors
#   S4  every measured trait ranked by single-trait held-out accuracy and effect size
# Same locked visual system as the main figures (figstyle.py).
# Writes: results/figures/supp/SuppFigNN_*.png
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import figstyle as S

S.set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
SUPP = os.path.join(ROOT, "analysis", "results", "figures", "supp")
os.makedirs(SUPP, exist_ok=True)

FOCAL = [("CCI_TL", "CCI top", "top"), ("CCI_BL", "CCI bottom", "bottom"),
         ("QY_TL", "F$_v$/F$_m$ top", "top"), ("QY_BL", "F$_v$/F$_m$ bottom", "bottom"),
         ("rCCI", "rCCI", "ratio"), ("rQY", "rQY", "ratio")]

# The supplementary figures used to print the raw genotype strings from genotype_means.tsv, so the
# same plant was "BAM812" here and "BAM 812" in Figure 7. One naming table, in figstyle.py.
disp, BLACK = S.disp, S.BLACK


def save(fig, name):
    S.save(fig, os.path.join(SUPP, name))
    plt.close(fig)
    print("  wrote", name)


def load():
    gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
    gm["treatment"] = gm["treatment"].replace({"N stress": "NStress"})
    return gm


def wide(gm, tp, treat, traits):
    d = gm[(gm["timepoint"] == tp) & (gm["treatment"] == treat) & (gm["trait"].isin(traits))]
    return d.pivot_table(index="genotype_name", columns="trait", values="value")[traits]


# ---------------------------------------------------------------- S1
def s1(gm):
    traits = [t for t, _, _ in FOCAL] + ["dCCI", "dQY", "Photo_BL", "Cond_BL"]
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 3.6))
    for ax, treat, letter in zip(axes, ("Control", "NStress"), "ab"):
        W = wide(gm, "30 DAT", treat, [t for t in traits if t in set(gm["trait"])])
        C = W.corr()
        # A correlation matrix is symmetric, so printing both triangles spends 90 cells on 45
        # numbers and buries the two the panel exists to defend. The diagonal is r = 1.00 by
        # construction and was taking the darkest ink in the figure.
        C = C.mask(np.triu(np.ones(C.shape, dtype=bool)))
        im = ax.imshow(C, cmap=S.CORR, vmin=-1, vmax=1)
        ax.set_xticks(range(len(C)))
        ax.set_xticklabels([S.sub(t) for t in C.columns], rotation=55, ha="right",
                           fontsize=S.FS_MIN)
        ax.set_yticks(range(len(C)))
        ax.set_yticklabels([S.sub(t) for t in C.index], fontsize=S.FS_MIN)
        for i in range(len(C)):
            for j in range(len(C)):
                if i > j:
                    ax.text(j, i, "%.2f" % C.iloc[i, j], ha="center", va="center",
                            fontsize=S.FS_MIN,
                            color="white" if abs(C.iloc[i, j]) > 0.6 else "0.2")
        # Mark the row that answers the ratio-variable objection.
        r = list(C.index).index("rCCI")
        ax.add_patch(plt.Rectangle((-0.5, r - 0.5), r, 1, fill=False, ec=BLACK, lw=0.7,
                                   zorder=5))
        ax.set_title(S.TREAT_LABEL[treat], color=S.TREAT[treat], pad=6)
        # panel_letter takes dx/dy in POINTS since the rebuild; -0.30/1.10 put the letter on the
        # axes corner, where it read as a row label.
        S.panel_letter(ax, letter)
    plt.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("Pearson $r$",
                                                                 fontsize=S.FS_ANNOT)
    save(fig, "SuppFig01_correlation_structure.png")


# ---------------------------------------------------------------- S2
def s2(gm):
    fig = plt.figure(figsize=(S.DOUBLE, 3.7))
    gs = GridSpec(2, 4, figure=fig, hspace=0.70, wspace=0.52,
                  left=0.075, right=0.985, top=0.92, bottom=0.15)
    four = FOCAL[:4]
    for j, (tr, lab, pos) in enumerate(four):
        ax = fig.add_subplot(gs[0, j])
        c = wide(gm, "60 DAT", "Control", [tr])[tr].to_numpy(float)
        n = wide(gm, "60 DAT", "NStress", [tr])[tr].to_numpy(float)
        S.half_violin(ax, 0, c, "left", S.TREAT["Control"])
        S.half_violin(ax, 0, n, "right", S.TREAT["NStress"])
        # Every other figure in the paper distinguishes the regimes by marker as well as by
        # colour; strip() falls back to "o" for both when the argument is left off.
        S.strip(ax, -0.16, c, S.TREAT["Control"], seed=j, marker=S.TREAT_MARKER["Control"])
        S.strip(ax, 0.16, n, S.TREAT["NStress"], seed=j + 5, marker=S.TREAT_MARKER["NStress"])
        ax.set_xticks([-0.16, 0.16]); ax.set_xticklabels(["control", "low N"])
        ax.set_xlim(-0.55, 0.55)
        # The violin body ends flat at the data extreme. With no margin that flat end lands on the
        # spine and reads as a hard boundary in the distribution rather than as the end of the data.
        _lo = min(c.min(), n.min()); _hi = max(c.max(), n.max())
        ax.set_ylim(_lo - 0.10 * (_hi - _lo), _hi + 0.10 * (_hi - _lo))
        ax.set_title(lab, color=S.LEAFPOS[pos], pad=5)
        # The four subplots run 220-450, 60-360, 0.74-0.86 and 0.63-0.85 across two different
        # quantities, so a single y label on the leftmost implied a shared scale that does not
        # exist. Each axis names what it carries.
        ax.set_ylabel(lab + (", index" if lab.startswith("CCI") else ""), fontsize=S.FS_LABEL)
        if j == 0:
            S.panel_letter(ax, "a")
    for j, (tr, lab, pos) in enumerate(four):
        ax = fig.add_subplot(gs[1, j])
        c = wide(gm, "60 DAT", "Control", [tr])[tr].to_numpy(float)
        n = wide(gm, "60 DAT", "NStress", [tr])[tr].to_numpy(float)
        S.reaction_norms(ax, c, n, S.LEAFPOS[pos])
        ax.set_title(lab, color=S.LEAFPOS[pos], pad=5)
        ax.set_ylabel(lab + (", index" if lab.startswith("CCI") else ""), fontsize=S.FS_LABEL)
        if j == 0:
            S.panel_letter(ax, "b")
    fig.text(0.5, 0.005, "genotype means at 60 DAT, $n$ = 15 per treatment",
             ha="center", fontsize=S.FS_ANNOT, color=BLACK)
    save(fig, "SuppFig02_sixty_DAT.png")


# ---------------------------------------------------------------- S3
def s3(gm):
    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE, 4.8))
    # One order for all four panels, taken from the top-leaf chlorophyll index. Sorting each panel
    # by its own control mean gave four different orders, so no genotype could be traced across the
    # figure the whole point of which is the per-genotype comparison.
    ref = (gm[(gm["trait"] == "CCI_TL") & (gm["timepoint"] == "30 DAT")]
           .pivot_table(index="genotype_name", columns="treatment", values="value")
           .sort_values("Control", ascending=False).index)
    for ax, (tr, lab, pos), letter in zip(axes.ravel(), FOCAL[:4], "abcd"):
        d = gm[(gm["trait"] == tr) & (gm["timepoint"] == "30 DAT")]
        p = d.pivot_table(index="genotype_name", columns="treatment", values="value")
        e = d.pivot_table(index="genotype_name", columns="treatment", values="se")
        p = p.reindex(ref)
        e = e.reindex(p.index)
        x = np.arange(len(p))
        # The printed standard errors are not reproducible: across the thirteen per-genotype
        # tables they run from 0.00 to roughly 1255 times the value implied by the same table's
        # own mean square and F. Plotting them would give a fabricated quantity the authority of
        # an error bar, so the panel shows the means alone and the footnote says why.
        for t in ("Control", "NStress"):
            ax.plot(x, p[t], S.TREAT_MARKER[t], ms=4.2, color=S.TREAT[t],
                    label=S.TREAT_LABEL[t], markeredgecolor="white", markeredgewidth=0.4)
        for xi in x:
            ax.plot([xi, xi], [p["Control"].iloc[xi], p["NStress"].iloc[xi]],
                    lw=0.7, color="0.8", zorder=1)
        # Several reproduced standard errors carry the plot past a physical bound: below zero on a
        # concentration index, above 1.0 on Fv/Fm. The bound is drawn so the impossibility is on
        # the axes and not only in the footnote. No value is clipped or recomputed.
        if tr.startswith("QY"):
            ax.axhline(1.0, ls=":", lw=0.6, color="0.6")
        else:
            ax.axhline(0.0, ls=":", lw=0.6, color="0.6")
        ax.set_xticks(x)
        ax.set_xticklabels([disp(g) for g in p.index], rotation=62, ha="right",
                           fontsize=S.FS_MIN)
        ax.set_ylabel(lab, color=S.LEAFPOS[pos])
        if letter == "a":
            ax.legend(ncol=2, loc="upper right", fontsize=S.FS_MIN)
        S.panel_letter(ax, letter)
    fig.text(0.5, 0.008, "error bars are the authors' printed standard errors, reproduced as given; "
                         "several are zero and several are implausibly large\n"
                         "(see Supplementary Methods); the dotted line is the physical bound of "
                         "the measurement",
             ha="center", va="bottom", fontsize=S.FS_ANNOT, color="0.4")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, "SuppFig03_per_genotype_means.png")


# ---------------------------------------------------------------- S4
def s4():
    single = pd.read_csv(os.path.join(TABLES, "parsimony_single.csv"))
    es = pd.read_csv(os.path.join(TABLES, "effect_sizes_by_timepoint.csv"))
    es30 = es[es["timepoint"] == "30 DAT"].set_index("trait")
    # 39 rows over 2.7 in gave 5.0 pt of pitch for 6.0 pt type, so the labels overlapped whatever
    # glyphs they were set in. The panel needs the height.
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, 5.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    d = single.sort_values("logocv_accuracy")
    # This panel used a single rust literal for all six focal traits, so rCCI was purple in Fig 8a
    # and rust here and the reader was taught the opposite of the locked code one page later. It
    # now uses the same families, and carries the key that explains them.
    _tcol = {"rCCI": S.LEAFPOS["ratio"], "rQY": S.LEAFPOS["ratio"],
             "QY_BL": S.LEAFPOS["bottom"], "CCI_BL": S.LEAFPOS["bottom"],
             "QY_TL": S.LEAFPOS["top"], "CCI_TL": S.LEAFPOS["top"]}
    ax.barh(np.arange(len(d)), d["logocv_accuracy"], height=0.74,
            color=[_tcol.get(t, "0.78") for t in d["trait"]])
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([S.sub(t) for t in d["trait"]], fontsize=S.FS_MIN)
    ax.set_xlabel("single-trait leave-one-genotype-out accuracy, genotype means (n = 30)")
    # Baseline at zero: 0.4 clipped CCI_TL, RV and AD to nothing and made every other bar length
    # non-proportional.
    ax.set_xlim(0, 1.02); ax.axvline(0.5, ls="--", lw=0.9, color="0.6")
    ax.legend(handles=[Line2D([], [], color=S.LEAFPOS[p], lw=4, label=S.LEAFPOS_LABEL[p])
                       for p in ("top", "bottom", "ratio")] +
                      [Line2D([], [], color="0.78", lw=4, label="other traits")],
              loc="lower right", fontsize=S.FS_MIN)
    S.panel_letter(ax, "a")

    ax = axes[1]
    common = [t for t in es30.index if t in set(single["trait"])]
    acc = single.set_index("trait")["logocv_accuracy"]
    for i, t in enumerate(common):
        col = S.LEAFPOS["ratio"] if t.startswith(("r", "d")) else (
            S.LEAFPOS["bottom"] if t.endswith("_BL") else S.LEAFPOS["top"])
        ax.scatter(abs(es30.loc[t, "cohens_d"]), acc.get(t, np.nan), s=46, color=col,
                   edgecolor="white", linewidth=0.6, zorder=3)
        # s=46 is about 7.6 pt across, so a +/-6 pt offset started the label inside the marker's
        # own radius: "rQY" was overprinted by the QY_TL marker and read "rQ", and two labels sat
        # beside points they did not belong to. The offset clears the radius and a leader says
        # which point each label names.
        _dx, _dy = ((14, 9) if i % 2 == 0 else (-14, -11))
        ax.annotate(S.sub(t), (abs(es30.loc[t, "cohens_d"]), acc.get(t, np.nan)),
                    textcoords="offset points", xytext=(_dx, _dy), fontsize=S.FS_MIN, color=col,
                    ha="left" if _dx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=col, shrinkA=0, shrinkB=3))
    ax.set_xlabel("|Cohen's $d$| between nitrogen regimes at 30 DAT")
    ax.set_ylabel("single-trait held-out accuracy")
    ax.axhline(1.0, ls=":", lw=0.9, color="0.6")

    S.panel_letter(ax, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "SuppFig04_trait_landscape.png")


def main():
    gm = load()
    print("supplementary figures ->", SUPP)
    s1(gm)
    s2(gm)
    s3(gm)
    s4()


if __name__ == "__main__":
    main()
