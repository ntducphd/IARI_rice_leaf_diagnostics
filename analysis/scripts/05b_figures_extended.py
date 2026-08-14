#!/usr/bin/env python
# =============================================================================
# 05b_figures_extended.py — main Figures 6 to 10, at journal print size.
#
#   Fig  6  when the signal appears, and why 30 DAT   effect size, overlap, 30 vs 60 DAT with paired t
#   Fig  7  does it hold across tolerance classes     per-class test, per-class accuracy, per-genotype margin
#   Fig  8  which traits, and how many                single-trait ranking, pairs, the saturation caveat
#   Fig  9  diagnostics and robustness                bootstraps, permutation null, three classifiers
#   Fig 10  why one split cannot decide               500 random splits against genotype held out
#
# Rebuilt 2026-08-03 after the figure audits and a further review. Four things are fixed throughout:
#   * canvas is 180 mm, so the type ladder in figstyle.py is legible at reproduction scale
#   * every number printed on a plot is black; colour is reserved for category, never for a value
#   * no legend or annotation sits over data — legends are placed outside the plotted region
#   * every panel that compares two groups prints the test that compares them, from group_tests.csv
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import beta
import figstyle as S

S.set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
MAIN = os.path.join(ROOT, "analysis", "results", "figures", "main")
os.makedirs(MAIN, exist_ok=True)

BLACK = "0.10"                       # every printed value uses this, never a category colour
TOL_COL = {"susceptible": "#8c8c8c", "moderately tolerant": "#c0c0c0", "tolerant": "#4a4a4a"}
TOL_ORDER = ["susceptible", "moderately tolerant", "tolerant"]
TOL_SHORT = {"susceptible": "susceptible", "moderately tolerant": "moderately\ntolerant",
             "tolerant": "tolerant"}

# One naming table, in figstyle.py, shared with 05_figures.py and 05c_figures_supp.py.
key, disp = S.key, S.disp


def save(fig, name):
    S.save(fig, os.path.join(MAIN, name))
    plt.close(fig)
    print("  wrote", name)


def T(name):
    return pd.read_csv(os.path.join(TABLES, name))


def cp(k, n, a=.05):
    return (0 if k == 0 else beta.ppf(a / 2, k, n - k + 1),
            1 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))


def pfmt(p):
    return "$P$ < 0.001" if p < 1e-3 else "$P$ = %.3f" % p


# =========================================================== FIGURE 6
def fig6(gm):
    es = T("effect_sizes_by_timepoint.csv")
    gt = T("group_tests.csv")
    gt6 = gt[gt.panel == "Fig6"]
    fig = plt.figure(figsize=(S.DOUBLE, 4.3))
    # right=0.90 left 0.71 in for panel c's outside legend and clipped "low nitrogen" to
    # "low nitroger" at the canvas edge.
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 0.95], hspace=0.72, wspace=0.85,
                  left=0.085, right=0.855, top=0.90, bottom=0.13)

    order = ["CCI_TL", "CCI_BL", "QY_TL", "QY_BL", "rCCI", "rQY", "dCCI", "dQY"]
    pos = {"CCI_TL": "top", "QY_TL": "top", "CCI_BL": "bottom", "QY_BL": "bottom",
           "rCCI": "ratio", "rQY": "ratio", "dCCI": "ratio", "dQY": "ratio"}

    ax = fig.add_subplot(gs[0, 0:3])
    pts = []
    for tr in order:
        a = es[(es.trait == tr) & (es.timepoint == "30 DAT")]["cohens_d"].abs().values
        b = es[(es.trait == tr) & (es.timepoint == "60 DAT")]["cohens_d"].abs().values
        if len(a) and len(b):
            pts.append((tr, float(a[0]), float(b[0])))

    def declutter(vals, gap):
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        out = list(vals)
        for n, i in enumerate(idx):
            if n and out[i] - out[idx[n - 1]] < gap:
                out[i] = out[idx[n - 1]] + gap
        return out

    # The gap has to exceed one line box, not one line. Panel a is 1.22 in over ~7.25 data units,
    # i.e. 12.1 pt per unit, and 6.0 pt type sets on a ~7.0 pt line box: 0.34 units was 4.1 pt and
    # the labels still overlapped by about 40%. 0.62 units is 7.5 pt.
    la = declutter([q[1] for q in pts], 0.62)
    lb = declutter([q[2] for q in pts], 0.62)
    for (tr, a, b), ya, yb in zip(pts, la, lb):
        col = S.LEAFPOS[pos[tr]]
        ax.plot([0, 1], [a, b], marker=S.LEAFPOS_MARKER[pos[tr]], ms=3.2, lw=1.0, color=col)
        ax.plot([-0.055, 0], [ya, a], lw=0.4, color=col, alpha=0.5)
        ax.plot([1, 1.055], [b, yb], lw=0.4, color=col, alpha=0.5)
        ax.text(-0.075, ya, tr, ha="right", va="center", fontsize=S.FS_MIN, color=col)
        ax.text(1.075, yb, tr, ha="left", va="center", fontsize=S.FS_MIN, color=col)
    ax.axhline(0.8, ls=":", lw=0.6, color="0.6")
    # x = 0.5 put this note across the rising CCI_TL line, which crosses |d| = 0.95 at x = 0.65.
    ax.text(0.25, 0.95, "large effect, |$d$| = 0.8", ha="center", fontsize=S.FS_MIN, color=BLACK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["30 DAT", "60 DAT"])
    ax.set_xlim(-0.62, 1.62)
    _all = [q[1] for q in pts] + [q[2] for q in pts] + list(la) + list(lb)
    ax.set_ylim(min(_all) - 0.40, max(_all) + 0.40)
    ax.set_ylabel("|Cohen's $d$| between regimes\n(genotype means, $n$ = 15)")
    S.panel_letter(ax, "a")

    ax = fig.add_subplot(gs[0, 3:5])
    w = 0.38
    x = np.arange(len(order))
    for i, tp in enumerate(("30 DAT", "60 DAT")):
        v = [es[(es.trait == t) & (es.timepoint == tp)]["overlap"].values for t in order]
        v = [q[0] if len(q) else np.nan for q in v]
        ax.bar(x + (i - 0.5) * w, v, w, label=tp, color=S.TIME[tp], hatch=S.TIME_HATCH[tp],
               edgecolor="white", lw=0.4)
        # Zero is the headline of this panel and a bar of length zero draws no ink, so four of the
        # eight columns read as missing measurements. Print the zero.
        for xi, vi in zip(x + (i - 0.5) * w, v):
            if np.isfinite(vi) and vi == 0:
                ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=S.FS_MIN, color=BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels([S.sub(t) for t in order], rotation=55, ha="right", fontsize=S.FS_MIN)
    ax.set_ylabel("class overlap\n(0 = disjoint)")
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=S.FS_MIN)
    S.panel_letter(ax, "b")

    ax = fig.add_subplot(gs[1, 0:5])
    xpos, ticks, labs = 0.0, [], []
    span, tmax = {}, {}
    for tr, trlab in (("rCCI", "rCCI"), ("QY_BL", "F$_v$/F$_m$ bottom leaf")):
        x0 = xpos
        for tp in ("30 DAT", "60 DAT"):
            for t in ("Control", "NStress"):
                d = gm[(gm.trait == tr) & (gm.timepoint == tp) & (gm.treatment == t)]
                v = d["value"].to_numpy(float)
                if not len(v):
                    continue
                off = -0.15 if t == "Control" else 0.15
                S.strip(ax, xpos + off, v, S.TREAT[t], jitter=0.045, size=6,
                        seed=int(xpos * 7), marker=S.TREAT_MARKER[t])
                ax.plot([xpos + off - 0.11, xpos + off + 0.11], [v.mean()] * 2, lw=1.2,
                        color=S.TREAT[t])
                tmax[tr] = max(tmax.get(tr, -np.inf), float(v.max()))
            ticks.append(xpos); labs.append(tp)
            xpos += 1.0
        span[tr] = (x0, xpos - 1.0)
        xpos += 0.55
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=S.FS_MIN)
    ax.set_ylabel("genotype mean")
    for xc, lab, col in ((0.5, "rCCI", S.LEAFPOS["ratio"]),
                         (3.05, "F$_v$/F$_m$, bottom leaf", S.LEAFPOS["bottom"])):
        ax.text(xc, -0.19, lab, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=S.FS_ANNOT, color=col, fontweight="bold")
    ax.axvline(2.28, ls=":", lw=0.6, color="0.8")
    # group_tests.csv carries the paired t comparing 30 with 60 DAT for each trait and regime.
    # These eight rows were loaded and thrown away by a dead loop, so the only panel of this figure
    # that compares two groups printed no test at all.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.26 * (hi - lo))
    for tr, (xa, xb) in span.items():
        rc = gt6[(gt6.trait == tr) & (gt6.group == "Control")].iloc[0]
        rn = gt6[(gt6.trait == tr) & (gt6.group == "NStress")].iloc[0]
        ax.text((xa + xb) / 2, tmax[tr] + 0.05 * (hi - lo),
                "paired $t$, 30 vs 60 DAT ($n$ = 15)\ncontrol %s;   low nitrogen %s"
                % (pfmt(rc.p), pfmt(rn.p)),
                ha="center", va="bottom", fontsize=S.FS_MIN, color=BLACK)
    ax.legend(handles=[Line2D([], [], color=S.TREAT[t], marker=S.TREAT_MARKER[t], ls="",
                              ms=3.5, label=S.TREAT_LABEL[t]) for t in ("Control", "NStress")],
              loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=S.FS_MIN)
    S.panel_letter(ax, "c")
    save(fig, "Fig06_temporal_development.png")


# =========================================================== FIGURE 7
def fig7(plants):
    st = T("tolerance_strata.csv")
    gt = T("group_tests.csv")
    gt7 = gt[gt.panel == "Fig7a"]
    tol = pd.read_csv(os.path.join(DERIVED, "genotype_tolerance.csv"))
    tol["k"] = tol["genotype_name"].map(key)
    tmap = dict(zip(tol["k"], tol["tolerance_class"]))
    plants = plants.copy()
    plants["k"] = plants["genotype"].map(key)
    plants["tol"] = plants["k"].map(tmap)

    fig = plt.figure(figsize=(S.DOUBLE, 4.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.94], hspace=0.78, wspace=0.55,
                  left=0.085, right=0.845, top=0.92, bottom=0.20)

    for j, (tr, lab) in enumerate((("rCCI_30", "rCCI$_{30}$"),
                                   ("QY_BL_30", "F$_v$/F$_m$, bottom leaf"))):
        ax = fig.add_subplot(gs[0, j])
        top = plants[tr].max()
        rng = plants[tr].max() - plants[tr].min()
        for i, cl in enumerate(TOL_ORDER):
            for t in ("Control", "NStress"):
                v = plants[(plants["tol"] == cl) & (plants["treatment"] == t)][tr]
                off = -0.16 if t == "Control" else 0.16
                S.strip(ax, i + off, v, S.TREAT[t], size=6, seed=i * 3,
                        marker=S.TREAT_MARKER[t], jitter=0.045)
                ax.plot([i + off - 0.12, i + off + 0.12], [v.mean()] * 2, lw=1.3, color=S.TREAT[t])
            row = gt7[(gt7.trait == tr) & (gt7.group == cl)].iloc[0]
            # group_tests.csv also holds a Mann-Whitney P that differs by up to eight orders of
            # magnitude, so which test produced the printed P is load-bearing. The test is named
            # once above the panel; three categories share 38 pt each, which will not hold it
            # three times over.
            ax.text(i, top + 0.11 * rng,
                    "%s\n$d$ = %.1f\n$n$ = %d/%d"
                    % (pfmt(row.p), row.cohens_d, row.n_control, row.n_lowN),
                    ha="center", va="bottom", fontsize=S.FS_MIN, color=BLACK)
        ax.text(0.5, 1.005, "Welch $t$ within each class", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=S.FS_MIN, color=BLACK)
        ax.set_ylim(plants[tr].min() - 0.06 * rng, top + 0.46 * rng)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["susceptible", "mod. tolerant", "tolerant"], fontsize=S.FS_MIN)
        ax.set_ylabel(lab)
        ax.set_xlabel("tolerance class  (9 / 4 / 2 genotypes)", fontsize=S.FS_MIN)
        S.panel_letter(ax, "a" if j == 0 else "b")
        if j == 0:
            # "lower left" put an oversized orange triangle inside the susceptible low-nitrogen
            # cloud at a plausible data position. The key goes below the axes.
            ax.legend(handles=[Line2D([], [], color=S.TREAT[t], marker=S.TREAT_MARKER[t], ls="",
                                      ms=3.5, label=S.TREAT_LABEL[t])
                               for t in ("Control", "NStress")],
                      loc="lower center", bbox_to_anchor=(0.5, -0.46), ncol=2, fontsize=S.FS_MIN,
                      handletextpad=0.3, columnspacing=1.0)

    ax = fig.add_subplot(gs[0, 2])
    agg = st.groupby("tolerance")[["correct", "n"]].sum().reindex(TOL_ORDER)
    for i, cl in enumerate(TOL_ORDER):
        k, n = int(agg.loc[cl, "correct"]), int(agg.loc[cl, "n"])
        lo, hi = cp(k, n)
        ax.errorbar([i], [k / n], yerr=[[k / n - lo], [hi - k / n]], fmt="o", ms=5,
                    lw=1.0, capsize=2.5, color=TOL_COL[cl],
                    markeredgecolor=BLACK, markeredgewidth=0.5)
        ax.text(i, 0.445, "%d/%d" % (k, n), ha="center", fontsize=S.FS_MIN, color=BLACK)
    ax.axhline(0.5, ls="--", lw=0.6, color="0.6")
    ax.text(2.45, 0.515, "majority class", fontsize=S.FS_MIN, color=BLACK, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["susc.", "mod. tol.", "tol."], fontsize=S.FS_MIN)
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0.42, 1.06); ax.set_xlim(-0.6, 2.6)
    ax.set_xlabel("tolerance class", fontsize=S.FS_MIN)
    S.panel_letter(ax, "c")

    ax = fig.add_subplot(gs[1, 0:3])
    st2 = st.copy()
    st2["tol"] = pd.Categorical(st2["tolerance"], TOL_ORDER, ordered=True)
    st2 = st2.sort_values(["tol", "margin_lowN"])
    x = np.arange(len(st2))
    ax.bar(x - 0.19, st2["margin_control"], 0.36, color=S.TREAT["Control"],
           label=S.TREAT_LABEL["Control"], edgecolor="white", lw=0.3)
    ax.bar(x + 0.19, st2["margin_lowN"], 0.36, color=S.TREAT["NStress"],
           hatch=S.TREAT_HATCH["NStress"], label=S.TREAT_LABEL["NStress"],
           edgecolor="white", lw=0.3)
    ax.axhline(0, lw=0.8, color=BLACK)
    lo = min(st2["margin_control"].min(), st2["margin_lowN"].min())
    hi = max(st2["margin_control"].max(), st2["margin_lowN"].max())
    ax.set_ylim(lo - 0.22 * (hi - lo), hi + 0.10 * (hi - lo))
    base = lo - 0.15 * (hi - lo)
    for i, cl in enumerate(st2["tolerance"]):
        ax.plot([x[i] - 0.42, x[i] + 0.42], [base] * 2, lw=2.4, color=TOL_COL[cl],
                solid_capstyle="butt")
    ax.set_xticks(x)
    ax.set_xticklabels([disp(g) for g in st2["genotype"]], rotation=55, ha="right",
                       fontsize=S.FS_MIN)
    ax.set_ylabel("mean discriminant score $D$")
    hand = ([Line2D([], [], color=S.TREAT[t], lw=4, label=S.TREAT_LABEL[t])
             for t in ("Control", "NStress")] +
            [Line2D([], [], color=TOL_COL[c], lw=4, label=c) for c in TOL_ORDER])
    ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(1.005, 1.05), fontsize=S.FS_MIN)
    S.panel_letter(ax, "d")
    save(fig, "Fig07_tolerance_classes.png")


# =========================================================== FIGURE 8
def fig8():
    single = T("parsimony_single.csv")
    pw = T("pairwise_separation.csv")
    perf = T("Table3_performance.csv").set_index("model")
    rob = T("robustness_checks.csv") if os.path.exists(
        os.path.join(TABLES, "robustness_checks.csv")) else None

    fig = plt.figure(figsize=(S.DOUBLE, 4.6))
    # right=0.985 pushed the colourbar off the canvas: its tick labels were cut to "1.", "0.", and
    # the "pair accuracy" label never appeared at all.
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.85], hspace=0.72, wspace=0.62,
                  left=0.115, right=0.90, top=0.93, bottom=0.24)

    ax = fig.add_subplot(gs[0:2, 0:2])
    d = single.sort_values("logocv_accuracy")
    focal = {"rCCI", "QY_BL", "CCI_TL", "CCI_BL", "QY_TL", "rQY"}
    # Colour by what the trait IS, not by a whitelist of the six the paper happens to use. The
    # hardcoded map that stood here coloured rCCI and rQY purple under a key reading "bottom : top"
    # while leaving rPhoto, rTr, rCond, rCi, rLW and rLL grey — six traits that are also
    # bottom-to-top ratios. A reader seeing rPhoto grey directly above rQY purple would conclude
    # rPhoto is not a ratio. The key stated a rule the figure did not follow.
    def _leafpos(t):
        if t.endswith("_BL"):
            return S.LEAFPOS["bottom"]
        if t.endswith("_TL"):
            return S.LEAFPOS["top"]
        if t.startswith("r") and len(t) > 1 and t[1].isupper():
            return S.LEAFPOS["ratio"]
        return "0.82"          # whole-plant traits, and the bottom-minus-top differences
    ax.barh(np.arange(len(d)), d["logocv_accuracy"], height=0.72,
            color=[_leafpos(t) for t in d["trait"]], edgecolor="white", lw=0.3)
    ax.set_yticks(np.arange(len(d)))
    # The underscore of _TL and _BL sits on the baseline and struck through the label of the row
    # beneath it on a 39-row axis. Subscripts, as everywhere else in the paper.
    ax.set_yticklabels([S.sub(t) for t in d["trait"]], fontsize=S.FS_MIN)
    ax.set_xlabel("single-trait held-out accuracy\n(genotype means, $n$ = 30)")
    # A bar baseline at 0.4 clipped CCI_TL (0.200), RV (0.333) and AD (0.400) to nothing, so three
    # labelled rows carried no bar and read as missing data. The baseline is zero.
    ax.set_xlim(0, 1.02)
    ax.axvline(0.5, ls="--", lw=0.6, color="0.6")
    ax.legend(handles=[Line2D([], [], color=S.LEAFPOS[p], lw=4, label=S.LEAFPOS_LABEL[p])
                       for p in ("top", "bottom", "ratio")] +
                      [Line2D([], [], color="0.82", lw=4, label="whole plant, or a difference")],
              loc="lower right", fontsize=S.FS_MIN)
    S.panel_letter(ax, "a", dx=-44)

    ax = fig.add_subplot(gs[0, 2:4])
    tr = ["rCCI", "QY_BL", "QY_TL", "CCI_BL", "CCI_TL", "rQY", "dCCI", "dQY"]
    M = np.full((len(tr), len(tr)), np.nan)
    for _, r in pw.iterrows():
        if r.trait_a in tr and r.trait_b in tr:
            i, j = tr.index(r.trait_a), tr.index(r.trait_b)
            M[i, j] = M[j, i] = r.logocv_accuracy
    # aspect="auto": with square cells the matrix was sized by the panel HEIGHT, giving 11.3 pt of
    # cell width for a 12 pt string, so the printed coefficients ran into each other.
    im = ax.imshow(M, cmap="viridis", vmin=0.82, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tr)))
    ax.set_xticklabels([S.sub(t) for t in tr], rotation=55, ha="right", fontsize=S.FS_MIN)
    ax.set_yticks(range(len(tr)))
    ax.set_yticklabels([S.sub(t) for t in tr], fontsize=S.FS_MIN)
    for i in range(len(tr)):
        for j in range(len(tr)):
            if np.isfinite(M[i, j]):
                # The luminance crossover for viridis under vmin 0.82 / vmax 1.00 is at M = 0.898,
                # so the switch belongs there, not at 0.95: white on viridis(0.63) is 2.78:1.
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=S.FS_MIN,
                        color="white" if M[i, j] < 0.90 else BLACK)
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.set_label("pair accuracy", fontsize=S.FS_MIN)
    cb.ax.tick_params(labelsize=S.FS_MIN, colors=BLACK)
    S.panel_letter(ax, "b")

    ax = fig.add_subplot(gs[1, 2:4])
    lab = ["rCCI$_{30}$", "F$_v$/F$_m$", "both"]
    _s = single.set_index("trait")["logocv_accuracy"]
    _both = pw[((pw.trait_a == "rCCI") & (pw.trait_b == "QY_BL")) |
               ((pw.trait_a == "QY_BL") & (pw.trait_b == "rCCI"))]
    gl = [float(_s.get("rCCI", np.nan)), float(_s.get("QY_BL", np.nan)),
          float(_both["logocv_accuracy"].iloc[0]) if len(_both) else np.nan]
    pl = [perf.loc["rCCI_30", "accuracy"], perf.loc["QY_BL_30", "accuracy"],
          perf.loc["both", "accuracy"]]
    # Bars on a baseline of 0.90 asked the reader to weigh 0.933 against 0.922 by length, when the
    # gap is 0.3 of a genotype mean. Points with exact intervals, and a y range that holds them.
    x = np.arange(3)
    lo_g, hi_g = zip(*[cp(int(round(a * 30)), 30) for a in gl])
    lo_p, hi_p = zip(*[cp(int(round(b * 90)), 90) for b in pl])
    ax.errorbar(x - 0.22, gl, yerr=[np.array(gl) - lo_g, np.array(hi_g) - np.array(gl)],
                fmt="o", ms=4, mfc="white", mec="0.30", ecolor="0.30", lw=0.9, capsize=2,
                label="genotype means ($n$ = 30)")
    ax.errorbar(x + 0.22, pl, yerr=[np.array(pl) - lo_p, np.array(hi_p) - np.array(pl)],
                fmt="s", ms=4, color="0.30", ecolor="0.30", lw=0.9, capsize=2,
                label="individual plants ($n$ = 90)")
    for xi in range(3):
        ax.text(xi - 0.22, hi_g[xi] + 0.006, "%.3f" % gl[xi], ha="center", va="bottom",
                fontsize=S.FS_MIN, color=BLACK)
        ax.text(xi + 0.22, hi_p[xi] + 0.006, "%.3f" % pl[xi], ha="center", va="bottom",
                fontsize=S.FS_MIN, color=BLACK)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=S.FS_MIN)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(min(lo_g + lo_p) - 0.02, 1.045)
    ax.set_ylabel("held-out accuracy\n(exact 95% interval)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=1, fontsize=S.FS_MIN)
    # The xlabel was a claim sentence, which the figure plan forbids inside an image. The claim
    # stands in the Figure 8 legend, where it belongs.
    ax.set_xlabel("decision rule", fontsize=S.FS_MIN)
    S.panel_letter(ax, "c")
    save(fig, "Fig08_trait_selection.png")


# =========================================================== FIGURE 9
def fig9():
    bd = T("bootstrap_draws.csv")
    nul = T("permutation_null.csv")["accuracy"].to_numpy(float)
    diag = T("diagnostics.csv").iloc[0]

    # Height cut from 4.1 after panel f was deleted and e widened: the canvas kept the six-panel
    # height, leaving about an eighth of the figure blank below the second row.
    fig = plt.figure(figsize=(S.DOUBLE, 3.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.86], hspace=0.72, wspace=0.50,
                  left=0.085, right=0.985, top=0.92, bottom=0.15)

    for j, (col, lab, pos) in enumerate((("b_rCCI", "rCCI$_{30}$", "ratio"),
                                         ("b_QYBL", "F$_v$/F$_m$, bottom leaf", "bottom"))):
        ax = fig.add_subplot(gs[0, j])
        v = bd[col].to_numpy(float)
        ax.hist(v, bins=44, color=S.LEAFPOS[pos], edgecolor="none")
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.axvline(0, lw=0.8, color=BLACK)
        for b in (lo, hi):
            ax.axvline(b, lw=0.6, ls=":", color=BLACK)
        ax.set_xlabel("standardised coefficient")
        ax.set_ylabel("bootstrap draws" if j == 0 else "")
        ax.set_title(lab, pad=3, fontsize=S.FS_ANNOT, color=S.LEAFPOS[pos])
        # The dotted percentile lines ran through the closing bracket of this string. A white
        # ground under the text means a reference line can never cross a glyph.
        ax.text(0.03, 0.96, "95%% CI [%.2f, %.2f]" % (lo, hi), transform=ax.transAxes,
                va="top", fontsize=S.FS_MIN, color=BLACK,
                bbox=dict(fc="white", ec="none", pad=0.8))
        S.panel_letter(ax, "ab"[j])

    ax = fig.add_subplot(gs[0, 2])
    # The bootstrap threshold takes four values and nothing between them. A histogram drew an empty
    # bin at 0.74 that reads as a gap in a continuous distribution rather than as a value the 0.01
    # recording grid never visits. A stem over the observed support says what the data are.
    v = bd["threshold"].to_numpy(float)
    vals, cnts = np.unique(v, return_counts=True)
    ax.vlines(vals, 0, cnts, lw=3, color=S.LEAFPOS["bottom"])
    ax.plot([diag["qybl_threshold"]], [cnts.max() * 1.06], marker="v", ms=4, color=BLACK)
    ax.set_ylim(0, cnts.max() * 1.34)
    ax.set_xlabel("optimal threshold on F$_v$/F$_m$")
    ax.set_ylabel("bootstrap draws")
    ax.text(0.03, 0.96, "%.3f\n95%% CI [%.3f, %.3f]"
            % (diag["qybl_threshold"], diag["qybl_thr_lo"], diag["qybl_thr_hi"]),
            transform=ax.transAxes, va="top", fontsize=S.FS_MIN, color=BLACK,
            bbox=dict(fc="white", ec="none", pad=0.8))
    S.panel_letter(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(nul, bins=np.arange(0.25, 0.85, 0.022), color=S.NEUTRAL, edgecolor="none")
    ax.axvline(1.0, lw=1.2, color=BLACK)
    ax.set_xlim(0.25, 1.06)
    ax.set_xlabel("accuracy under permuted labels")
    ax.set_ylabel("permutations")
    # Right-aligned, this block sat on the observed-accuracy line at 1.000. The upper-left quadrant
    # is clear of the null distribution, which peaks near 0.5 at 60% of the panel height.
    ax.text(0.03, 0.96, "observed 1.000\n%d permutations\n%s"
            % (len(nul), pfmt((np.sum(nul >= 1.0) + 1) / (len(nul) + 1))),
            transform=ax.transAxes, va="top", ha="left", fontsize=S.FS_MIN, color=BLACK)
    S.panel_letter(ax, "d")

    # The assumption-test panel that stood in gs[1, 2] was a four-row table set as text, with an
    # interpretive sentence inside the image. Every quantity it held is in Results and in
    # diagnostics.csv, and the Figure 9 legend keeps the Box's M caveat. The cell goes to panel e,
    # which can then carry its exact intervals.
    ax = fig.add_subplot(gs[1, 1:3])
    names = ["linear", "quadratic", "logistic"]
    corr = [90, 90, 86]
    acc = [c / 90 for c in corr]
    lo_e, hi_e = zip(*[cp(c, 90) for c in corr])
    # Colour encoded nothing here — the three bars are labelled — and the hexes it borrowed meant
    # "both traits" and "30/60 DAT" elsewhere in the same paper.
    ax.errorbar(np.arange(3), acc, yerr=[np.array(acc) - lo_e, np.array(hi_e) - np.array(acc)],
                fmt="o", ms=4.5, color="0.30", ecolor="0.30", lw=0.9, capsize=2.5)
    for xi, c in enumerate(corr):
        ax.text(xi, hi_e[xi] + 0.004, "%d/90" % c, ha="center", va="bottom",
                fontsize=S.FS_MIN, color=BLACK)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=S.FS_MIN)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(min(lo_e) - 0.02, 1.02)
    ax.set_ylabel("held-out accuracy\n(exact 95% interval)")
    ax.set_xlabel("discriminant / regression", fontsize=S.FS_MIN)
    S.panel_letter(ax, "e")
    save(fig, "Fig09_diagnostics_robustness.png")


# =========================================================== FIGURE 10
def fig10():
    sc = T("split_comparison.csv").set_index("model")
    fig = plt.figure(figsize=(S.DOUBLE, 4.1))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=1.00, wspace=0.44,
                  left=0.155, right=0.985, top=0.92, bottom=0.14)
    order = ["rCCI_30", "QY_BL_30", "both"]

    ax = fig.add_subplot(gs[0, 0])
    for i, m in enumerate(order):
        r = sc.loc[m]
        ax.plot([r["random_lo"], r["random_hi"]], [i, i], lw=5, color=S.NEUTRAL_LIGHT,
                solid_capstyle="butt", zorder=1)
        # The "both" rule has random_lo = random_hi = 1.000, so the range bar draws no ink and the
        # row that carries the headline comparison rendered as a bare black marker. A degenerate
        # range gets a tick, and the grey mean sits above the black mark rather than under it.
        if r["random_hi"] - r["random_lo"] < 0.004:
            ax.plot([r["random_lo"]], [i], marker="|", ms=14, mew=3.0, color=S.NEUTRAL_LIGHT,
                    zorder=1)
        ax.plot([r["random_mean"]], [i + 0.18], marker="o", ms=4, color="0.45", zorder=3)
        ax.plot([r["logocv"]], [i], marker=S.MODEL_MARKER[m], ms=6, color=BLACK, zorder=4)
    ax.set_yticks(range(3))
    ax.set_yticklabels([S.MODEL_LABEL[m] for m in order], fontsize=S.FS_MIN)
    ax.set_xlabel("accuracy"); ax.set_xlim(0.80, 1.015); ax.set_ylim(-0.7, 2.7)
    # The held-out mark is a square, a triangle and a circle; a single circle in the key
    # misdescribed two of the three marks it claimed to explain. All three go in one handle. The
    # key also sat on the data, so it moves below the axes.
    _hout = tuple(Line2D([], [], color=BLACK, marker=S.MODEL_MARKER[m], ls="", ms=4.5)
                  for m in order)
    ax.legend(handles=[Line2D([], [], color=S.NEUTRAL_LIGHT, lw=5),
                       Line2D([], [], color="0.45", marker="o", ls="", ms=4), _hout],
              labels=["500 random 80:20 splits, 95% range", "mean of those splits",
                      "genotype held out (shape names the rule)"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
              loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=1, fontsize=S.FS_MIN,
              handlelength=3.2)
    S.panel_letter(ax, "a", dx=-58)

    ax = fig.add_subplot(gs[0, 1])
    pct = [sc.loc[m, "random_pct_perfect"] for m in order]
    ax.bar(np.arange(3), pct, 0.5, color=[S.MODEL[m] for m in order],
           edgecolor="white", lw=0.4)
    for i, p in enumerate(pct):
        ax.text(i, p + 2.5, "%.1f%%" % p, ha="center", fontsize=S.FS_MIN, color=BLACK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["rCCI$_{30}$", "F$_v$/F$_m$", "both"], fontsize=S.FS_MIN)
    ax.set_ylabel("random splits reporting\na perfect score (%)")
    ax.set_ylim(0, 118)
    S.panel_letter(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    from scipy.stats import binom
    truths = np.arange(0.70, 1.001, 0.005)
    ax.plot(truths, binom.pmf(18, 18, truths), lw=1.4, color=S.NEUTRAL)
    for t, lab in ((0.89, "0.89"), (0.95, "0.95")):
        p = binom.pmf(18, 18, t)
        ax.plot([t], [p], marker="o", ms=4, color=BLACK)
        ax.annotate("true accuracy %s\nscores 18/18 in %.0f%%" % (lab, 100 * p),
                    xy=(t, p), xytext=(0.715, 0.92 if t == 0.89 else 0.66),
                    textcoords="axes fraction", fontsize=S.FS_MIN, color=BLACK,
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.45"))
    ax.set_xlabel("true accuracy of the rule")
    ax.set_ylabel("probability that an 18-plant\ntest set scores 18 of 18")
    ax.set_ylim(0, 1.16); ax.set_xlim(0.70, 1.005)
    S.panel_letter(ax, "c", dx=-58)

    ax = fig.add_subplot(gs[1, 1])
    # The curve used to plot the width below 1.0, so the marker sat at 0.218 under a label reading
    # "lower bound 0.78", and the curve fell as the evidence strengthened. Plot the bound itself.
    ns = np.arange(3, 31)
    ax.plot(ns, [cp(n, n)[0] for n in ns], lw=1.4, color=S.NEUTRAL)
    ax.axvline(15, ls="--", lw=0.7, color=BLACK)
    lo15 = cp(15, 15)[0]
    ax.plot([15], [lo15], marker="o", ms=4.5, color=BLACK)
    ax.annotate("15 genotypes, all correct:\nlower bound %.2f" % lo15,
                xy=(15, lo15), xytext=(0.40, 0.28), textcoords="axes fraction",
                fontsize=S.FS_MIN, color=BLACK,
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.45"))
    ax.set_xlabel("genotypes held out, all classified correctly")
    ax.set_ylabel("lower bound of the exact\n95% interval")
    S.panel_letter(ax, "d")
    save(fig, "Fig10_validation_design.png")


def main():
    bad = S.check_palette()
    if bad:
        raise SystemExit("palette self-test failed:\n  " + "\n  ".join(bad))
    gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
    gm["treatment"] = gm["treatment"].replace({"N stress": "NStress"})
    plants = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
    print("extended figures ->", MAIN)
    fig6(gm)
    fig7(plants)
    fig8()
    fig9()
    fig10()


if __name__ == "__main__":
    main()
