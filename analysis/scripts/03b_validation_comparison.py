#!/usr/bin/env python
"""Stage 3b — the analyses that Figures 6 to 10 rest on.

Four questions the main pipeline does not answer:
  1. how the separation develops between 30 and 60 days after transplanting
  2. whether the rule works equally in tolerant and susceptible genotypes
  3. how few traits are enough, and which pairs separate at all
  4. how much a random train/test split flatters a model relative to holding a genotype out

Question 4 is the paper's methodological contribution and needs the comparison made explicitly:
the same models, the same data, scored both ways.

Writes: results/tables/{effect_sizes_by_timepoint,tolerance_strata,parsimony_single,
        parsimony_scan,pairwise_separation,split_comparison,bootstrap_draws,permutation_null}.csv
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
os.makedirs(TABLES, exist_ok=True)
RNG = np.random.default_rng(20260803)


def norm(x):
    k = "".join(str(x).upper().split()).replace("-", "").replace("(", "").replace(")", "")
    return {"MOROBEREKAN": "MOROBERAKAN", "RPW9SS1": "RPW94SS1"}.get(k, k)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def overlap(a, b):
    """Fraction of the pooled range in which the two classes overlap; 0 means disjoint."""
    lo, hi = max(min(a), min(b)), min(max(a), max(b))
    if hi <= lo:
        return 0.0
    return (hi - lo) / (max(max(a), max(b)) - min(min(a), min(b)))


def main():
    gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
    gm["treatment"] = gm["treatment"].replace({"N stress": "NStress"})
    gm["k"] = gm["genotype_name"].map(norm)
    plants = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
    plants["k"] = plants["genotype"].map(norm)
    tol = pd.read_csv(os.path.join(DERIVED, "genotype_tolerance.csv"))
    tol["k"] = tol["genotype_name"].map(norm)
    tmap = dict(zip(tol["k"], tol["tolerance_class"]))

    # ---- 1. effect size and overlap by trait and timepoint --------------------
    rows = []
    for tp in ("30 DAT", "60 DAT"):
        for tr in ("CCI_TL", "CCI_BL", "QY_TL", "QY_BL", "rCCI", "rQY", "dCCI", "dQY"):
            d = gm[(gm["trait"] == tr) & (gm["timepoint"] == tp)]
            c = d[d["treatment"] == "Control"]["value"].to_numpy(float)
            n = d[d["treatment"] == "NStress"]["value"].to_numpy(float)
            if len(c) < 3 or len(n) < 3:
                continue
            t, p = stats.ttest_ind(c, n, equal_var=False)
            rows.append(dict(trait=tr, timepoint=tp, mean_control=c.mean(), mean_lowN=n.mean(),
                             cohens_d=cohens_d(c, n), overlap=overlap(c, n), t=t, p=p))
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "effect_sizes_by_timepoint.csv"),
                              index=False, float_format="%.6g")
    print("  effect_sizes_by_timepoint.csv   %d rows" % len(rows))

    # ---- 2. performance within tolerance strata ------------------------------
    X = plants[["rCCI_30", "QY_BL_30"]].to_numpy(float)
    y = (plants["treatment"] == "NStress").astype(int).to_numpy()
    gid = plants["k"].to_numpy()
    G = sorted(set(gid))
    pred = np.empty(len(y), int)
    for g in G:
        te = gid == g
        pred[te] = LDA().fit(X[~te], y[~te]).predict(X[te])
    score = LDA().fit(X, y).decision_function(X)
    strata = []
    for g in G:
        m = gid == g
        strata.append(dict(genotype=g, tolerance=tmap.get(g, "unknown"),
                           correct=int((pred[m] == y[m]).sum()), n=int(m.sum()),
                           margin_control=float(score[m & (y == 0)].mean()),
                           margin_lowN=float(score[m & (y == 1)].mean())))
    st = pd.DataFrame(strata)
    st.to_csv(os.path.join(TABLES, "tolerance_strata.csv"), index=False, float_format="%.6g")
    print("  tolerance_strata.csv            %d genotypes" % len(st))
    print(st.groupby("tolerance")[["correct", "n"]].sum().to_string())

    # ---- 3. parsimony and pairwise separation, at genotype level -------------
    g30 = gm[gm["timepoint"] == "30 DAT"].pivot_table(index=["k", "treatment"],
                                                      columns="trait", values="value").dropna(axis=1)
    traits = list(g30.columns)
    Xg = g30.to_numpy(float)
    yg = (g30.index.get_level_values("treatment") == "NStress").astype(int)
    gg = np.array(g30.index.get_level_values("k"))

    def logo_g(cols):
        M = Xg[:, cols]
        p = np.empty(len(yg), int)
        for g in np.unique(gg):
            te = gg == g
            if len(np.unique(yg[~te])) < 2:
                return np.nan
            p[te] = LDA().fit(M[~te], yg[~te]).predict(M[te])
        return float((p == yg).mean())

    single = sorted(((logo_g([i]), t) for i, t in enumerate(traits)), reverse=True)
    pd.DataFrame([{"trait": t, "logocv_accuracy": a} for a, t in single]).to_csv(
        os.path.join(TABLES, "parsimony_single.csv"), index=False, float_format="%.6g")

    idx = {t: i for i, t in enumerate(traits)}
    focal = [t for t in ("rCCI", "QY_BL", "CCI_TL", "CCI_BL", "QY_TL", "rQY", "dCCI", "dQY")
             if t in idx]
    pw = [dict(trait_a=a, trait_b=b, logocv_accuracy=logo_g([idx[a], idx[b]]))
          for a, b in itertools.combinations(focal, 2)]
    pd.DataFrame(pw).to_csv(os.path.join(TABLES, "pairwise_separation.csv"),
                            index=False, float_format="%.6g")
    print("  parsimony_single.csv            %d traits" % len(single))
    print("  pairwise_separation.csv         %d pairs" % len(pw))

    chosen, curve, remaining = [], [], list(range(len(traits)))
    for k in range(1, 9):
        best = max(remaining, key=lambda i: (logo_g(chosen + [i]), -i))
        chosen.append(best); remaining.remove(best)
        curve.append(dict(n_traits=k, added=traits[best], logocv_accuracy=logo_g(chosen)))
    pd.DataFrame(curve).to_csv(os.path.join(TABLES, "parsimony_scan.csv"),
                               index=False, float_format="%.6g")
    print("  parsimony_scan.csv              8 steps")

    # ---- 4. random split versus genotype held out ---------------------------
    comp = []
    for name, cols in (("rCCI_30", [0]), ("QY_BL_30", [1]), ("both", [0, 1])):
        M = X[:, cols]
        accs = []
        for tr_i, te_i in StratifiedShuffleSplit(n_splits=500, test_size=18,
                                                 random_state=7).split(M, y):
            accs.append((LDA().fit(M[tr_i], y[tr_i]).predict(M[te_i]) == y[te_i]).mean())
        accs = np.array(accs)
        p = np.empty(len(y), int)
        for g in G:
            te = gid == g
            p[te] = LDA().fit(M[~te], y[~te]).predict(M[te])
        comp.append(dict(model=name, random_mean=accs.mean(),
                         random_lo=np.percentile(accs, 2.5), random_hi=np.percentile(accs, 97.5),
                         random_pct_perfect=100 * (accs == 1).mean(),
                         logocv=float((p == y).mean())))
    cd = pd.DataFrame(comp)
    cd.to_csv(os.path.join(TABLES, "split_comparison.csv"), index=False, float_format="%.6g")
    print("  split_comparison.csv")
    print(cd.to_string(index=False))

    # ---- 5. bootstrap and permutation distributions -------------------------
    draws = []
    for _ in range(5000):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = LDA(solver="lsqr", store_covariance=True).fit(X[i], y[i])
        b = m.coef_.ravel() * np.sqrt(np.diag(m.covariance_))
        q, yy = X[i, 1], y[i]
        grid = np.arange(0.60, 0.90, 0.002)
        thr = grid[int(np.argmax([(((q < t).astype(int)) == yy).mean() for t in grid]))]
        draws.append(dict(b_rCCI=b[0], b_QYBL=b[1], threshold=thr))
    pd.DataFrame(draws).to_csv(os.path.join(TABLES, "bootstrap_draws.csv"),
                               index=False, float_format="%.6g")

    null = []
    for _ in range(2000):
        yp = RNG.permutation(y)
        p = np.empty(len(y), int)
        ok = True
        for g in G:
            te = gid == g
            if len(np.unique(yp[~te])) < 2:
                ok = False
                break
            p[te] = LDA().fit(X[~te], yp[~te]).predict(X[te])
        if ok:
            null.append(float((p == yp).mean()))
    pd.DataFrame({"accuracy": null}).to_csv(os.path.join(TABLES, "permutation_null.csv"),
                                            index=False, float_format="%.6g")
    print("  bootstrap_draws.csv             %d draws" % len(draws))
    print("  permutation_null.csv            %d permutations" % len(null))


if __name__ == "__main__":
    main()
