#!/usr/bin/env python
"""Stage 4 — the discriminant function, its cut-off, and the assumption tests the source omits.

The source thesis and all three manuscript drafts report a two-trait model with 100% accuracy but give
no coefficients, no constant and no threshold anywhere. A reader cannot compute the score. This script
produces the missing object and the diagnostics a reviewer will ask for:

  * Fisher linear discriminant on the two focal traits, with the explicit equation and cut-off
  * Wilks' lambda with its chi-square approximation and p
  * Box's M test of equal covariance matrices (the assumption LDA is stated to rest on)
  * canonical correlation and the proportion of between-class variance explained
  * standardised (within-class) coefficients with percentile bootstrap intervals
  * the single-trait threshold with its interval, obtained by bootstrap over plants

Outputs: results/tables/Table4_discriminant.csv, lda_model.csv, diagnostics.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
os.makedirs(TABLES, exist_ok=True)
TRAITS = ["rCCI_30", "QY_BL_30"]
RNG = np.random.default_rng(20260803)
NBOOT = 5000


def box_m(X, y):
    """Box's M test for homogeneity of covariance matrices, with the Box chi-square approximation."""
    groups = [X[y == g] for g in np.unique(y)]
    p = X.shape[1]
    k = len(groups)
    ns = np.array([len(g) for g in groups])
    nu = ns - 1
    Ss = [np.cov(g, rowvar=False, ddof=1) for g in groups]
    Sp = sum(v * S for v, S in zip(nu, Ss)) / nu.sum()
    M = nu.sum() * np.log(np.linalg.det(Sp)) - sum(v * np.log(np.linalg.det(S)) for v, S in zip(nu, Ss))
    c1 = (sum(1 / nu) - 1 / nu.sum()) * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    dfree = (k - 1) * p * (p + 1) / 2
    chi2 = M * (1 - c1)
    return M, chi2, dfree, stats.chi2.sf(chi2, dfree)


def wilks(X, y):
    """Wilks' lambda for a two-group problem, with Bartlett's chi-square approximation."""
    n, p = X.shape
    k = len(np.unique(y))
    grand = X.mean(axis=0)
    W = np.zeros((p, p))
    B = np.zeros((p, p))
    for g in np.unique(y):
        Xg = X[y == g]
        d = (Xg.mean(axis=0) - grand).reshape(-1, 1)
        B += len(Xg) * d @ d.T
        W += (Xg - Xg.mean(axis=0)).T @ (Xg - Xg.mean(axis=0))
    lam = np.linalg.det(W) / np.linalg.det(W + B)
    chi2 = -(n - 1 - (p + k) / 2) * np.log(lam)
    dfree = p * (k - 1)
    canon = np.sqrt(1 - lam)          # two groups: one canonical function
    return lam, chi2, dfree, stats.chi2.sf(chi2, dfree), canon


def fit(X, y):
    m = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True).fit(X, y)
    return m.coef_.ravel(), float(m.intercept_[0]), m


def main():
    df = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
    X = df[TRAITS].to_numpy(float)
    y = (df["treatment"] == "NStress").astype(int).to_numpy()

    b, b0, model = fit(X, y)
    Sw = model.covariance_
    sd_w = np.sqrt(np.diag(Sw))
    b_std = b * sd_w                                    # within-class standardised coefficients

    lam, wchi, wdf, wp, canon = wilks(X, y)
    M, bchi, bdf, bp = box_m(X, y)

    # bootstrap: coefficients, and the single-trait threshold on QY_BL_30
    boot_b, boot_std, boot_thr = [], [], []
    q = df["QY_BL_30"].to_numpy(float)
    for _ in range(NBOOT):
        idx = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        bb, _, mm = fit(X[idx], y[idx])
        boot_b.append(bb)
        boot_std.append(bb * np.sqrt(np.diag(mm.covariance_)))
        qs, ys = q[idx], y[idx]
        grid = np.arange(0.60, 0.90, 0.001)
        acc = [(((qs < t).astype(int)) == ys).mean() for t in grid]
        boot_thr.append(grid[int(np.argmax(acc))])
    boot_b = np.array(boot_b); boot_std = np.array(boot_std); boot_thr = np.array(boot_thr)

    # decision rule: score > 0 => low nitrogen
    score = X @ b + b0
    err = int(((score > 0).astype(int) != y).sum())

    print("DISCRIMINANT FUNCTION  (resubstitution; class 1 = low nitrogen)\n")
    eq = "  D = %.4f  %+.4f x rCCI_30  %+.4f x QY_BL_30" % (b0, b[0], b[1])
    print(eq)
    print("  classify as low-nitrogen when D > 0        misclassified: %d / %d" % (err, len(y)))
    print("\n  standardised (within-class) coefficients, 95%% percentile bootstrap (B=%d)" % len(boot_b))
    for i, t in enumerate(TRAITS):
        lo, hi = np.percentile(boot_std[:, i], [2.5, 97.5])
        print("    %-10s %8.3f   [%7.3f, %7.3f]" % (t, b_std[i], lo, hi))

    print("\nASSUMPTION TESTS AND FIT DIAGNOSTICS")
    print("  Wilks' lambda            %.5f   chi2 = %.1f, df = %d, p = %.3e" % (lam, wchi, wdf, wp))
    print("  canonical correlation    %.4f   (%.1f%% of variance between classes)" % (canon, 100 * canon ** 2))
    print("  Box's M                  %.2f    chi2 = %.1f, df = %.0f, p = %.3e" % (M, bchi, bdf, bp))
    print("     %s" % ("covariances differ; report and note LDA robustness at balanced n"
                       if bp < 0.05 else "no evidence against equal covariances"))

    thr_lo, thr_hi = np.percentile(boot_thr, [2.5, 97.5])
    grid = np.arange(0.60, 0.90, 0.001)
    acc = [(((q < t).astype(int)) == y).mean() for t in grid]
    best = grid[int(np.argmax(acc))]
    gap_lo, gap_hi = q[y == 1].max(), q[y == 0].min()
    print("\nSINGLE-TRAIT RULE ON QY_BL_30")
    print("  optimal threshold        %.3f   95%% bootstrap CI [%.3f, %.3f]" % (best, thr_lo, thr_hi))
    print("  separating interval      (%.3f, %.3f)   any value in it classifies all 90 correctly"
          % (gap_lo, gap_hi))

    pd.DataFrame([{
        "term": "constant", "coefficient": b0, "standardised": np.nan,
        "boot_lo": np.nan, "boot_hi": np.nan}] + [{
        "term": t, "coefficient": b[i], "standardised": b_std[i],
        "boot_lo": np.percentile(boot_std[:, i], 2.5),
        "boot_hi": np.percentile(boot_std[:, i], 97.5)} for i, t in enumerate(TRAITS)]
    ).to_csv(os.path.join(TABLES, "Table4_discriminant.csv"), index=False, float_format="%.10g")

    pd.DataFrame([{
        "wilks_lambda": lam, "wilks_chi2": wchi, "wilks_df": wdf, "wilks_p": wp,
        "canonical_r": canon, "canonical_r2_pct": 100 * canon ** 2,
        "box_M": M, "box_chi2": bchi, "box_df": bdf, "box_p": bp,
        "resub_misclassified": err, "n": len(y),
        "qybl_threshold": best, "qybl_thr_lo": thr_lo, "qybl_thr_hi": thr_hi,
        "qybl_gap_lo": gap_lo, "qybl_gap_hi": gap_hi,
        "n_bootstrap": len(boot_b)}]
    ).to_csv(os.path.join(TABLES, "diagnostics.csv"), index=False, float_format="%.6g")

    # The coefficients are the paper's deliverable: a reader must be able to reproduce the
    # boundary from the table. %.6g truncated 145.3104 to 145.310 and left the legend disagreeing
    # with the body at the fourth decimal, so this file carries full double precision.
    pd.DataFrame([{"trait": t, "coef": b[i], "coef_std": b_std[i]} for i, t in enumerate(TRAITS)]
                 + [{"trait": "(constant)", "coef": b0, "coef_std": np.nan}]
                 ).to_csv(os.path.join(TABLES, "lda_model.csv"), index=False, float_format="%.10g")
    print("\nwritten: Table4_discriminant.csv, diagnostics.csv, lda_model.csv")


if __name__ == "__main__":
    main()
