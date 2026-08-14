"""Robustness checks raised in review: paired model comparison, recording-precision
sensitivity, decision margins, and within-treatment genotype variance.

Inputs : analysis/results/tables/SuppTable_S5_plant_level_30DAT.csv
Outputs: analysis/results/tables/robustness_checks.csv
         analysis/results/tables/within_treatment_genotype.csv

Every quantity written here is derived from the same 90-plant matrix that
03_lda_logocv.py and 04_diagnostics.py use. Seed fixed at 20260803.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SEED = 20260803
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

# S5 now carries commented provenance lines, so the comment character has to be declared or the
# parser reads them as data rows and this whole script dies before writing anything.
df = pd.read_csv(TABLES / "SuppTable_S5_plant_level_30DAT.csv", comment="#")
y = (df["treatment"] == "NStress").astype(int).to_numpy()
g = df["genotype"].to_numpy()


def logocv(X, y, g):
    """Pooled leave-one-genotype-out predictions."""
    pred = np.empty_like(y)
    for held in np.unique(g):
        m = g == held
        lda = LinearDiscriminantAnalysis().fit(X[~m], y[~m])
        pred[m] = lda.predict(X[m])
    return pred


X_both = df[["rCCI_30", "QY_BL_30"]].to_numpy()
X_qy = df[["QY_BL_30"]].to_numpy()
X_rcci = df[["rCCI_30"]].to_numpy()

p_both = logocv(X_both, y, g)
p_qy = logocv(X_qy, y, g)
p_rcci = logocv(X_rcci, y, g)


def mcnemar_exact(a, b, y):
    """Exact two-sided McNemar on paired correctness of two rules."""
    ca, cb = a == y, b == y
    n01 = int(np.sum(~ca & cb))  # a wrong, b right
    n10 = int(np.sum(ca & ~cb))  # a right, b wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * stats.binom.cdf(k, n, 0.5))
    return n01, n10, p


rows = []

b01, b10, p_mc = mcnemar_exact(p_qy, p_both, y)
rows.append(
    dict(check="mcnemar_QY_BL_vs_pair", stat_1=b01, stat_2=b10, value=p_mc,
         detail="discordant plants: QY_BL wrong and pair right / pair wrong and QY_BL right; exact two-sided P")
)
r01, r10, p_mr = mcnemar_exact(p_rcci, p_both, y)
rows.append(
    dict(check="mcnemar_rCCI_vs_pair", stat_1=r01, stat_2=r10, value=p_mr,
         detail="discordant plants; exact two-sided P")
)

# Decision margins against the published boundary, expressed in Fv/Fm units.
D = 145.3104 - 33.9622 * df["rCCI_30"] - 160.3115 * df["QY_BL_30"]
margin_qy = (D.abs() / 160.3115).to_numpy()
order = np.argsort(margin_qy)
rows.append(dict(check="min_margin_two_trait_Fv_Fm", stat_1=int(order[0]), stat_2=np.nan,
                 value=round(float(margin_qy[order[0]]), 4),
                 detail=f"nearest plant {df.genotype[order[0]]} {df.treatment[order[0]]}; next two "
                        f"{margin_qy[order[1]]:.4f}, {margin_qy[order[2]]:.4f}"))

# Single-threshold rule at 0.78, the midpoint of the indifference set.
margin_single = np.abs(df["QY_BL_30"] - 0.78).to_numpy()
rows.append(dict(check="min_margin_single_cut_0.78", stat_1=np.nan, stat_2=np.nan,
                 value=round(float(margin_single.min()), 4),
                 detail="smallest distance of any plant to a single cut on QY_BL_30 at 0.78"))

# Distinct recorded values per class.
for name, sub in [("control", df[df.treatment == "Control"]), ("lowN", df[df.treatment == "NStress"])]:
    rows.append(dict(check=f"distinct_QY_values_{name}", stat_1=sub.QY_BL_30.nunique(), stat_2=np.nan,
                     value=sub.QY_BL_30.nunique(),
                     detail=f"{sub.QY_BL_30.min():.2f}-{sub.QY_BL_30.max():.2f}"))

# Recording-precision sensitivity: uniform jitter of the size 2-dp rounding implies.
rng = np.random.default_rng(SEED)
for amp in (0.005, 0.010):
    perfect_pair = perfect_qy = 0
    correct_pair = []
    draws = 1000
    # Counting how often each rule reaches 90 of 90 is not a robustness comparison, because the two
    # rules do not start level: with no jitter at all the pair scores 90 and QY_BL_30 alone scores 88.
    # The single arm's 0% therefore follows from its baseline, not from noise sensitivity, and quoting
    # "72.0% against none" as a robustness result overstates what these draws show. What the draws can
    # support is how far each rule falls FROM ITS OWN baseline, so both are recorded here.
    correct_qy = []
    base_pair = int((logocv(X_both, y, g) == y).sum())
    base_qy = int((logocv(X_both[:, [1]], y, g) == y).sum())
    for _ in range(draws):
        Xj = X_both + rng.uniform(-amp, amp, size=X_both.shape)
        pj = logocv(Xj, y, g)
        n_ok = int((pj == y).sum())
        correct_pair.append(n_ok)
        perfect_pair += n_ok == 90
        qj = logocv(Xj[:, [1]], y, g)
        n_qy = int((qj == y).sum())
        correct_qy.append(n_qy)
        perfect_qy += n_qy == 90
    rows.append(dict(check=f"jitter_{amp}_pair_pct_perfect", stat_1=draws, stat_2=perfect_pair,
                     value=round(100 * perfect_pair / draws, 1),
                     detail=f"mean correct {np.mean(correct_pair):.2f} of 90, baseline {base_pair}"))
    rows.append(dict(check=f"jitter_{amp}_QY_BL_pct_perfect", stat_1=draws, stat_2=perfect_qy,
                     value=round(100 * perfect_qy / draws, 1),
                     detail=f"single-trait rule, same draws; mean correct {np.mean(correct_qy):.2f} "
                            f"of 90, baseline {base_qy}"))
    rows.append(dict(check=f"jitter_{amp}_loss_from_own_baseline", stat_1=round(base_pair - np.mean(correct_pair), 3),
                     stat_2=round(base_qy - np.mean(correct_qy), 3),
                     value=round((base_qy - np.mean(correct_qy)) - (base_pair - np.mean(correct_pair)), 3),
                     detail="plants lost on average: pair, then single, then the difference"))

# Re-optimised threshold on rCCI_30, to separate calibration from index in the Wu benchmark.
best = max(
    ((t, int(((df.rCCI_30 <= t).astype(int) == y).sum())) for t in np.arange(0.40, 1.10, 0.005)),
    key=lambda z: z[1],
)
imported = int(((df.rCCI_30 < 0.909).astype(int) == y).sum())
rows.append(dict(check="rCCI_threshold_imported_0.909", stat_1=imported, stat_2=90,
                 value=round(imported / 90, 4), detail="Wu et al. cut-off applied unchanged"))
rows.append(dict(check="rCCI_threshold_refitted", stat_1=best[1], stat_2=90,
                 value=round(best[0], 3),
                 detail=f"same index, cut-off re-optimised on these data: {best[1]} of 90 correct"))

pd.DataFrame(rows).to_csv(TABLES / "robustness_checks.csv", index=False)

# Within-treatment genotype effect: one-way ANOVA on genotype inside each class.
wt = []
for trait in ("rCCI_30", "QY_BL_30"):
    for name, key in (("control", "Control"), ("lowN", "NStress")):
        sub = df[df.treatment == key]
        groups = [v[trait].to_numpy() for _, v in sub.groupby("genotype")]
        F, p = stats.f_oneway(*groups)
        k, n = len(groups), 3
        # intraclass correlation from the one-way random model
        icc = (F - 1) / (F - 1 + n)
        means = sub.groupby("genotype")[trait].mean()
        wt.append(dict(trait=trait, treatment=name, df_G=k - 1, df_e=k * (n - 1),
                       F_genotype=round(float(F), 3), p_genotype=float(p),
                       icc_genotype=round(float(icc), 3),
                       genotype_mean_min=round(float(means.min()), 3),
                       genotype_mean_max=round(float(means.max()), 3),
                       genotype_mean_range=round(float(means.max() - means.min()), 3)))
wtd = pd.DataFrame(wt)
for trait in ("rCCI_30", "QY_BL_30"):
    shift = (df[df.treatment == "Control"][trait].mean() - df[df.treatment == "NStress"][trait].mean())
    wtd.loc[wtd.trait == trait, "treatment_shift"] = round(float(shift), 3)
wtd.to_csv(TABLES / "within_treatment_genotype.csv", index=False)

print(pd.DataFrame(rows).to_string(index=False))
print()
print(wtd.to_string(index=False))
