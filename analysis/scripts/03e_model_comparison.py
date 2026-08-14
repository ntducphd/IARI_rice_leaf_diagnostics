#!/usr/bin/env python
"""Stage 3e — fixed 10-model comparison under leave-one-genotype-out CV.

Chotted model list (2026-08-03), agreed with the supervisor's request to "finalise the model
count and run the comparison": 2 single-trait threshold baselines (one re-optimised per fold,
one the published Wu et al. 2024 cutoff applied unchanged) + 8 two-trait classifiers spanning
the linear/quadratic/regularised/naive-Bayes/margin-based families discussed as candidates for
this small-n (90 plants, 15 genotypes), 2-trait, genotype-held-out classification problem.
A 10th model, linear SVM, was added on 2026-08-03 to check whether a margin-based classifier
joins the group of models that reach 90/90 with identical per-plant predictions.

Every model is evaluated under the SAME leave-one-genotype-out scheme (15 folds, 6 held-out
plants per fold, 90 pooled out-of-fold predictions), so the comparison is paired. The optimised
threshold models are refit inside each fold (cutoff chosen on the 14 training genotypes only)
to avoid the leakage that a whole-dataset threshold search would introduce -- this differs from
(and is stricter than) 03_lda_logocv.py's whole-data threshold search, which is retained
separately as a descriptive/exploratory number, not as this comparison's baseline.
"""
import os
import csv
import numpy as np
from scipy.stats import beta
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived"))
rows = list(csv.DictReader(open(os.path.join(D, "plants_30DAT.tsv")), delimiter="\t"))
G = sorted({r["genotype"] for r in rows})
gid = np.array([r["genotype"] for r in rows])
y = np.array([1 if r["treatment"] == "NStress" else 0 for r in rows])
rcci = np.array([float(r["rCCI_30"]) for r in rows])
qybl = np.array([float(r["QY_BL_30"]) for r in rows])
X = np.column_stack([rcci, qybl])
N = len(y)
assert N == 90 and len(G) == 15


def cp(k, n, a=.05):
    lo = 0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def logo_classifier(make):
    """LOGO-CV for an sklearn-style classifier fit on both traits."""
    pred = np.empty(N, dtype=int)
    for g in G:
        te = gid == g
        tr = ~te
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return pred


def logo_threshold(values, direction):
    """LOGO-CV for a single-trait threshold optimised on the training genotypes only
    (cutoff re-searched inside every fold; no test-fold information is used to pick it)."""
    pred = np.empty(N, dtype=int)
    grid = np.linspace(values.min(), values.max(), 400)
    for g in G:
        te = gid == g
        tr = ~te
        v_tr, y_tr = values[tr], y[tr]
        if direction == "low_is_class1":
            scored = [(((v_tr < t).astype(int) == y_tr).sum(), t) for t in grid]
        else:
            scored = [(((v_tr > t).astype(int) == y_tr).sum(), t) for t in grid]
        _, best_t = max(scored, key=lambda z: z[0])
        if direction == "low_is_class1":
            pred[te] = (values[te] < best_t).astype(int)
        else:
            pred[te] = (values[te] > best_t).astype(int)
    return pred


def logo_fixed_threshold(values, cutoff, direction):
    """No fitting at all -- a literature-published cutoff applied unchanged."""
    if direction == "low_is_class1":
        return (values < cutoff).astype(int)
    return (values > cutoff).astype(int)


MODELS = []

# 1-3: single-trait threshold baselines
MODELS.append(("QY_BL_30 threshold (fit per fold)", lambda: logo_threshold(qybl, "low_is_class1")))
MODELS.append(("rCCI_30 Wu-2024 fixed cutoff 0.909", lambda: logo_fixed_threshold(rcci, 0.909, "low_is_class1")))
MODELS.append(("rCCI_30 threshold (fit per fold)", lambda: logo_threshold(rcci, "low_is_class1")))

# 4-9: two-trait classifiers, all on [rCCI_30, QY_BL_30]
MODELS.append(("LDA", lambda: logo_classifier(lambda: LDA())))
MODELS.append(("Logistic regression", lambda: logo_classifier(lambda: LogisticRegression(max_iter=5000, penalty=None))))
MODELS.append(("Ridge logistic (L2, C=1.0)", lambda: logo_classifier(lambda: LogisticRegression(max_iter=5000, penalty="l2", C=1.0))))
MODELS.append(("QDA", lambda: logo_classifier(lambda: QDA(reg_param=0.0))))
MODELS.append(("Shrinkage LDA (RDA proxy, lsqr+auto)", lambda: logo_classifier(lambda: LDA(solver="lsqr", shrinkage="auto"))))
MODELS.append(("Gaussian Naive Bayes", lambda: logo_classifier(lambda: GaussianNB())))
# Linear SVM needs its features standardised (unlike LDA/QDA/NB/logistic here, which are on the
# raw [rCCI_30, QY_BL_30] scale already close in range); wrap in a pipeline so the LOGO refit
# fits the scaler on the training genotypes only, same no-leakage discipline as the thresholds.
MODELS.append(("Linear SVM (C=1.0)", lambda: logo_classifier(lambda: make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=10000)))))

print("FIXED 10-MODEL COMPARISON UNDER LEAVE-ONE-GENOTYPE-OUT CV")
print("15 folds, 6 held-out plants per fold, 90 pooled out-of-fold predictions\n")
print("%-38s %8s %9s %20s %8s %8s" % ("model", "correct", "accuracy", "95% CI (Clopper-Pearson)", "sens", "spec"))

results = {}
ref_pred = None
rows_out = []
for name, fn in MODELS:
    p = fn()
    k = int((p == y).sum())
    lo, hi = cp(k, N)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn_ = int(((p == 0) & (y == 1)).sum())
    sens = tp / (tp + fn_) if (tp + fn_) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    if ref_pred is None:
        ref_pred = p  # first model (QY_BL_30 threshold) as arbitrary agreement reference is not useful;
    results[name] = p
    print("%-38s %5d/%-3d %9.4f   [%.3f, %.3f]        %6.3f %8.3f" % (name, k, N, k / N, lo, hi, sens, spec))
    folds_perfect = sum(1 for g in G if (p[gid == g] == y[gid == g]).all())
    flo, fhi = cp(folds_perfect, 15)
    rows_out.append(dict(model=name, correct=k, n=N, accuracy=round(k / N, 6),
                          acc_ci_lo=round(lo, 6), acc_ci_hi=round(hi, 6),
                          sensitivity=round(sens, 6), specificity=round(spec, 6),
                          TP=tp, TN=tn, FP=fp, FN=fn_,
                          genotype_folds_perfect=folds_perfect, n_folds=15,
                          fold_ci_lo=round(flo, 6), fold_ci_hi=round(fhi, 6)))

# pairwise agreement matrix (which models predict identically on all 90 plants)
print("\nPAIRWISE AGREEMENT (out of 90 pooled predictions)")
names = [m[0] for m in MODELS]
header = "%-38s" % ""
for n in names:
    header += " %6s" % (n[:6])
print(header)
for a in names:
    line = "%-38s" % a
    for b in names:
        agree = int((results[a] == results[b]).sum())
        line += " %6d" % agree
    print(line)

# McNemar exact test: LDA (current primary) vs Ridge logistic (recommended primary candidate)
from scipy.stats import binomtest
p_lda = results["LDA"]
p_ridge = results["Ridge logistic (L2, C=1.0)"]
disagree_lda_right = int(((p_lda == y) & (p_ridge != y)).sum())
disagree_ridge_right = int(((p_ridge == y) & (p_lda != y)).sum())
n_disagree = disagree_lda_right + disagree_ridge_right
if n_disagree > 0:
    mc_p = binomtest(min(disagree_lda_right, disagree_ridge_right), n_disagree, 0.5).pvalue
else:
    mc_p = 1.0
print("\nMcNEMAR (exact, two-sided): LDA vs Ridge logistic")
print("  LDA-only-correct=%d  Ridge-only-correct=%d  n_discordant=%d  P=%.4f"
      % (disagree_lda_right, disagree_ridge_right, n_disagree, mc_p))

# ---- export ----
import pandas as pd
T = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables"))
os.makedirs(T, exist_ok=True)
out_path = os.path.join(T, "model_comparison_10.csv")
pd.DataFrame(rows_out).to_csv(out_path, index=False, float_format="%.6g")
print("\nwrote", out_path)

agree_rows = []
for a in names:
    for b in names:
        agree_rows.append(dict(model_a=a, model_b=b, agree_n=int((results[a] == results[b]).sum())))
agree_path = os.path.join(T, "model_comparison_10_agreement.csv")
pd.DataFrame(agree_rows).to_csv(agree_path, index=False)
print("wrote", agree_path)

mcnemar_path = os.path.join(T, "model_comparison_10_mcnemar_lda_vs_ridge.csv")
pd.DataFrame([dict(model_a="LDA", model_b="Ridge logistic (L2, C=1.0)",
                    lda_only_correct=disagree_lda_right, ridge_only_correct=disagree_ridge_right,
                    n_discordant=n_disagree, mcnemar_exact_p=round(mc_p, 6))]).to_csv(mcnemar_path, index=False)
print("wrote", mcnemar_path)
