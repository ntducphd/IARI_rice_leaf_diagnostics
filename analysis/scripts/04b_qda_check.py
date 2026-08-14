#!/usr/bin/env python
"""Stage 4b — robustness of the linear rule to the violated equal-covariance assumption.

Box's M rejects equality of the two class covariance matrices (p = 1.5e-12). Linear discriminant
analysis is known to be robust to this at equal class sizes, but the claim must be checked rather
than asserted. This refits the same leave-one-genotype-out scheme with a quadratic discriminant,
which does not assume equal covariances, and with logistic regression, which assumes neither
normality nor equal covariance.
"""
import os, csv
import numpy as np
from scipy.stats import beta
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression

D = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived"))
rows = list(csv.DictReader(open(os.path.join(D, "plants_30DAT.tsv")), delimiter="\t"))
G = sorted({r["genotype"] for r in rows})
gid = np.array([r["genotype"] for r in rows])
y = np.array([1 if r["treatment"] == "NStress" else 0 for r in rows])
X = np.array([[float(r["rCCI_30"]), float(r["QY_BL_30"])] for r in rows])


def cp(k, n, a=.05):
    return (0 if k == 0 else beta.ppf(a / 2, k, n - k + 1),
            1 if k == n else beta.ppf(1 - a / 2, k + 1, n - k))


def logo(make):
    pred = np.empty(len(y), dtype=int)
    for g in G:
        te = gid == g
        pred[te] = make().fit(X[~te], y[~te]).predict(X[te])
    return pred


print("ROBUSTNESS TO THE EQUAL-COVARIANCE ASSUMPTION")
print("leave-one-genotype-out, both traits, 90 pooled predictions\n")
print("%-34s %8s %9s %20s" % ("classifier", "correct", "accuracy", "95% CI"))
# This script used to print and write nothing. analysis/results/tables/qda_check.csv existed as an
# orphan from an earlier version, carried no logistic row, and no script regenerated it, so the
# Results claim that logistic regression reached 86 of 90 had no file behind it. It does now.
out = []
ref = logo(lambda: LDA())
for name, make in (("linear discriminant", lambda: LDA()),
                   ("quadratic discriminant", lambda: QDA(reg_param=0.0)),
                   ("logistic regression", lambda: LogisticRegression(max_iter=5000))):
    p = logo(make)
    k = int((p == y).sum())
    lo, hi = cp(k, 90)
    agree = int((p == ref).sum())
    print("%-34s %5d/90 %9.4f      [%.3f, %.3f]   agrees with LDA on %d/90"
          % (name, k, k / 90, lo, hi, agree))
    out.append({"scheme": "leave-one-genotype-out", "model": name, "correct": k, "n": 90,
                "accuracy": round(k / 90, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                "agreement_with_linear": agree})

import csv as _csv
_p = os.path.join(os.path.dirname(D), "..", "results", "tables", "qda_check.csv")
_p = os.path.abspath(_p)
with open(_p, "w", newline="", encoding="utf-8") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("wrote", _p)
