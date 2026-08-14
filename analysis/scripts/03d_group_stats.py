#!/usr/bin/env python
"""Stage 3d — the per-group tests the figures must display.

Every panel that puts two groups side by side has to show the test that compares them; a figure
that shows a difference without its statistic invites the reader to eyeball significance.

Writes: results/tables/group_tests.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")


def norm(x):
    k = "".join(str(x).upper().split()).replace("-", "").replace("(", "").replace(")", "")
    return {"MOROBEREKAN": "MOROBERAKAN", "RPW9SS1": "RPW94SS1"}.get(k, k)


def d(a, b):
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


rows = []
plants = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
plants["k"] = plants["genotype"].map(norm)
tol = pd.read_csv(os.path.join(DERIVED, "genotype_tolerance.csv"))
tol["k"] = tol["genotype_name"].map(norm)
tmap = dict(zip(tol["k"], tol["tolerance_class"]))
plants["tol"] = plants["k"].map(tmap)

# control vs low nitrogen, within each tolerance class, plant level
for tr in ("rCCI_30", "QY_BL_30"):
    for cl in ("susceptible", "moderately tolerant", "tolerant"):
        s = plants[plants["tol"] == cl]
        a = s[s.treatment == "Control"][tr].to_numpy(float)
        b = s[s.treatment == "NStress"][tr].to_numpy(float)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(panel="Fig7a", trait=tr, group=cl, n_control=len(a), n_lowN=len(b),
                         test="Welch t", stat=t, p=p, mannwhitney_p=pu, cohens_d=d(a, b)))

# 30 DAT vs 60 DAT, genotype means, paired within genotype
gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
gm["treatment"] = gm["treatment"].replace({"N stress": "NStress"})
for tr in ("CCI_TL", "CCI_BL", "QY_TL", "QY_BL", "rCCI", "rQY"):
    for treat in ("Control", "NStress"):
        w = gm[(gm.trait == tr) & (gm.treatment == treat)].pivot_table(
            index="genotype_name", columns="timepoint", values="value")
        if {"30 DAT", "60 DAT"}.issubset(w.columns):
            a, b = w["30 DAT"].to_numpy(float), w["60 DAT"].to_numpy(float)
            t, p = stats.ttest_rel(a, b)
            rows.append(dict(panel="Fig6", trait=tr, group=treat, n_control=len(a), n_lowN=len(b),
                             test="paired t, 30 vs 60 DAT", stat=t, p=p, mannwhitney_p=np.nan,
                             cohens_d=(a - b).mean() / (a - b).std(ddof=1)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(TABLES, "group_tests.csv"), index=False, float_format="%.6g")
print("wrote group_tests.csv  (%d tests)" % len(out))
print(out[out.panel == "Fig7a"][["trait", "group", "stat", "p", "cohens_d"]].to_string(index=False))
