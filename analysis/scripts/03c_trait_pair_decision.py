#!/usr/bin/env python
"""Stage 3c — choose the second trait to pair with rCCI_30, on evidence rather than preference.

The candidates are QY_BL_30, the absolute dark-adapted Fv/Fm of the bottom leaf, and rQY_30, the
bottom-to-top ratio of the same quantity. Earlier drafts asserted that rQY has no dynamic range;
that was a misreading of the control group alone and is wrong. Both traits separate the two
regimes. The choice therefore has to be made on grounds that actually distinguish them:

  1. INFORMATION — how much does each add beyond rCCI_30? Two ratios built from the same two leaf
     positions may carry the same information twice.
  2. MEASUREMENT ERROR — a ratio of two nearly equal bounded quantities amplifies instrument
     error. Fv/Fm of the two leaves differ by about 0.005 in the control, so rQY is a ratio of two
     numbers that are almost identical; rCCI's components differ by tens of units.
  3. BURDEN — how many instrument readings does the rule need in the field?

Writes: results/tables/trait_pair_decision.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")

# FluorPen FP-100 and MC-100 repeatability, from the instrument documentation. Used only to
# compare the two candidate traits on the same footing (rQY's error is propagated from this same
# constant through its ratio formula below), so the comparison is invariant to this value's
# exact precision, not an absolute claim.
SD_FVFM = 0.010          # Fv/Fm, single reading
SD_CCI_REL = 0.02        # CCI, relative


def main():
    gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
    gm["treatment"] = gm["treatment"].replace({"N stress": "NStress"})
    g30 = gm[gm["timepoint"] == "30 DAT"].pivot_table(
        index=["genotype_name", "treatment"], columns="trait", values="value")
    y = (g30.index.get_level_values("treatment") == "NStress").astype(int)

    rows = []
    for cand in ("QY_BL", "rQY"):
        v = g30[cand].to_numpy(float)
        r = g30["rCCI"].to_numpy(float)
        c, n = v[y == 0], v[y == 1]

        # 1. information beyond rCCI
        rho, p_rho = stats.pearsonr(r, v)
        # partial: variance of the candidate not explained by rCCI, within treatment
        resid = []
        for t in (0, 1):
            m = y == t
            b = np.polyfit(r[m], v[m], 1)
            resid.append(v[m] - np.polyval(b, r[m]))
        resid = np.concatenate(resid)
        r_within = np.corrcoef(np.concatenate([r[y == 0], r[y == 1]]),
                               np.concatenate([v[y == 0] - v[y == 0].mean(),
                                               v[y == 1] - v[y == 1].mean()]))[0, 1]

        # 2. measurement error propagation
        if cand == "QY_BL":
            sd_meas = SD_FVFM
        else:
            # ratio of two Fv/Fm readings: sd_rQY ~ rQY * sqrt((sd/top)^2 + (sd/bottom)^2)
            top = g30["QY_TL"].to_numpy(float)
            sd_meas = float(np.mean(v * np.sqrt((SD_FVFM / top) ** 2 + (SD_FVFM / (v * top)) ** 2)))
        gap = c.min() - n.max()
        rows.append(dict(
            candidate=cand,
            control_lo=c.min(), control_hi=c.max(), lowN_lo=n.min(), lowN_hi=n.max(),
            gap=gap, control_range=c.max() - c.min(),
            cohens_d=abs((c.mean() - n.mean()) /
                         np.sqrt(((len(c) - 1) * c.var(ddof=1) + (len(n) - 1) * n.var(ddof=1)) /
                                 (len(c) + len(n) - 2))),
            corr_with_rCCI=rho, corr_p=p_rho, within_treatment_corr=r_within,
            measurement_sd=sd_meas, gap_in_measurement_sd=gap / sd_meas,
            readings_needed=1 if cand == "QY_BL" else 2,
        ))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(TABLES, "trait_pair_decision.csv"), index=False, float_format="%.6g")

    print("SECOND-TRAIT DECISION — candidates to pair with rCCI_30\n")
    for _, r in d.iterrows():
        print("%-8s control %.3f-%.3f   low N %.3f-%.3f   gap %.4f   |d| %.2f"
              % (r.candidate, r.control_lo, r.control_hi, r.lowN_lo, r.lowN_hi, r.gap, r.cohens_d))
        print("         correlation with rCCI_30            r = %+.3f (p = %.3g)"
              % (r.corr_with_rCCI, r.corr_p))
        print("         instrument sd for this quantity     %.4f" % r.measurement_sd)
        print("         gap expressed in instrument sd      %.1f" % r.gap_in_measurement_sd)
        print("         fluorescence readings per plant     %d" % r.readings_needed)
        print()

    a = d[d.candidate == "QY_BL"].iloc[0]
    b = d[d.candidate == "rQY"].iloc[0]
    print("VERDICT")
    print("  gap in instrument standard deviations:  QY_BL %.1f  vs  rQY %.1f  (%.1fx)"
          % (a.gap_in_measurement_sd, b.gap_in_measurement_sd,
             a.gap_in_measurement_sd / b.gap_in_measurement_sd))
    print("  correlation with rCCI_30:               QY_BL %+.3f  vs  rQY %+.3f"
          % (a.corr_with_rCCI, b.corr_with_rCCI))
    print("  readings per plant for the pair:        3 (2 CCI + 1 Fv/Fm)  vs  4 (2 CCI + 2 Fv/Fm)")


if __name__ == "__main__":
    main()
