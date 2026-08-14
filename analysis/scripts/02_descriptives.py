#!/usr/bin/env python
"""Stage 2 — descriptive statistics, two-way ANOVA and variance components.

Answers the supervisor's comment 103: "Currently, Genotypic, environmental variance data is missing.
Results of phenotypic variance is available. Analyse the data for sub components and include results."

Sigma-squared components follow the standard random-model expectations for a genotype x treatment
factorial with r replicates:
    sigma2_GxT = (MS_GxT - MS_e) / r
    sigma2_G   = (MS_G   - MS_GxT) / (r * t)
    sigma2_T   = (MS_T   - MS_GxT) / (r * g)
    sigma2_e   =  MS_e
    sigma2_P   =  sigma2_G + sigma2_T + sigma2_GxT + sigma2_e
    H2 (broad) =  sigma2_G / sigma2_P
Negative estimates are reported as 0 with the raw estimate retained, because a negative variance is an
estimator artefact and must never be printed as a result.

Outputs: results/tables/Table2_anova.csv, variance_components.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
os.makedirs(TABLES, exist_ok=True)


def two_way(df, trait):
    """Balanced two-way factorial ANOVA, genotype x treatment, computed from sums of squares."""
    y = df[trait].to_numpy(float)
    g = df["genotype"].to_numpy()
    t = df["treatment"].to_numpy()
    gl, tl = np.unique(g), np.unique(t)
    ng, nt = len(gl), len(tl)
    r = len(y) // (ng * nt)
    grand = y.mean()

    ss_g = r * nt * sum((y[g == a].mean() - grand) ** 2 for a in gl)
    ss_t = r * ng * sum((y[t == b].mean() - grand) ** 2 for b in tl)
    ss_gt = 0.0
    ss_e = 0.0
    for a in gl:
        for b in tl:
            cell = y[(g == a) & (t == b)]
            ss_gt += r * (cell.mean() - y[g == a].mean() - y[t == b].mean() + grand) ** 2
            ss_e += ((cell - cell.mean()) ** 2).sum()

    df_g, df_t, df_gt = ng - 1, nt - 1, (ng - 1) * (nt - 1)
    df_e = ng * nt * (r - 1)
    ms = dict(G=ss_g / df_g, T=ss_t / df_t, GxT=ss_gt / df_gt, e=ss_e / df_e)
    f = {k: ms[k] / ms["e"] for k in ("G", "T", "GxT")}
    p = {
        "G": stats.f.sf(f["G"], df_g, df_e),
        "T": stats.f.sf(f["T"], df_t, df_e),
        "GxT": stats.f.sf(f["GxT"], df_gt, df_e),
    }

    v_gt = (ms["GxT"] - ms["e"]) / r
    v_g = (ms["G"] - ms["GxT"]) / (r * nt)
    v_t = (ms["T"] - ms["GxT"]) / (r * ng)
    v_e = ms["e"]
    raw = dict(G=v_g, T=v_t, GxT=v_gt, e=v_e)
    clip = {k: max(v, 0.0) for k, v in raw.items()}
    v_p = sum(clip.values())
    h2 = clip["G"] / v_p if v_p > 0 else np.nan

    return dict(
        trait=trait, n=len(y), n_genotypes=ng, n_treatments=nt, n_reps=r,
        mean=y.mean(), sd=y.std(ddof=1), cv_pct=100 * y.std(ddof=1) / y.mean(),
        df_G=df_g, MS_G=ms["G"], F_G=f["G"], p_G=p["G"],
        df_T=df_t, MS_T=ms["T"], F_T=f["T"], p_T=p["T"],
        df_GxT=df_gt, MS_GxT=ms["GxT"], F_GxT=f["GxT"], p_GxT=p["GxT"],
        df_e=df_e, MS_e=ms["e"],
        var_G=clip["G"], var_T=clip["T"], var_GxT=clip["GxT"], var_e=clip["e"], var_P=v_p,
        var_G_raw=raw["G"], var_GxT_raw=raw["GxT"],
        H2_broad=h2,
        pct_var_G=100 * clip["G"] / v_p, pct_var_T=100 * clip["T"] / v_p,
        pct_var_GxT=100 * clip["GxT"] / v_p, pct_var_e=100 * clip["e"] / v_p,
    )


def main():
    df = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
    rows = [two_way(df, t) for t in ("rCCI_30", "QY_BL_30")]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "Table2_anova.csv"), index=False, float_format="%.6g")

    print("TWO-WAY ANOVA — 30 DAT, plant level (n=90, 15 genotypes x 2 treatments x 3 reps)\n")
    hdr = f"{'trait':<10}{'F_G':>9}{'p_G':>11}{'F_T':>11}{'p_T':>12}{'F_GxT':>9}{'p_GxT':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['trait']:<10}{r['F_G']:>9.2f}{r['p_G']:>11.2e}{r['F_T']:>11.1f}"
              f"{r['p_T']:>12.2e}{r['F_GxT']:>9.2f}{r['p_GxT']:>11.2e}")

    print("\nVARIANCE COMPONENTS (% of phenotypic variance)\n")
    print(f"{'trait':<10}{'sigma2_G':>11}{'%G':>7}{'sigma2_T':>11}{'%T':>7}"
          f"{'sigma2_GxT':>12}{'%GxT':>8}{'sigma2_e':>11}{'%e':>7}{'H2':>8}")
    for r in rows:
        print(f"{r['trait']:<10}{r['var_G']:>11.5f}{r['pct_var_G']:>7.1f}{r['var_T']:>11.5f}"
              f"{r['pct_var_T']:>7.1f}{r['var_GxT']:>12.5f}{r['pct_var_GxT']:>8.1f}"
              f"{r['var_e']:>11.5f}{r['pct_var_e']:>7.1f}{r['H2_broad']:>8.3f}")

    vc = out[["trait", "var_G", "var_T", "var_GxT", "var_e", "var_P", "H2_broad",
              "pct_var_G", "pct_var_T", "pct_var_GxT", "pct_var_e"]]
    vc.to_csv(os.path.join(TABLES, "variance_components.csv"), index=False, float_format="%.6g")
    print(f"\nwritten: {TABLES}/Table2_anova.csv, variance_components.csv")


if __name__ == "__main__":
    main()
