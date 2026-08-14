#!/usr/bin/env python
"""Stage 2b — two-way ANOVA for the top-leaf chlorophyll index (CCI_TL) at 30 and 60 DAT,
plant level, same method as 02_descriptives.py's two_way(). Backs the four top-leaf *F* values
in the Abstract/Results, which Table 2 does not cover (Table 2 is rCCI_30 and QY_BL_30 only).

30 DAT reads raw_LDA.xlsx column CCI_TL_30 (also used by 01_ingest_raw_data.py's
rCCI_30 = CCI_BL_30 / CCI_TL_30). 60 DAT reads CC_TL_60 -- the raw file's own 60 DAT
chlorophyll columns are spelled "CC_" rather than "CCI_" (columns 70-71 vs. the 30 DAT columns'
"CCI_" at 36-37).

Output: results/tables/Table_abstract_top_leaf_stats.csv
"""
import os
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
RAW_XLSX = os.path.join(ROOT, "analysis", "data", "raw_LDA.xlsx")
os.makedirs(TABLES, exist_ok=True)

# (raw column, output label, historical F_G, historical F_T, source of the historical value)
CLAIMS = [
    ("CCI_TL_30", "CCI_TL_30", 57.24, 0.09,
     "Abstract: \"did not separate the regimes at 30 DAT (F = 0.09)\"; "
     "\"cut the genotype F from 57.24 to 5.34\""),
    ("CC_TL_60", "CCI_TL_60", None, 476.38,
     "Abstract: \"did by 60 DAT (F = 476.38)\""),
]


def load_raw():
    wb = openpyxl.load_workbook(RAW_XLSX, read_only=True, data_only=True)
    ws = wb["Final data"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
    recs = []
    for r in rows[1:]:
        if r[idx["Genotype name"]] is None:
            continue
        recs.append({"genotype": str(r[idx["Genotype name"]]).strip(),
                      "treatment": str(r[idx["Treatment"]]).strip(),
                      "row": r, "idx": idx})
    return recs


def two_way(y, g, t):
    """Identical method to 02_descriptives.py's two_way(): balanced two-way factorial ANOVA,
    genotype x treatment, computed from sums of squares. Kept as a standalone copy rather than
    an import so this stage does not depend on 02_descriptives.py's plants_30DAT.tsv-only main()."""
    gl, tl = np.unique(g), np.unique(t)
    ng, nt = len(gl), len(tl)
    r = len(y) // (ng * nt)
    grand = y.mean()
    ss_g = r * nt * sum((y[g == a].mean() - grand) ** 2 for a in gl)
    ss_t = r * ng * sum((y[t == b].mean() - grand) ** 2 for b in tl)
    ss_gt, ss_e = 0.0, 0.0
    for a in gl:
        for b in tl:
            cell = y[(g == a) & (t == b)]
            ss_gt += r * (cell.mean() - y[g == a].mean() - y[t == b].mean() + grand) ** 2
            ss_e += ((cell - cell.mean()) ** 2).sum()
    df_g, df_t, df_gt = ng - 1, nt - 1, (ng - 1) * (nt - 1)
    df_e = ng * nt * (r - 1)
    ms = dict(G=ss_g / df_g, T=ss_t / df_t, GxT=ss_gt / df_gt, e=ss_e / df_e)
    f = {k: ms[k] / ms["e"] for k in ("G", "T", "GxT")}
    p = {"G": stats.f.sf(f["G"], df_g, df_e), "T": stats.f.sf(f["T"], df_t, df_e),
         "GxT": stats.f.sf(f["GxT"], df_gt, df_e)}
    return dict(df_G=df_g, F_G=f["G"], p_G=p["G"], df_T=df_t, F_T=f["T"], p_T=p["T"],
                df_GxT=df_gt, F_GxT=f["GxT"], p_GxT=p["GxT"], df_e=df_e)


def main():
    recs = load_raw()
    assert len(recs) == 90, f"expected 90 plants, got {len(recs)}"
    g = np.array([r["genotype"] for r in recs])
    t = np.array([r["treatment"] for r in recs])

    rows = []
    for col, label, hist_fg, hist_ft, claim in CLAIMS:
        idx = recs[0]["idx"]
        y = np.array([float(r["row"][idx[col]]) for r in recs])
        res = two_way(y, g, t)
        res.update(trait=label, raw_column=col, n=len(y), source=claim,
                    historical_F_G=hist_fg, historical_F_T=hist_ft)
        rows.append(res)

    out = pd.DataFrame(rows)[
        ["trait", "raw_column", "n", "df_G", "F_G", "p_G", "historical_F_G",
         "df_T", "F_T", "p_T", "historical_F_T", "df_GxT", "F_GxT", "p_GxT", "df_e", "source"]]
    out.to_csv(os.path.join(TABLES, "Table_abstract_top_leaf_stats.csv"), index=False,
               float_format="%.6g")

    print("TOP-LEAF CCI TWO-WAY ANOVA — recomputed from raw_LDA.xlsx vs. Abstract/Results claims\n")
    for r in rows:
        print(f"{r['trait']}: F_G={r['F_G']:.2f} (claimed {r['historical_F_G']}), "
              f"F_T={r['F_T']:.2f} (claimed {r['historical_F_T']})")
    print(f"\nwritten: {TABLES}/Table_abstract_top_leaf_stats.csv")


if __name__ == "__main__":
    main()
