#!/usr/bin/env python
"""Stage 7 — Source_Data.xlsx: one worksheet per data-bearing figure.

Lets a reviewer check any figure without the raw file. Figure 1 is a schematic and is excluded.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
TABLES = os.path.join(ROOT, "analysis", "results", "tables")
OUT = os.path.join(ROOT, "analysis", "results", "source_data")
os.makedirs(OUT, exist_ok=True)

plants = pd.read_csv(os.path.join(DERIVED, "plants_30DAT.tsv"), sep="\t")
gm = pd.read_csv(os.path.join(DERIVED, "genotype_means.tsv"), sep="\t")
gm30 = gm[(gm["timepoint"] == "30 DAT") &
          gm["trait"].isin(["CCI_TL", "CCI_BL", "rCCI", "QY_BL"])]

sheets = {
    "Figure2": gm30.pivot_table(index=["genotype_name", "treatment"],
                                columns="trait", values="value").reset_index(),
    "Figure3": pd.read_csv(os.path.join(TABLES, "variance_components.csv")),
    "Figure4": pd.read_csv(os.path.join(TABLES, "logocv_per_genotype.csv")),
    "Figure5": plants,
    "Table2_anova": pd.read_csv(os.path.join(TABLES, "Table2_anova.csv")),
    "Table3_performance": pd.read_csv(os.path.join(TABLES, "Table3_performance.csv")),
    "Table4_discriminant": pd.read_csv(os.path.join(TABLES, "Table4_discriminant.csv")),
}
p = os.path.join(OUT, "Source_Data.xlsx")
with pd.ExcelWriter(p, engine="openpyxl") as xl:
    for name, df in sheets.items():
        df.to_excel(xl, sheet_name=name[:31], index=False)
print("wrote", p)
for k, v in sheets.items():
    print("  %-22s %d rows x %d cols" % (k, len(v), v.shape[1]))
