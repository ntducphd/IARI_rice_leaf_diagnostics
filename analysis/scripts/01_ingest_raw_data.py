#!/usr/bin/env python
"""Stage 1 — ingest the raw source matrix (analysis/data/raw_LDA.xlsx, sheet "Final data",
90 plants x 119 columns) and derive the two focal 30-DAT traits used throughout the analysis.

Derived index, per Methods ("rCCI = CCI_bottom / CCI_top"): rCCI_30 = CCI_BL_30 / CCI_TL_30.
QY_BL_30 is used directly from the raw column.

Precision note: CCI_TL_30/CCI_BL_30 are recorded to 0.1 in the raw file, so rCCI_30 is written
here at full computed float precision rather than rounded. QY_BL_30 is itself already recorded
to 2 decimal places in the source data, so its precision is unchanged by this step; the
rounding-sensitivity analysis in 04c_robustness_checks.py remains relevant for QY-based results.

Optional legacy check: if a two-decimal-place reference copy of this output is present at
plants_30DAT.tsv.legacy_reference (not shipped with this repository), the script verifies its
own output reproduces it row-for-row when rounded to 2dp, matched on (genotype, treatment) with
replicate order ignored, and raises if any group diverges. This is a defensive consistency
check retained from development; its absence is not an error.

Replicate numbering: reps are assigned per (genotype, treatment) in file order as 1-3 for
Control and 4-6 for NStress, matching the numbering convention used throughout this compendium
(a within-treatment 1-3 numbering is also present in the raw file's own Rep column but is not
used here, for consistency with downstream tables that expect the combined 1-6 range).

Output: analysis/data/derived/plants_30DAT.tsv (genotype, treatment, rep, rCCI_30, QY_BL_30)
"""
import os

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
RAW_XLSX = os.path.join(ROOT, "analysis", "data", "raw_LDA.xlsx")
os.makedirs(DERIVED, exist_ok=True)


def load_final_data():
    wb = openpyxl.load_workbook(RAW_XLSX, read_only=True, data_only=True)
    ws = wb["Final data"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
    recs = []
    for r in rows[1:]:
        if r[idx["Genotype name"]] is None:
            continue
        recs.append({
            "genotype": str(r[idx["Genotype name"]]).strip(),
            "treatment": str(r[idx["Treatment"]]).strip(),
            "rep": int(r[idx["Rep"]]),
            "CCI_TL_30": float(r[idx["CCI_TL_30"]]),
            "CCI_BL_30": float(r[idx["CCI_BL_30"]]),
            "QY_BL_30": float(r[idx["QY_BL_30"]]),
        })
    return recs


def main():
    recs = load_final_data()
    print(f"[01] loaded {len(recs)} plant-level records from raw_LDA.xlsx 'Final data'")
    assert len(recs) == 90, f"expected 90 plants, got {len(recs)}"

    out_rows = []
    seen = {}
    for r in recs:
        key = (r["genotype"], r["treatment"])
        seen[key] = seen.get(key, 0) + 1
        synth_rep = seen[key] + (3 if r["treatment"] == "NStress" else 0)
        rcci_30 = r["CCI_BL_30"] / r["CCI_TL_30"]
        out_rows.append((r["genotype"], r["treatment"], synth_rep, rcci_30, r["QY_BL_30"]))

    out_path = os.path.join(DERIVED, "plants_30DAT.tsv")
    with open(out_path, "w") as f:
        f.write("genotype\ttreatment\trep\trCCI_30\tQY_BL_30\n")
        for g, t, rep, rcci, qy in out_rows:
            f.write(f"{g}\t{t}\t{rep}\t{rcci:.6f}\t{qy:.6f}\n")
    print(f"[01] wrote {len(out_rows)} rows -> {out_path}")

    ref_path = os.path.join(DERIVED, "plants_30DAT.tsv.legacy_reference")
    if os.path.exists(ref_path):
        old = {}
        with open(ref_path) as f:
            next(f)
            for line in f:
                g, t, rep, rcci, qy = line.rstrip("\n").split("\t")
                old.setdefault((g, t), []).append((round(float(rcci), 2), round(float(qy), 2)))
        new = {}
        for g, t, rep, rcci, qy in out_rows:
            new.setdefault((g, t), []).append((round(rcci, 2), round(qy, 2)))
        mismatches = 0
        for key in old:
            if sorted(old[key]) != sorted(new.get(key, [])):
                mismatches += 1
                print(f"[01] MISMATCH {key}: old={sorted(old[key])} new={sorted(new.get(key, []))}")
        if mismatches:
            raise AssertionError(f"[01] {mismatches} genotype x treatment groups diverged from "
                                 f"the legacy reference file at 2dp")
        print(f"[01] OK -- all {len(old)} genotype x treatment groups match the legacy "
             f"reference file at 2dp (replicate order ignored)")
    print("[01] done")


if __name__ == "__main__":
    main()
