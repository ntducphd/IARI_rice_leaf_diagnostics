#!/usr/bin/env python
"""Stage 1b — build genotype_means.tsv (genotype x treatment x timepoint x trait means/SE)
directly from the plant-level raw matrix (analysis/data/raw_LDA.xlsx, sheet "Final data").

Scope: computes means/SE for (a) every raw-measured trait present in the "Final data" sheet,
and (b) the ratio/difference derived traits whose formulas are given explicitly by
trait_dictionary.tsv's own full_name column (rLL/rLW/rCCI/rQY/rPhoto/rCond/rCi/rTr = bottom:top
ratio; dCCI/dQY = top-bottom difference). AGR_*/RGR_* (absolute/relative growth rate) traits are
deliberately NOT computed here: the dictionary marks these as biomass-based (RGR unit
"mg g-1 day-1"), but the raw file's SDW_30C/SDW_60C column naming does not unambiguously map to
a single dictionary base trait/formula, so they are left for explicit author confirmation.

Column-naming note: the raw file's abbreviated 60-DAT column names (P_TL_60, Co_TL_60,
CC_TL_60, ...) map to trait_dictionary.tsv's canonical names (Photo_TL_60, Cond_TL_60,
CCI_TL_60, ...) except where the dictionary itself already uses the raw file's abbreviated
spelling (CC_TL_60/CC_BL_60), handled explicitly below.

Output: analysis/data/derived/genotype_means.tsv (genotype_code, genotype_name, treatment,
timepoint, trait, value, se)
"""
import csv
import os
import statistics

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "analysis", "data", "derived")
RAW_XLSX = os.path.join(ROOT, "analysis", "data", "raw_LDA.xlsx")

# genotype_code -> (raw "Genotype name" spelling, display spelling used throughout this
# compendium's tables and figures).
GENOTYPES = [
    (1, "MOROBEREKAN", "Moroberakan"), (2, "IR83388-B-B108-3", "IR83388-B-B108-3"),
    (3, "IR77298-14-1-2-10", "IR77298-14-1-2-10"), (4, "PUSA1121", "PUSA1121"),
    (5, "BAM8315", "BAM8315"), (6, "BAM812", "BAM812"), (7, "MALCHI", "Malchi"),
    (8, "BAM3690", "BAM 3690"), (9, "BAM4138", "BAM 4138"), (10, "BAM4521", "BAM 4521"),
    (11, "BLACK GORA", "Black gora"), (12, "SUWEON", "Suweon"), (13, "KUNJUKUNJU", "Kunjukunju"),
    (14, "CAUVERY", "Cauvery"), (15, "RPW-9-(SS1)", "RPW9-4(SS1)"),
]

TIMEPOINTS = {"0": "0 DAT", "30": "30 DAT", "60": "60 DAT", "FL": "Flag leaf", "MAT": "End of season"}

# raw-column -> canonical dictionary trait name, only where they differ.
RAW_TO_CANON_60 = {
    "P_TL_60": "Photo_TL_60", "Co_TL_60": "Cond_TL_60",
    "P_BL_60": "Photo_BL_60", "Co_BL_60": "Cond_BL_60",
}


def canon(raw_col):
    return RAW_TO_CANON_60.get(raw_col, raw_col)


# trait_dictionary.tsv names this trait "CC_TL_60"/"CC_BL_60" (base trait "CC_TL"/"CC_BL"),
# while every other genotype_means.tsv consumer (figures, tables) expects "CCI_TL"/"CCI_BL"
# uniformly across both timepoints, matching the 30-DAT columns' own naming. Remapped here so
# the output trait names are consistent across timepoints.
BASE_TRAIT_OVERRIDE = {"CC_TL": "CCI_TL", "CC_BL": "CCI_BL"}


def base_trait(canon_name, timepoint_label):
    """Strip the timepoint suffix convention this table uses: "_0"/"_30"/"_60" stripped for
    those three timepoints (CCI_TL_30 -> CCI_TL); flag-leaf traits strip only a trailing "_C"
    (FLL_C -> FLL) and otherwise pass through unchanged (CCI_FL stays CCI_FL); end-of-season /
    maturity traits are already bare in trait_dictionary.tsv and pass through unchanged.
    SDW_30C/SDW_60C (raw) map to base trait "SDWC" at 30/60 DAT. Every path funnels through
    BASE_TRAIT_OVERRIDE at the end, for the CC_TL/CC_BL naming difference above."""
    if timepoint_label == "0 DAT":
        base = canon_name[:-2] if canon_name.endswith("_0") else canon_name
    elif timepoint_label in ("30 DAT", "60 DAT"):
        suffix = "_30" if timepoint_label == "30 DAT" else "_60"
        base = canon_name[: -len(suffix)] if canon_name.endswith(suffix) else canon_name
    elif timepoint_label == "Flag leaf":
        base = canon_name[:-2] if canon_name.endswith("_C") else canon_name
    else:
        base = canon_name
    return BASE_TRAIT_OVERRIDE.get(base, base)


# trait_dictionary.tsv is the ground truth for timepoint labels. Its own timepoint strings are
# translated to this table's existing convention ("Flag leaf", not "Flag leaf stage"; "End of
# season", not "Maturity / harvest") so downstream consumers that filter on those exact strings
# keep working unchanged.
TP_TRANSLATE = {"0 DAT": "0 DAT", "30 DAT": "30 DAT", "60 DAT": "60 DAT",
               "Flag leaf stage": "Flag leaf", "Maturity / harvest": "End of season"}


def load_trait_timepoints():
    path = os.path.join(DERIVED, "trait_dictionary.tsv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tp = row.get("timepoint", "").strip()
            if tp in TP_TRANSLATE:
                out[row["trait"]] = TP_TRANSLATE[tp]
    return out


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
                    "treatment": str(r[idx["Treatment"]]).strip(), "row": r, "idx": idx})
    return recs


def mean_se(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    se = (statistics.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return m, se


def main():
    recs = load_raw()
    idx = recs[0]["idx"]
    raw_cols = [c for c in idx if c not in ("G.ID", "Genotype name", "Treatment", "Rep")]
    trait_tp = load_trait_timepoints()

    out_rows = []
    skipped_no_dict_entry = set()
    for code, raw_g, disp_g in GENOTYPES:
        for treatment in ["Control", "NStress"]:
            group = [r["row"] for r in recs if r["genotype"] == raw_g and r["treatment"] == treatment]
            if not group:
                continue

            def col(name):
                return [row[idx[name]] for row in group] if name in idx else None

            # (a) every raw-measured column, trait name = canonicalized raw column name.
            # SDW_30C/SDW_60C -> base trait "SDWC" at 30/60 DAT (trait_dictionary.tsv's own
            # "SDW_30" entry, without the trailing "C", does not match the raw column name).
            EXTRA_TRAIT_TP = {"SDW_30C": ("SDWC", "30 DAT"), "SDW_60C": ("SDWC", "60 DAT")}
            for raw_col in raw_cols:
                vals = col(raw_col)
                if vals is None:
                    continue
                m, se = mean_se(vals)
                if m is None:
                    continue
                if raw_col in EXTRA_TRAIT_TP:
                    trait, tp = EXTRA_TRAIT_TP[raw_col]
                else:
                    canon_name = canon(raw_col)
                    tp = trait_tp.get(canon_name, "")
                    if not tp:
                        skipped_no_dict_entry.add(raw_col)
                        continue  # not present in trait_dictionary.tsv -- skip rather than guess
                    trait = base_trait(canon_name, tp)
                out_rows.append((code, disp_g, treatment, tp, trait, m, se))

            # (b) unambiguous ratio (bottom:top) and difference (top-bottom) derived traits
            def ratio_diff(prefix_tl, prefix_bl, out_r, out_d):
                tl, bl = col(prefix_tl), col(prefix_bl)
                if tl is None or bl is None:
                    return
                r_vals = [b / t for b, t in zip(bl, tl) if t not in (0, None) and b is not None]
                d_vals = [t - b for t, b in zip(tl, bl) if t is not None and b is not None]
                if r_vals:
                    m, se = mean_se(r_vals)
                    tp = trait_tp.get(out_r, "")
                    out_rows.append((code, disp_g, treatment, tp, base_trait(out_r, tp), m, se))
                if out_d and d_vals:
                    m, se = mean_se(d_vals)
                    tp = trait_tp.get(out_d, "")
                    out_rows.append((code, disp_g, treatment, tp, base_trait(out_d, tp), m, se))

            ratio_diff("TLL_30", "BLL_30", "rLL_30", None)
            ratio_diff("TLW_30", "BLW_30", "rLW_30", None)
            ratio_diff("CCI_TL_30", "CCI_BL_30", "rCCI_30", "dCCI_30")
            ratio_diff("QY_TL_30", "QY_BL_30", "rQY_30", "dQY_30")
            ratio_diff("Photo_TL_30", "Photo_BL_30", "rPhoto_30", None)
            ratio_diff("Cond_TL_30", "Cond_BL_30", "rCond_30", None)
            ratio_diff("Ci_TL_30", "Ci_BL_30", "rCi_30", None)
            ratio_diff("Tr_TL_30", "Tr_BL_30", "rTr_30", None)
            ratio_diff("TLL_60", "BLL_60", "rLL_60", None)
            ratio_diff("TLW_60", "BLW_60", "rLW_60", None)
            ratio_diff("CC_TL_60", "CC_BL_60", "rCCI_60", "dCCI_60")
            ratio_diff("QY_TL_60", "QY_BL_60", "rQY_60", "dQY_60")
            ratio_diff("P_TL_60", "P_BL_60", "rPhoto_60", None)
            ratio_diff("Co_TL_60", "Co_BL_60", "rCond_60", None)
            ratio_diff("Ci_TL_60", "Ci_BL_60", "rCi_60", None)
            ratio_diff("Tr_TL_60", "Tr_BL_60", "rTr_60", None)

    out_path = os.path.join(DERIVED, "genotype_means.tsv")
    with open(out_path, "w") as f:
        f.write("genotype_code\tgenotype_name\ttreatment\ttimepoint\ttrait\tvalue\tse\n")
        for row in out_rows:
            code, g, t, tp, trait, m, se = row
            f.write(f"{code}\t{g}\t{t}\t{tp}\t{trait}\t{m:.6g}\t{se:.6g}\n")
    print(f"[01b] wrote {len(out_rows)} rows -> {out_path}")
    if skipped_no_dict_entry:
        print(f"[01b] raw columns skipped (canonical name not found in trait_dictionary.tsv): "
             f"{sorted(skipped_no_dict_entry)}")
    print(f"[01b] NOT COMPUTED (AGR_*/RGR_* growth-rate traits -- base trait/formula ambiguous, "
         f"needs author confirmation): AGR_30-0, AGR_60-30, RGR_30-0, RGR_60-30")
    print("[01b] done")


if __name__ == "__main__":
    main()
