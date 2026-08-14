#!/usr/bin/env bash
# run_all.sh — regenerate every float in manuscript order.
# Entry point for the whole compendium. Stages are numbered to match analysis/scripts/.
#   bash restore_env.sh && conda activate ldapaper && bash run_all.sh
#
# 06_tables.py runs TWICE by design: pass 1 produces SuppTable_S5, which stage [4c]
# (04c_robustness_checks.py) needs as its own input; pass 2 then folds 04c's own outputs
# (robustness_checks.csv, within_treatment_genotype.csv) into SuppTable S6-S8.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PY=${PY:-python}
echo "[1] ingest raw data -> tidy plant-level + genotype-mean tables"
$PY analysis/scripts/01_ingest_raw_data.py
$PY analysis/scripts/01b_regenerate_genotype_means.py
echo "[2] descriptives — two-way ANOVA, variance components, heritability"
$PY analysis/scripts/02_descriptives.py
echo "[3] LDA + leave-one-genotype-out CV + permutation null"
$PY analysis/scripts/03_lda_logocv.py
echo "[3b] validation comparison, tolerance strata, parsimony, bootstraps"
$PY analysis/scripts/03b_validation_comparison.py
echo "[3c] trait-pair choice, on evidence"
$PY analysis/scripts/03c_trait_pair_decision.py
echo "[3d] per-group tests the figures display"
$PY analysis/scripts/03d_group_stats.py
echo "[3e] fixed 10-model comparison, leave-one-genotype-out CV"
$PY analysis/scripts/03e_model_comparison.py
echo "[4] LDA diagnostics — Wilks lambda, Box M, canonical correlation"
$PY analysis/scripts/04_diagnostics.py
$PY analysis/scripts/04b_qda_check.py
echo "[5] tables, pass 1 — produces SuppTable_S5, needed by stage [4c] below"
$PY analysis/scripts/06_tables.py
echo "[4c] robustness checks — paired model comparison, precision sensitivity, decision margins"
$PY analysis/scripts/04c_robustness_checks.py
echo "[5b] tables, pass 2 — folds robustness_checks.csv/within_treatment_genotype.csv into S6-S8"
$PY analysis/scripts/06_tables.py
echo "[6] figures -> analysis/results/figures/{main,supp}"
$PY analysis/scripts/05_figures.py
$PY analysis/scripts/05b_figures_extended.py
$PY analysis/scripts/05c_figures_supp.py
echo "[7] Source_Data.xlsx"
$PY analysis/scripts/07_source_data.py
echo "DONE."
