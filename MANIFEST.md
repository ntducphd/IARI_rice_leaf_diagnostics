# MANIFEST — reproducibility map (float → producing script → inputs → output)

For every manuscript float (figure, table, Source Data) this file names the script that
produces it and the data it consumes. With `run_all.sh` it lets a reader regenerate every
result. All paths are relative to `analysis/` unless noted.

## Reproduce in one command
```bash
bash restore_env.sh && conda activate ldapaper
bash run_all.sh
```
Outputs land in `analysis/data/derived/` and `analysis/results/{figures,tables,source_data}/`.

## Pipeline execution order (`run_all.sh`)
| Stage | Script(s) | Role |
|-------|-----------|------|
| [1] | `01_ingest_raw_data.py`, `01b_regenerate_genotype_means.py` | raw matrix -> `data/derived/plants_30DAT.tsv`, `genotype_means.tsv` |
| [2] | `02_descriptives.py` | two-way ANOVA, variance components, heritability |
| [3] | `03_lda_logocv.py` | LDA + leave-one-genotype-out CV + permutation null |
| [3b] | `03b_validation_comparison.py` | validation comparison, tolerance strata, parsimony, bootstraps |
| [3c] | `03c_trait_pair_decision.py` | trait-pair choice, on evidence -> SuppTable S7 |
| [3d] | `03d_group_stats.py` | per-group tests the figures display -> SuppTable S8 |
| [3e] | `03e_model_comparison.py` | fixed 10-model comparison, leave-one-genotype-out CV |
| [4] | `04_diagnostics.py`, `04b_qda_check.py` | Wilks' lambda, Box's M, canonical correlation, QDA check |
| [5] | `06_tables.py` (pass 1) | produces SuppTable S5, needed by stage [4c] |
| [4c] | `04c_robustness_checks.py` | paired model comparison, precision sensitivity, decision margins -> SuppTable S6 |
| [5b] | `06_tables.py` (pass 2) | folds `04c`'s own outputs into SuppTable S6-S8 |
| [6] | `05_figures.py`, `05b_figures_extended.py`, `05c_figures_supp.py` | main + supplementary figures -> `results/figures/` |
| [7] | `07_source_data.py` | `results/source_data/Source_Data.xlsx` |

## Main figures (1–10)
| Figure | Producing script | Output |
|--------|------------------|--------|
| 1  study design / workflow            | `05_figures.py`         | `Fig01_design_workflow` |
| 2  signal location on the plant       | `05_figures.py`         | `Fig02_signal_location` |
| 3  ratio mechanism (genotype F cut)   | `05_figures.py`         | `Fig03_ratio_mechanism` |
| 4  discrimination + validation        | `05_figures.py`         | `Fig04_discrimination_validation` |
| 5  threshold benchmark                | `05_figures.py`         | `Fig05_threshold_benchmark` |
| 6  temporal development (30 -> 60 DAT)| `05b_figures_extended.py` | `Fig06_temporal_development` |
| 7  genotype tolerance classes         | `05b_figures_extended.py` | `Fig07_tolerance_classes` |
| 8  trait selection                    | `05b_figures_extended.py` | `Fig08_trait_selection` |
| 9  diagnostics + robustness           | `05b_figures_extended.py` | `Fig09_diagnostics_robustness` |
| 10 validation design                  | `05b_figures_extended.py` | `Fig10_validation_design` |

Each figure is shipped as both PNG and PDF in `results/figures/main/`.

## Supplementary figures (1–4)
| Fig | Producing script | Output |
|-----|------------------|--------|
| S1  trait correlation structure     | `05c_figures_supp.py` | `SuppFig01_correlation_structure` |
| S2  60-DAT comparison               | `05c_figures_supp.py` | `SuppFig02_sixty_DAT` |
| S3  per-genotype means              | `05c_figures_supp.py` | `SuppFig03_per_genotype_means` |
| S4  full trait landscape            | `05c_figures_supp.py` | `SuppFig04_trait_landscape` |

## Main tables (1–4)
| Table | Producing script | Output |
|-------|------------------|--------|
| 1 trait definitions/summary | `06_tables.py` | `results/tables/Table1_traits.csv` |
| 2 two-way ANOVA              | `06_tables.py` | `results/tables/Table2_anova.csv` |
| 3 classifier performance     | `06_tables.py` | `results/tables/Table3_performance.csv` |
| 4 discriminant function      | `06_tables.py` | `results/tables/Table4_discriminant.csv` |

A consolidated `Tables.docx` is also produced by `06_tables.py`.

## Supplementary tables (1–9)
| Table | Producing script | Output |
|-------|------------------|--------|
| S1 genotype list                    | `06_tables.py` | `SuppTable_S1_genotype_list.csv` |
| S2 soil properties                  | `06_tables.py` | `SuppTable_S2_soil_properties.csv` |
| S3 trait dictionary                 | `06_tables.py` | `SuppTable_S3_trait_dictionary.csv` |
| S4 genotype means, 30 DAT           | `06_tables.py` | `SuppTable_S4_genotype_means_30DAT.csv` |
| S5 plant-level data, 30 DAT         | `06_tables.py` (pass 1) | `SuppTable_S5_plant_level_30DAT.csv` |
| S6 robustness checks                | `04c_robustness_checks.py` (folded in by `06_tables.py` pass 2) | `SuppTable_S6_robustness_checks.csv` |
| S7 trait-pair decision              | `03c_trait_pair_decision.py` (folded in by `06_tables.py` pass 2) | `SuppTable_S7_trait_pair_decision.csv` |
| S8 within-treatment genotype effect | `03d_group_stats.py` (folded in by `06_tables.py` pass 2) | `SuppTable_S8_within_treatment_genotype.csv` |
| S9 tolerance classes                | `06_tables.py` | `SuppTable_S9_tolerance_classes.csv` |

## Derived data provenance (`analysis/data/derived/`)
| File | Regenerated by | Provenance if not regenerated |
|------|-----------------|-------------------------------|
| `plants_30DAT.tsv` | `01_ingest_raw_data.py` | — |
| `genotype_means.tsv` | `01b_regenerate_genotype_means.py` | — |
| `genotype_means_full.tsv` | not regenerated by this pipeline | pre-computed genotype-level summary across all timepoints/traits |
| `genotype_tolerance.csv` | not regenerated by this pipeline | genotype tolerance classification from prior clustering analysis |
| `trait_dictionary.tsv` | not regenerated by this pipeline | full trait name/timepoint/units dictionary for the raw column set |
| `anova_F_authors_table1.csv` | not regenerated by this pipeline | author-provided ANOVA F-values used as a cross-check on `02_descriptives.py`'s own output |

## Source Data
`07_source_data.py` ← `results/tables/` → `results/source_data/Source_Data.xlsx` (one worksheet per data-bearing main figure).
