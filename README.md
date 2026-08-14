# IARI_rice_leaf_diagnostics

Reproducible research compendium (code + data) for:

> **Leaf-position readings for qualitative discrimination of fertilised from unfertilised rice using linear discriminant analysis with whole genotypes held out**
> Amooru Harika¹²⁴, Dhandapani Raju¹²﹡, Nguyen Trung Duc¹²³, Sivapragasam Ezhumalai², Sudhir Kumar¹², Viswanathan Chinnusamy¹² (corresponding: dandyman2k6@gmail.com)
> ¹ Division of Plant Physiology, ICAR-Indian Agricultural Research Institute (IARI), New Delhi, India
> ² Nanaji Deshmukh Plant Phenomics Centre (NDPPC), Division of Plant Physiology, ICAR-IARI, New Delhi, India
> ³ Vietnam National University of Agriculture, Hanoi, Vietnam
> ⁴ Plant Sciences, Department of Plant and Environmental Sciences, Clemson University, Clemson, SC, United States

Manuscript in preparation (not yet submitted; target journal Plant Methods, BMC). This repository contains the analysis code and the derived data needed to regenerate every quantitative result, figure and table in the paper.

## What the study does

Nitrogen is phloem-mobile and leaves the oldest leaves first, so a deficit reaches the upper canopy last. Published within-plant two-position chlorophyll indices for rice and maize read no lower than the third-to-fifth leaf from the top; none pairs a pigment ratio with a photochemical-efficiency reading, and none is validated with whole genotypes held out. Fifteen rice genotypes were grown under two fertiliser regimes, three plants each (*n* = 90), and read at 30 days after transplanting (DAT) at two leaf positions: the youngest fully expanded leaf and the oldest green leaf. Top-leaf chlorophyll index alone did not separate the regimes at 30 DAT (*F* = 0.09), though it did by 60 DAT (*F* = 476.38). Dividing bottom-leaf by top-leaf chlorophyll index (rCCI\_30) cut the genotype *F* from 57.24 to 5.34 while holding the treatment effect at *F* = 576.1; bottom-leaf dark-adapted PSII quantum yield (QY\_BL\_30) gave the largest single-trait effect at that date (*F* = 1890.5). Under leave-one-genotype-out cross-validation, a two-trait linear discriminant rule on rCCI\_30 and QY\_BL\_30 classified 90 of 90 plants and 15 of 15 genotype folds (exact 95% CI 0.960–1.000 and 0.782–1.000; *P* < 0.0005 against 2000 label permutations). Alone, QY\_BL\_30 reached 88 of 90 and rCCI\_30 83 of 90; the pair and QY\_BL\_30 alone disagree on only two plants (exact McNemar *P* = 0.50), and under ±0.005 measurement perturbation the pair lost 0.28 plants from its own score on average while QY\_BL\_30 alone lost none — these data do not by themselves justify adding the second instrument over the single reading. Classifications are scored against the imposed fertiliser regime; no tissue nitrogen concentration was measured, and the two pot soils differed before transplanting by 1.19–1.44× in available nitrogen, phosphorus, potassium and organic carbon, so regime is confounded with soil batch.

## Repository layout

```
IARI_rice_leaf_diagnostics/
  README.md              this file
  MANIFEST.md            float -> producing-script -> inputs -> output (the reproducibility contract)
  LICENSE                MIT (code) + CC-BY-4.0 (data)
  CITATION.cff            how to cite
  environment.yml         pinned Python environment
  restore_env.sh          build the conda env
  run_all.sh              one-command pipeline (ingest -> stats -> LDA/CV -> figures -> tables -> Source Data)
  analysis/
    scripts/               numbered pipeline (01-01b ingest, 02 descriptives, 03-03e LDA/CV/robustness,
                           04-04c diagnostics, 05-05c figures, 06 tables, 07 source data)
    data/
      raw_LDA.xlsx          plant-level source matrix (90 plants x 119 columns, sheet "Final data")
      derived/               tidy plant- and genotype-level tables consumed by the pipeline
    results/
      figures/               10 main (main/Fig01-10) + 4 supplementary (supp/SuppFig01-04), PNG + PDF
      tables/                 4 main tables + 9 supplementary tables (CSV) + Tables.docx
      source_data/            Source_Data.xlsx (one sheet per data-bearing figure)
```

## Reproduce

```bash
# 1. Build the pinned Python environment
bash restore_env.sh          # -> conda env 'ldapaper'
conda activate ldapaper
# 2. Regenerate the whole pipeline from the shipped raw data
bash run_all.sh
```

`run_all.sh` runs: **[1]** ingest the raw matrix and derive the plant- and genotype-level tables → **[2]** descriptive statistics (two-way ANOVA, variance components, heritability, including the top-leaf chlorophyll index at 30 and 60 DAT) → **[3]** LDA with leave-one-genotype-out cross-validation, permutation null, validation comparison, trait-pair decision, per-group tests and a fixed 10-model comparison → **[4]** LDA diagnostics (Wilks' lambda, Box's M, canonical correlation), a QDA check and robustness checks (paired model comparison, precision-sensitivity, decision margins) → **[5]** the 10 main + 4 supplementary figures → **[6]** the tables → **[7]** the Source Data workbook. `06_tables.py` runs twice (stages [5] and [5b] in the script): pass 1 produces SuppTable\_S5, which the robustness-checks stage needs as its own input; pass 2 folds that stage's own outputs into SuppTable S6–S8.

## Data scope

This repository ships the raw plant-level matrix (`analysis/data/raw_LDA.xlsx`) together with every derived table the pipeline consumes or produces. Two of the shipped derived tables — `plants_30DAT.tsv` and `genotype_means.tsv` — are regenerated from the raw matrix by stages [1] above; the remaining derived tables (`genotype_tolerance.csv`, `trait_dictionary.tsv`, `anova_F_authors_table1.csv`, `genotype_means_full.tsv`) are shipped as pre-computed inputs rather than regenerated by this pipeline, and their provenance is documented in `MANIFEST.md`.

## Licence and citation

Code is released under the MIT Licence; data under CC-BY-4.0 (see `LICENSE`). Please cite the paper (details on publication) and this archive — see `CITATION.cff`.
