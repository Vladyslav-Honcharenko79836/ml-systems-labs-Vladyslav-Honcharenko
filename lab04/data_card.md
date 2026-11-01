# Data card — UCI Adult (Census Income), Lab 4 pipeline

| | |
|---|---|
| **Name** | Adult / Census Income |
| **Source** | UCI Machine Learning Repository, donated 1996 by R. Kohavi and B. Becker; extracted by Becker from the **1994 US Census** (Current Population Survey) database. Loaded from OpenML: `fetch_openml(name='adult', version=2, as_frame=True)`, dataset id 1590, <https://www.openml.org/d/1590> |
| **Citation** | Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. <https://doi.org/10.24432/C5XW20> |
| **License** | **CC BY 4.0** on the UCI repository (attribution required); OpenML lists the licence as "Public". |
| **Task** | Binary classification: `income` `>50K` vs `<=50K` per year (see `problem_definition.md`) |
| **Version used** | raw content SHA256 `8f970e20…40adddc`; OpenML ARFF file `1595261.gz` SHA256 `bd3946bc…6b8`; processed SHA256 `dd945be0…df41c` (full hashes in `results/data_version.yaml`) |

## Size and structure

| | Rows | Columns | File |
|---|---:|---:|---|
| Raw (as fetched) | 48 842 | 15 (14 features + label) | `data/lab04/raw/adult_raw.parquet`, 0.52 MB (ARFF download 0.66 MB gzip) |
| Processed train | 34 165 (34 189 − 24 exact duplicates) | 18 = `raw_row` + 16 features + `income` | `data/lab04/processed/train.csv`, 5.0 MB |
| Processed test | 14 653 (untouched) | 18 | `data/lab04/processed/test.csv`, 2.1 MB |

* Features: 6 integer (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`,
  `hours-per-week`), 8 categorical (`workclass`, `education`, `marital-status`, `occupation`,
  `relationship`, `race`, `sex`, `native-country`); 2 derived in the processed data
  (`capital_net`, `hours_per_age`).
* Label: `<=50K` 37 155 (76.1 %), `>50K` 11 687 (23.9 %). OpenML calls the column `class`.
* `fnlwgt` is the census **sampling weight** (how many people the row represents), not a property
  of the person. It is kept as a feature because the lab defines 14 inputs, but a model that uses
  it learns survey design, not income.
* `data/` is git-ignored; every file is rebuilt bit-for-bit by `python lab04/src/pipeline.py`.

## Collection and processing

1994 CPS survey answers; Becker kept records with age > 16, adjusted gross income > 100,
`fnlwgt` > 1 and hours-per-week > 0 (`adult.names`). Missing answers are recorded as `?` (the OpenML parser returns them as NaN):
`workclass` 2 799 (5.7 %), `occupation` 2 809 (5.8 %), `native-country` 857 (1.8 %).
Pipeline: split 70/30 stratified (seed 42) → drop exact duplicates in train → impute median /
mode → clip numeric columns to the train 1st/99th percentiles → add derived features. All fitted on
train only; the fitted transformer is `results/preprocessor.joblib`.

## Data quality (details in `results/`)

* Validation: 49 checks, 0 failures; all values inside the schema ranges and the data-dictionary
  categories (`validation_report.md`).
* **Top-coding:** `capital-gain` = 99 999 (244 rows) is a cap, not a real amount; `age` 90
  (55 rows) and `hours-per-week` 99 (137 rows) are also caps. Clipping to the 99th percentile
  removes these extremes.
* Duplicates: 52 exact duplicate rows (101 rows in 49 groups).
* **Label noise:** 5 groups / 10 rows have identical features and opposite labels (0.02 %).
* **Contradictions:** 14 rows break the data dictionary: 1 Husband who is Female, 3 Wives who are
  Male, 10 `Never-worked` people who work 4–40 h/week; 4 more have a degree that is implausible for
  their age (18 with a Masters). Kept, not corrected.
* **Missing not at random:** `workclass` and `occupation` are missing together, mostly for the
  young (12.2 % of ages 17–24) and the old (24.0 % of 65+; 3.3 % of ages 25–64). Those rows earn
  `>50K` 9.5 % of the time against 24.8 % otherwise. Mode imputation fills `Private` /
  `Prof-specialty`, which is likely wrong for many retirees.

## Known biases

* **Sex:** 66.8 % male. `>50K` rate is **30.4 % for men vs 10.9 % for women**. A model trained on
  this data reproduces the 1994 pay gap and will under-predict high income for women.
* **Race:** 85.5 % White, 9.6 % Black, 3.1 % Asian-Pac-Islander, 1.0 % Amer-Indian-Eskimo,
  0.8 % Other. `>50K` rate: White 25.4 %, Asian-Pac-Islander 26.9 %, Black 12.1 %,
  Amer-Indian-Eskimo 11.7 %. The two smallest groups have only about 400–500 rows each, so metrics on them are noisy.
* **Geography:** 89.7 % United-States; the other 40 countries share 8.5 %, many with < 100 rows.
* **Time:** 1994 dollars, not inflation-adjusted: \$50 000 then is more than \$100 000 today; occupations, education
  returns and household structure have changed. The data is not representative of today.
* **Population:** the CPS sample and Becker's filters (adults with positive work hours) leave out
  people outside the labour force, so the dataset is not representative even of 1994.
* **Label definition:** a binary threshold hides everything else about income.

## Privacy

* No direct identifiers (no names, addresses, IDs); the data is public and anonymised by the Census
  Bureau.
* **Quasi-identifiers remain.** On `age, sex, race, native-country, education, marital-status,
  occupation`, **15 620 rows (32.0 %) are unique** (k = 1) and 55.4 % sit in classes with k < 5.
  Linked with another source holding the same attributes, some records could be re-identified, and
  the income bracket would then be disclosed.
* **Sensitive attributes:** race, sex, marital status, country of origin. Using a model on this data
  to make decisions about individuals (credit, hiring, pricing) would be discriminatory and, in many
  jurisdictions, illegal; they are suitable for fairness auditing, not for decisions.
* Processed files and the fitted preprocessor contain no information beyond the public dataset.

## Intended use

Teaching and benchmarking of data pipelines and tabular models. Not suitable for decisions about
real people or for estimating present-day incomes.
