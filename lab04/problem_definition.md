# Lab 4 — Problem definition

**Task.** Binary classification on the UCI Adult (Census Income) dataset, loaded with
`sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)` (OpenML id 1590,
48 842 rows). Predict whether a person's annual income is above \$50 000 from 14 demographic and
employment attributes recorded in the 1994 US Census.

**Inputs (14 features).**

* 6 numeric: `age`, `fnlwgt` (census sampling weight), `education-num`, `capital-gain`,
  `capital-loss`, `hours-per-week`.
* 8 categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`,
  `sex`, `native-country`.

**Output.** `income` ∈ {`<=50K`, `>50K`}; the positive class is `>50K`. OpenML names this column
`class`; the pipeline renames it to `income`.

**Success metrics.** **F1 of the positive class** (primary) and **accuracy** on a held-out test set.
The classes are imbalanced (about 24 % `>50K`), so a model that always predicts `<=50K` already
reaches ≈ 76 % accuracy with F1 = 0. Accuracy alone would hide that; F1 measures how well the
minority class is found. The reference model is a `LogisticRegression` baseline; the goal of this
lab is the data pipeline, not the model.

**Constraints.**

1. **No leakage.** Train/test split (70/30, stratified on `income`, seed 42) happens before any
   statistic is computed. Every fitted quantity — imputation values, clipping percentiles, scaler
   and encoder parameters, de-duplication — comes from the training split only. The test split is
   transformed with those train statistics and is otherwise left untouched.
2. **Reproducible.** The raw data is stored unmodified with its SHA256 hash; all parameters live in
   one `config.yaml`; seeds are fixed; package versions are pinned in `requirements.txt`; the
   processed dataset is identified by a SHA256 hash; and the whole pipeline runs with one command.
3. **Transparent data quality.** Schema, range, missing-value, duplicate and label-quality checks
   report every violation; no row is dropped silently.

**Out of scope.** Fairness mitigation and model tuning. Known biases (sex, race, 1994 data) are
documented in the data card, not corrected.
