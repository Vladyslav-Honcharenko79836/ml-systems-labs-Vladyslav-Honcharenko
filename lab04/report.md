# Lab 4 — Data pipeline and quality

## 1. Goal

Build a reproducible, leakage-free data pipeline for the UCI Adult dataset (mixed numeric and
categorical data): ingest the raw data unchanged, validate it against a schema, clean and transform
it with statistics from the training split only, engineer and evaluate derived features, audit
label quality, and finish with a hash of the processed data, a config file, a data card and a
single command that rebuilds everything.

| Learning outcome | Evidence in this report |
|---|---|
| Build a reproducible, leakage-free data pipeline for mixed numerical and categorical data | §2.4, §3.3 split before any statistic, train-only fit, test untouched; §3.6 one command, byte-identical rerun from scratch |
| Define and execute schema, range, missing-value, duplicate, and label-quality checks | §3.2 validation (49 checks); §3.5 duplicate-conflict audit and 50-row dictionary check; `results/validation_report.md`, `results/label_quality.txt` |
| Engineer and evaluate derived features without contaminating the test set | §3.4 `capital_net`, `hours_per_age`; test metrics + paired 5-fold CV on train |
| Produce a processed-data hash, configuration, and complete data card | §3.6 `results/data_version.yaml`, `config.yaml`, `data_card.md` |

## 2. Method

**Hardware and software.** AMD Ryzen 7 2700X, 16 GB RAM, Windows 10 Pro 19045. Python 3.11.8,
scikit-learn 1.5.0, pandas 2.2.2, numpy 1.26.4, pyarrow 16.1.0, PyYAML 6.0.3. Seed 42 everywhere.
Every parameter (seed, split, schema, ranges, imputation, clipping, features, model, audit size)
is in **`config.yaml`**; the scripts read it and hard-code nothing.

### 2.1 Problem definition (`problem_definition.md`)

Inputs: the 14 features (6 integer, 8 categorical). Output: `income` `>50K` (positive) vs
`<=50K`. Metrics: **F1 of `>50K`** (primary; the classes are 24/76, so "always `<=50K`" already
scores 76 % accuracy and F1 0) and **accuracy**. Constraints: no leakage, reproducible pipeline,
no silent data changes.

### 2.2 Ingestion (`src/ingest.py` → `results/ingest_log.txt`)

`fetch_openml(name='adult', version=2, as_frame=True)` with the download cache in
`data/lab04/openml/`. The frame is written as is to `data/lab04/raw/adult_raw.parquet`. It is
not renamed, cast or cleaned, and re-reading the file must give a frame equal to the fetched one
(`pd.testing.assert_frame_equal`, values, dtypes and categories). The step logs the row and
column count (and asserts 48 842 × 15), the SHA256 of the content, of the snapshot file and of the
downloaded ARFF file, and dtypes, NaN counts and the label balance.

### 2.3 Validation (`src/validate.py` → `results/validation_report.{md,json}`, `validation_violations.csv`)

Schema from the manual, stored in `config.yaml`:
types (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week `int`; the rest
`category`), inclusive ranges (age 17–90, hours 1–99, capital-gain 0–99 999, capital-loss
0–4 356), null policy `'?'` = missing and allowed in categorical columns, and the allowed category
values from the UCI data dictionary (`adult.names`). Checks: column names, set and order; dtype per
column; ranges; missing values (allowed in categorical, forbidden in numeric columns and the label);
values outside the dictionary; exact duplicate rows; duplicate features with conflicting labels;
education ↔ education-num one-to-one. Each check is PASS / INFO / WARN / FAIL with a count; every
offending row is listed in the CSV. **The function only reads the data; no row is dropped.** In the
one-command pipeline a FAIL stops the run.

### 2.4 Cleaning and transformation (`src/cleaning.py`, `src/preprocess.py` → `results/preprocessing.txt`, `results/preprocessor.joblib`)

Order chosen so that no statistic sees the test rows:

1. **Split first:** `train_test_split` 70/30, stratified on `income`, `random_state=42`, on the
   raw data.
2. **Duplicates:** exact duplicates (all 15 columns, keep first) dropped **from train only**. The
   test split is not de-duplicated, filtered or reordered (asserted).
3. **Preprocessor, fitted on train only** (one sklearn `Pipeline`, pandas output):
   `'?'`/NaN → missing → `SimpleImputer` **median** (numeric) / **most_frequent** (categorical) →
   `PercentileClipper`: clips each numeric column to its **train 1st / 99th percentile** → derived
   features (§2.5). The test split is only `transform`ed.
4. `joblib.dump` → `results/preprocessor.joblib`; reloaded and checked to give the identical
   transform. Processed train / test (16 features + label + `raw_row` back-reference) are written
   as CSV to `data/lab04/processed/`.

### 2.5 Derived features (`src/features.py` → `results/features.{txt,csv}`)

* `capital_net = capital-gain − capital-loss`
* `hours_per_age = hours-per-week / (age − 16)` (age ≥ 17, so the denominator is ≥ 1)

Both are computed row by row, after imputation and clipping, so they fit nothing and cannot leak.
Baseline: preprocessor → `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown='ignore')`
(categorical) → `LogisticRegression(C=1, max_iter=5000)`, the whole chain fitted on train. Four
variants: no derived feature, each one alone, both. Reported: test accuracy and F1, plus 5-fold
stratified CV **on the training split only**. The same folds are used for every variant, so
per-fold differences against the baseline are paired.

### 2.6 Label quality (`src/label_quality.py` → `results/label_quality.txt`, `label_conflicts.csv`, `label_audit_sample.csv`, `label_audit_review.md`)

Adult labels are objective census answers, so the subjective-label protocol (relabel after 24 h,
Cohen's κ) does not apply. Instead, as the manual prescribes for Adult:

* **Duplicate-conflict audit:** group all 48 842 rows by the 14 features (NaN equal to NaN) and
  count groups that carry both labels.
* **50 random rows** (`sample(50, random_state=42)`) checked against the data dictionary by 13
  rules (6 *impossible*, e.g. education-num ≠ dictionary code, Husband not Male, Never-worked with
  hours > 0; 7 *suspicious*, e.g. top-coded values, degree too young), and then read row by row.
  The same rules are run on all rows for context.

### 2.7 Hash, config, data card, one command (`src/pipeline.py`, `src/datacard_stats.py` → `results/data_version.yaml`, `data_card.md`)

`pipeline.py` runs §2.2–2.6, computes the data-card numbers, and writes `data_version.yaml`:
SHA256 of the processed dataset (train.csv bytes followed by test.csv bytes), of each file, of the
raw content and of `config.yaml`; row and column counts; the baseline metrics; package versions; and
a full copy of the config. No timestamps are written, so a rerun must reproduce every file exactly.

## 3. Results

### 3.1 Ingestion (`results/ingest_log.txt`)

**48 842 rows × 15 columns**, as expected. Label: `<=50K` 37 155 (76.07 %), `>50K` 11 687
(23.93 %). Raw content SHA256 `8f970e20…40adddc`; ARFF file `bd3946bc…6b8`.

Two differences from the manual's description, both handled without touching the raw file:

* The label column is called **`class`** in OpenML, not `income`. Later steps rename it in a copy.
* There are **no literal `'?'` values**: the OpenML ARFF parser turns them into NaN. The downloaded
  ARFF file contains exactly 6 465 `?` characters, the same as the NaN count in the frame
  (2 799 + 2 809 + 857), so the mapping is one-to-one and validation counts both forms as missing.

### 3.2 Validation (`results/validation_report.md`)

**49 checks: 43 PASS, 3 INFO, 3 WARN, 0 FAIL; 52 row-level findings, 0 rows dropped.**

| Check | Result |
|---|---|
| Columns | 15, in schema order; label named `class` → WARN (renamed in a copy) |
| Types | all 15 match (6 `int64`, 9 `category`) |
| Ranges | 0 violations. Observed min/max sit **exactly on the limits**: age 17–90, hours 1–99, capital-gain 0–99 999, capital-loss 0–4 356 |
| Missing (`'?'`) | `workclass` 2 799 (5.73 %), `occupation` 2 809 (5.75 %), `native-country` 857 (1.75 %): INFO, allowed by the policy. Numeric columns and the label: 0 |
| Categories | 0 values outside the data dictionary; all 41 countries etc. observed |
| Exact duplicates | **52 duplicate rows** (101 rows in 49 groups) → WARN, listed in the CSV |
| Duplicate features, conflicting label | **5 groups** → WARN, audited in §3.5 |
| education ↔ education-num | one-to-one (16 ↔ 16) |

The range maxima are census **top-codes**, not measurements: capital-gain = 99 999 in 244 rows,
age 90 in 55, hours 99 in 137. They pass the schema but are handled by clipping in §3.3 and
documented in the data card.

### 3.3 Cleaning and transformation (`results/preprocessing.txt`)

| | Train | Test |
|---|---:|---:|
| After split | 34 189 | 14 653 |
| `>50K` share | 0.2393 | 0.2393 (full data 0.2393) |
| Exact duplicates | **24 dropped** → 34 165 | 5 present, **kept** → 14 653 |
| Missing filled (workclass / occupation / native-country) | 1 923 / 1 933 / 590 | 873 / 873 / 267 |
| Missing after transform | 0 | 0 |

Train/test index overlap is 0. 24 test rows are identical (all 15 columns) to a train row. They
are different respondents with identical answers, not a processing leak, and removing them would
alter the test set, which the manual forbids.

**Fitted on train** (imputation values; the train median of `fnlwgt` is 178 272, against 178 144
on the full data, which shows that the statistic comes from train only). Modes: `Private`,
`Prof-specialty`, `United-States`.

| Column | Train p1 | Train p99 | Clipped in train (low / high) | Clipped in test (low / high) |
|---|---:|---:|---:|---:|
| age | 17 | 74 | 0 / 302 | 0 / 143 |
| fnlwgt | 27 035 | 509 629 | 342 / 341 | 127 / 145 |
| education-num | 3 | 16 | 240 / 0 | 88 / 0 |
| capital-gain | 0 | **15 024** | 0 / 255 | 0 / 136 |
| capital-loss | 0 | 1 980 | 0 / 340 | 0 / 151 |
| hours-per-week | 8 | 80 | 311 / 226 | 144 / 92 |

Clipping at p99 removes the 99 999 capital-gain sentinel (it becomes 15 024), and the age and
hours top-codes. For skewed columns that are mostly zero (capital-gain, -loss) the lower bound is
0, so only the upper tail is affected. `education-num` 1–2 (Preschool, 1st–4th) is raised to 3 by
the rule; the categorical `education` column still distinguishes them.

`preprocessor.joblib` (7 KB) is the fitted `Pipeline(missing → impute → clip → derive)`.
Reloading it reproduces the test transform exactly. Loading it needs `src/cleaning.py` on the
import path.

### 3.4 Derived features (`results/features.txt`)

| Variant | Test accuracy | Test F1 | CV accuracy (mean ± sd) | CV F1 (mean ± sd) | Paired ΔF1 vs none (CV) | Folds with higher F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority class (`<=50K`) | 0.7607 | 0.0000 | | | | |
| No derived features | 0.8540 | 0.6601 | 0.8507 ± 0.0037 | 0.6552 ± 0.0118 | — | — |
| + `capital_net` | 0.8542 | 0.6610 | 0.8508 ± 0.0040 | 0.6554 ± 0.0124 | +0.0002 | 2 / 5 |
| + `hours_per_age` | 0.8576 | 0.6694 | 0.8527 ± 0.0044 | 0.6615 ± 0.0126 | +0.0063 | **5 / 5** |
| **Both** | **0.8578** | **0.6696** | 0.8528 ± 0.0046 | 0.6618 ± 0.0129 | +0.0065 | **5 / 5** |

* **Both features together: accuracy +0.38 points, F1 +0.95 points on the test set.** Almost all of
  it comes from `hours_per_age`, which improves F1 in every one of the 5 training folds, so the gain
  is real even though it is smaller than the fold-to-fold spread of F1 (± 0.013).
* **`capital_net` adds nothing to a linear model,** and that is expected. It is an exact linear
  combination of two inputs the model already has (correlation with capital-gain 0.988). After
  standardisation the logistic regression can already form `w·gain − w·loss`. The +0.0002 is
  regularisation noise (2 of 5 folds better). It would matter for a model that cannot add features,
  such as a single decision tree split.
* **`hours_per_age` is non-linear** (a ratio), which the linear model cannot build from its inputs.
  Its coefficient is −1.78 (standardised): working many hours *for one's age* is typical of young
  people in low-paid jobs (age 17 with 40 h/week gives 40), while the same hours at 45 give 1.4.
* **No contamination.** The features use only the row's own values after a train-fitted cleaner.
  The comparison was decided on train-only CV; the test set was evaluated once per variant.

### 3.5 Label quality (`results/label_quality.txt`, `results/label_audit_review.md`)

**Duplicate-conflict audit.** 111 rows share their 14 features with another row (54 groups);
49 groups agree on the label, **5 groups / 10 rows have conflicting labels (0.02 % of the data)**:

| Group | Age | Education | Marital status | Occupation | Sex | h/week | Labels (split) |
|---:|---:|---|---|---|---|---:|---|
| 0 | 34 | Masters | Never-married | Exec-managerial | F | 50 | `<=50K` (test), `>50K` (train) |
| 1 | 39 | Bachelors | Married-civ-spouse | Craft-repair | M | 40 | `<=50K` (train), `>50K` (train) |
| 2 | 41 | HS-grad | Married-civ-spouse | Transport-moving | M | 40 | `<=50K` (test), `>50K` (train) |
| 3 | 46 | HS-grad | Married-civ-spouse | Transport-moving | M | 40 | `<=50K` (train), `>50K` (train) |
| 4 | 46 | HS-grad | Married-civ-spouse | Craft-repair | M | 40 | `<=50K` (train), `>50K` (test) |

All 10 rows are Private, United-States, with no capital income (4 groups White, 1 Black). These are people whose income really sits
near the \$50K threshold, or a real difference the 14 features do not capture. No model can
classify both rows of a group correctly, so this is a floor on the error, but a negligible one. 3 of
the 5 groups cross the train/test split; the test rows are kept as they are.

**50 random rows checked against the data dictionary: 0 impossible values, 0 labels judged
wrong.** 4 rows triggered *suspicious* rules and 3 more were noted by reading (7 unusual but
possible): two married people listed as Own-child / Other-relative of the household head, a divorced
40-year-old Own-child, two people aged 62 and 67 with missing workclass and occupation but positive
hours (imputation will give them `Private` / `Prof-specialty`), and one person working 4 h/week
(clipped to 8). Every education-num matched its level, every Husband was Male and every Wife Female
and married. 10 of the 50 labels are `>50K` (dataset 24 %).

**The same rules on all 48 842 rows** find 14 contradictions (0.03 %): 1 Husband who is Female,
3 Wives who are Male, and **10 `Never-worked` people who report 4–40 working hours a week**. A
further 4 have a degree implausible for their age (18 with a Masters, 19 with a Bachelors). They are
reported and kept. A random sample of 50 rows has a 1.4 % chance of containing one of the 14
contradictions, which is why the full-data rule scan is needed in addition to the sample.

The rule scan also shows the structure of the missing data: **all 2 799 rows with missing
workclass also miss occupation** (the other 10 missing occupations are the Never-worked rows), and
they are concentrated among the young and the old. See the data card.

### 3.6 Hash, config, data card, reproducibility (`results/data_version.yaml`, `data_card.md`)

```yaml
sha256:
  processed_dataset:   dd945be021c39f0662e2f231680e52846b983c4537b60fe3fe194d8bf93df41c
  processed_train_csv: 389b7fb65a18eddad7b0b38289b08aede648d64fa53c6d58fbeb1ea5831c6846
  processed_test_csv:  5b936f23548d0a03ebba7ea2680cefddea73f247dd681399571962e3b1613cf0
  raw_content:         8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc
  config_yaml:         ccfc2b594bb10cdbb8087cbe63aea8290a44230fa5484f41c826f38f51cbd028
```

followed by row counts, baseline metrics, package versions and the full config.

**One-command reproducibility, tested:** `data/lab04/` was deleted completely (OpenML cache, raw
snapshot, processed files) and `pipeline.py` was run again. It re-downloaded the data, and
**every file in `results/`, including `preprocessor.joblib` and the processed-data hash, was
byte-identical** (empty `git diff`). The first test found one non-reproducible item: the SHA256 of
OpenML's JSON *metadata* files changed, because sklearn gzips them locally with the download time
in the gzip header. The log now hashes only the data file (unchanged across downloads). Runtime
about 17 s plus the download.

**Data card** (`data_card.md`): source and citation, licence (CC BY 4.0 on UCI), size, collection
filters, quality findings, known biases (`>50K` rate 30.4 % for men vs 10.9 % for women; 85.5 %
White; 89.7 % US; 1994 dollars; missingness of workclass/occupation 24 % at ages 65+ vs 3.3 % at
25–64), privacy (no direct identifiers, but 32 % of rows are unique on 7 quasi-identifiers;
sensitive attributes), and intended use.

## 4. Conclusions

* The raw data (48 842 × 15) is ingested unchanged and fingerprinted. OpenML's version differs from
  the manual's description in two ways (`class` instead of `income`; NaN instead of `'?'`, checked
  one-to-one against the ARFF file), and both are handled in copies.
* Validation: 49 checks, 0 failures. The real issues are not schema violations: 52 exact duplicates,
  top-coded maxima (capital-gain 99 999), 5.7 % structured missingness, and 14 rows that contradict
  the data dictionary. All are reported; none were silently dropped.
* Leakage is prevented by order: split first, then de-duplicate train only, then fit imputer and
  clipper on train only. The test split keeps all 14 653 rows in the original order.
* `hours_per_age` raises test F1 from 0.660 to 0.669 (better in 5/5 folds); `capital_net` is
  redundant for a linear model because it is a linear combination of existing inputs. Final
  baseline: **accuracy 0.858, F1 0.670**.
* Label noise is small: 5 conflicting duplicate groups (10 rows, 0.02 %), and 0 problems in the
  50-row manual check.
* The whole pipeline runs with one command and reproduces every output byte for byte from an empty
  data directory; the processed dataset is identified by SHA256 `dd945be0…df41c`.

## 5. Reproduce

```powershell
# from the repository root, with the environment from README.md
.venv\Scripts\python lab04\src\pipeline.py
```

This one command runs all steps and rewrites everything in `lab04/results/`. Data goes to
`data/lab04/` (git-ignored). Single steps can be run on their own from `lab04/src`:
`ingest.py`, `validate.py`, `preprocess.py`, `features.py`, `label_quality.py`, `datacard_stats.py`.
