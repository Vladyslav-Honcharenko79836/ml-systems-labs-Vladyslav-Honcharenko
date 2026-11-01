# Manual re-check of 50 random rows (seed 42)

Rows: `label_audit_sample.csv` (drawn with `df.sample(50, random_state=42)` from all 48 842 rows).
Each row was read against the UCI `adult.names` data dictionary: value domains, education ↔
education-num code, relationship ↔ sex ↔ marital-status, age vs. degree, hours and capital values,
and whether the income label is plausible for the rest of the row. The automatic rules in
`label_quality.txt` were run first; the reading below adds what the rules cannot judge.

**Result: 0 impossible values, 0 labels judged wrong, 7 rows unusual but possible.**

| raw_row | Observation | Verdict |
|---:|---|---|
| 23881 | age 25, Married-civ-spouse, relationship Own-child | possible: married child living in the parents' household (the household head is the reference person) |
| 23288 | age 22, Married-spouse-absent, Own-child | possible, same reason |
| 47649 | age 27, Married-civ-spouse, Other-relative, Some-college, Transport-moving, **>50K** | possible; the least expected `>50K` label in the sample (self-employed, 50 h/week) but not contradictory |
| 7155 | age 40, Divorced, Own-child | possible: adult child back in the parents' household |
| 21063 | age 67, workclass and occupation `?`, but 60 h/week and capital-gain 9 386, >50K | missing values, not impossible ones; imputation will fill `Private` / `Prof-specialty`, which may be wrong for a 67-year-old |
| 18957 | age 62, workclass and occupation `?`, 24 h/week | as above |
| 3375 | 4 h/week, Divorced, Other-service, <=50K | possible (part-time); below the train 1st percentile (8 h), so the clipper raises it to 8 |

All other 43 rows: values inside the dictionary domains, education-num equals the dictionary code for
the education level, every Husband is Male and every Wife is Female and married, ages fit the
degrees (youngest Bachelors 24, youngest Masters 35), capital-gain and capital-loss are never both
non-zero, and the labels are plausible (e.g. both rows with capital gain ≥ 9 386 are `>50K`; the
10 `>50K` rows are all married, hold a degree, or both).

**Sample label balance:** 10 of 50 are `>50K` (20 %, dataset 24 %); a 50-row sample cannot detect
label noise at the 0.02 % rate found by the duplicate-conflict audit.

**Not applicable:** relabelling after 24 hours and Cohen's kappa. Adult labels are objective census
answers, not subjective judgements, so the manual prescribes the duplicate-conflict audit for Adult instead.

## Impossible values found by the same rules on all 48 842 rows (not in the sample)

| Rule | Rows | Example |
|---|---:|---|
| relationship Husband, sex Female | 1 | row 23390 |
| relationship Wife, sex Male | 3 | rows 5661, 16856, 43422 |
| workclass Never-worked but 4–40 hours per week | 10 | rows 8785, 48585, … |
| degree completed too young (18–20 with Masters, 19 with Bachelors) | 4 | rows 28464, 47333, 17851, 19872 (flagged *suspicious*) |

14 rows contradict the dictionary (0.03 %). They are kept: the manual says not to drop rows
silently, and they are too few to affect the model. They are listed in the data card.
