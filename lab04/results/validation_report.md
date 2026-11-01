# Validation report

Input: raw snapshot, 48842 rows x 15 columns. **No rows were dropped or changed.**

Summary: PASS 43, INFO 3, WARN 3, FAIL 0
Row-level violations listed in `validation_violations.csv`: 52

| # | Check | Column | Violations | Status | Detail |
|---:|---|---|---:|---|---|
| 1 | column names | `income` | 1 | WARN | 'income' is called 'class' in the source; renamed in a copy (values are identical, raw snapshot untouched) |
| 2 | column set | `*` | 0 | PASS | 15 columns as in schema |
| 3 | column order | `*` | 0 | PASS | same order as schema |
| 4 | type int | `age` | 0 | PASS | dtype int64 |
| 5 | type category | `workclass` | 0 | PASS | dtype category |
| 6 | type int | `fnlwgt` | 0 | PASS | dtype int64 |
| 7 | type category | `education` | 0 | PASS | dtype category |
| 8 | type int | `education-num` | 0 | PASS | dtype int64 |
| 9 | type category | `marital-status` | 0 | PASS | dtype category |
| 10 | type category | `occupation` | 0 | PASS | dtype category |
| 11 | type category | `relationship` | 0 | PASS | dtype category |
| 12 | type category | `race` | 0 | PASS | dtype category |
| 13 | type category | `sex` | 0 | PASS | dtype category |
| 14 | type int | `capital-gain` | 0 | PASS | dtype int64 |
| 15 | type int | `capital-loss` | 0 | PASS | dtype int64 |
| 16 | type int | `hours-per-week` | 0 | PASS | dtype int64 |
| 17 | type category | `native-country` | 0 | PASS | dtype category |
| 18 | type category | `income` | 0 | PASS | dtype category |
| 19 | range | `age` | 0 | PASS | [17, 90], observed [17, 90] |
| 20 | range | `hours-per-week` | 0 | PASS | [1, 99], observed [1, 99] |
| 21 | range | `capital-gain` | 0 | PASS | [0, 99999], observed [0, 99999] |
| 22 | range | `capital-loss` | 0 | PASS | [0, 4356], observed [0, 4356] |
| 23 | missing not allowed | `age` | 0 | PASS | numeric/target column must be complete |
| 24 | missing ('?') | `workclass` | 0 | INFO | 2799 missing (5.73%): 0 literal '?', 2799 NaN (parsed from '?') -- allowed by null policy |
| 25 | missing not allowed | `fnlwgt` | 0 | PASS | numeric/target column must be complete |
| 26 | missing ('?') | `education` | 0 | PASS | 0 missing (0.00%): 0 literal '?', 0 NaN (parsed from '?') -- allowed by null policy |
| 27 | missing not allowed | `education-num` | 0 | PASS | numeric/target column must be complete |
| 28 | missing ('?') | `marital-status` | 0 | PASS | 0 missing (0.00%): 0 literal '?', 0 NaN (parsed from '?') -- allowed by null policy |
| 29 | missing ('?') | `occupation` | 0 | INFO | 2809 missing (5.75%): 0 literal '?', 2809 NaN (parsed from '?') -- allowed by null policy |
| 30 | missing ('?') | `relationship` | 0 | PASS | 0 missing (0.00%): 0 literal '?', 0 NaN (parsed from '?') -- allowed by null policy |
| 31 | missing ('?') | `race` | 0 | PASS | 0 missing (0.00%): 0 literal '?', 0 NaN (parsed from '?') -- allowed by null policy |
| 32 | missing ('?') | `sex` | 0 | PASS | 0 missing (0.00%): 0 literal '?', 0 NaN (parsed from '?') -- allowed by null policy |
| 33 | missing not allowed | `capital-gain` | 0 | PASS | numeric/target column must be complete |
| 34 | missing not allowed | `capital-loss` | 0 | PASS | numeric/target column must be complete |
| 35 | missing not allowed | `hours-per-week` | 0 | PASS | numeric/target column must be complete |
| 36 | missing ('?') | `native-country` | 0 | INFO | 857 missing (1.75%): 0 literal '?', 857 NaN (parsed from '?') -- allowed by null policy |
| 37 | missing not allowed | `income` | 0 | PASS | numeric/target column must be complete |
| 38 | allowed categories | `workclass` | 0 | PASS | 8 allowed + '?'; observed 8 distinct |
| 39 | allowed categories | `education` | 0 | PASS | 16 allowed + '?'; observed 16 distinct |
| 40 | allowed categories | `marital-status` | 0 | PASS | 7 allowed + '?'; observed 7 distinct |
| 41 | allowed categories | `occupation` | 0 | PASS | 14 allowed + '?'; observed 14 distinct |
| 42 | allowed categories | `relationship` | 0 | PASS | 6 allowed + '?'; observed 6 distinct |
| 43 | allowed categories | `race` | 0 | PASS | 5 allowed + '?'; observed 5 distinct |
| 44 | allowed categories | `sex` | 0 | PASS | 2 allowed + '?'; observed 2 distinct |
| 45 | allowed categories | `native-country` | 0 | PASS | 41 allowed + '?'; observed 41 distinct |
| 46 | allowed categories | `income` | 0 | PASS | 2 allowed + '?'; observed 2 distinct |
| 47 | exact duplicate rows | `*` | 52 | WARN | 101 rows involved in 49 groups |
| 48 | duplicate features, conflicting label | `*` | 5 | WARN | 5 feature groups carry both labels (audited in step 6) |
| 49 | education <-> education-num 1:1 | `education` | 0 | PASS | 16 education levels map to 16 codes |
