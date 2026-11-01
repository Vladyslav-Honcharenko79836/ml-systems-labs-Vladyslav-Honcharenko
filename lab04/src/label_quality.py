"""Step 6: label quality. Duplicate-conflict audit + 50-row check against the data dictionary.

Adult labels are objective (census income), so the subjective-label protocol (relabel after 24 h,
Cohen's kappa) does not apply. Instead:
  (a) duplicate-conflict audit: identical 14-feature rows that carry different income labels;
  (b) 50 random rows checked against the data dictionary for impossible or suspicious values,
      by the rules below and by reading each row (notes in report.md, sample in label_audit_sample.csv).
Nothing is changed or dropped; the findings are reported.
"""
import pandas as pd

from common import RESULTS_DIR, feature_columns, load_config, load_named, write_result
from preprocess import split

# UCI adult.names: education level -> education-num.
EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
           "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
           "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}
# Youngest plausible age for a completed degree (18 at high-school graduation + programme length).
MIN_AGE = {"Bachelors": 20, "Masters": 21, "Prof-school": 22, "Doctorate": 24}
MARRIED = {"Married-civ-spouse", "Married-AF-spouse"}

RULES = {
    # impossible: contradict the data dictionary or each other
    "education-num != dictionary code": ("impossible",
        lambda d: d["education-num"] != d["education"].astype(str).map(EDU_NUM)),
    "Husband but not Male": ("impossible",
        lambda d: (d["relationship"] == "Husband") & (d["sex"] != "Male")),
    "Wife but not Female": ("impossible",
        lambda d: (d["relationship"] == "Wife") & (d["sex"] != "Female")),
    "Husband/Wife but not married": ("impossible",
        lambda d: d["relationship"].isin(["Husband", "Wife"]) & ~d["marital-status"].isin(MARRIED)),
    "capital-gain and capital-loss both > 0": ("impossible",
        lambda d: (d["capital-gain"] > 0) & (d["capital-loss"] > 0)),
    "Never-worked with an occupation": ("impossible",
        lambda d: (d["workclass"] == "Never-worked") & d["occupation"].notna()),
    "Never-worked but hours-per-week > 0": ("impossible",
        lambda d: (d["workclass"] == "Never-worked") & (d["hours-per-week"] > 0)),
    # suspicious: possible, but worth a look
    "degree too young for its length": ("suspicious",
        lambda d: d["age"] < d["education"].astype(str).map(MIN_AGE).fillna(0)),
    "capital-gain = 99999 (top-code)": ("suspicious", lambda d: d["capital-gain"] == 99999),
    "hours-per-week = 99 (top-code)": ("suspicious", lambda d: d["hours-per-week"] == 99),
    "age = 90 (top-code)": ("suspicious", lambda d: d["age"] == 90),
    "workclass and occupation missing, hours > 0": ("suspicious",
        lambda d: d["workclass"].isna() & d["occupation"].isna() & (d["hours-per-week"] > 0)),
    "married but Own-child / Other-relative": ("suspicious",
        lambda d: d["marital-status"].isin(MARRIED) & d["relationship"].isin(["Own-child", "Other-relative"])),
}


def apply_rules(d):
    return pd.DataFrame({name: fn(d).fillna(False).astype(bool) for name, (_, fn) in RULES.items()},
                        index=d.index)


def conflict_audit(df, feats, target):
    dup = df[df.duplicated(subset=feats, keep=False)]
    labels = dup.groupby(feats, observed=True, dropna=False)[target].agg(["nunique", "size"])
    conflicts = labels[labels["nunique"] > 1]
    keys = conflicts.reset_index()[feats]
    rows = df.reset_index(names="raw_row").merge(keys, on=feats, how="inner")
    return dup, labels, conflicts, rows


def run(cfg):
    target, feats = cfg["target"]["column"], feature_columns(cfg)
    df = load_named(cfg)
    train_raw, test_raw = split(df, cfg)

    # (a) duplicate-conflict audit on the whole dataset (an audit reads labels; it fits nothing).
    dup, labels, conflicts, conf_rows = conflict_audit(df, feats, target)
    conf_rows["split"] = conf_rows["raw_row"].map(
        lambda r: "train" if r in train_raw.index else "test")
    conf_rows = conf_rows.sort_values(feats + [target]).reset_index(drop=True)
    conf_rows.insert(0, "group", conf_rows.groupby(feats, observed=True, dropna=False).ngroup())
    conf_rows.to_csv(RESULTS_DIR / "label_conflicts.csv", index=False)
    lines = [
        "(a) duplicate-conflict audit (14 features identical, NaN = NaN), whole dataset",
        f"  rows whose 14 features occur more than once: {len(dup)} in {len(labels)} groups",
        f"  groups with a single label: {int((labels['nunique'] == 1).sum())}",
        f"  groups with CONFLICTING labels: {len(conflicts)}  "
        f"({len(conf_rows)} rows; {conf_rows['split'].value_counts().to_dict()})",
        f"  conflict rate: {len(conf_rows) / len(df):.4%} of all rows",
        "",
        conf_rows[["group", "raw_row", "split", "age", "workclass", "education", "marital-status",
                   "occupation", "sex", "hours-per-week", "native-country", target]]
        .to_string(index=False),
    ]
    split_groups = conf_rows.groupby("group")["split"].nunique()
    lines.append(f"  conflict groups spread across train and test: {int((split_groups > 1).sum())}")

    # (b) 50 random rows against the data dictionary, plus the same rules on all rows for context.
    n = cfg["label_audit"]["sample_size"]
    sample = df.sample(n=n, random_state=cfg["seed"])
    flags_s, flags_all = apply_rules(sample), apply_rules(df)
    out = sample.copy()
    out.insert(0, "raw_row", sample.index)
    out["rule_flags"] = flags_s.apply(lambda r: "; ".join(r.index[r]), axis=1)
    out.to_csv(RESULTS_DIR / "label_audit_sample.csv", index=False)

    lines += ["", f"(b) {n} random rows (seed {cfg['seed']}) checked against the data dictionary",
              f"  sample >50K share {(sample[target] == cfg['target']['positive']).mean():.2f}  "
              f"(dataset {(df[target] == cfg['target']['positive']).mean():.2f})",
              f"  {'rule':<52}{'kind':<12}{'sample':>7}{'all rows':>10}"]
    for name, (kind, _) in RULES.items():
        lines.append(f"  {name:<52}{kind:<12}{flags_s[name].sum():>7}{flags_all[name].sum():>10}")
    imp = [k for k, (kind, _) in RULES.items() if kind == "impossible"]
    lines += [f"  rows with any impossible value: sample {int(flags_s[imp].any(axis=1).sum())}  "
              f"all {int(flags_all[imp].any(axis=1).sum())}",
              f"  rows with any flag: sample {int(flags_s.any(axis=1).sum())}  "
              f"all {int(flags_all.any(axis=1).sum())}",
              "", "flagged rows in the sample:"]
    flagged = out[out["rule_flags"] != ""]
    lines.append(flagged[["raw_row", "age", "education", "relationship", "sex", "workclass",
                          "occupation", "capital-gain", "capital-loss", "hours-per-week", target,
                          "rule_flags"]].to_string(index=False) if len(flagged) else "  none")
    return "\n".join(lines), sample


if __name__ == "__main__":
    text, sample = run(load_config())
    print(text)
    write_result("label_quality.txt", text)
