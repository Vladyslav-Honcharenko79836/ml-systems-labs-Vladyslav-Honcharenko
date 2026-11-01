"""Step 3: validation. Schema, type, range, missing-value, category, duplicate and consistency checks.

Reports every violation (results/validation_report.{md,json}, results/validation_violations.csv).
Rows are never dropped here: the function only reads the data.
"""
import json

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from common import RESULTS_DIR, load_config, load_raw, write_result

MAX_EXAMPLES = 5


def is_missing(s, token):
    """Null policy: '?' means missing. The OpenML parser already turns '?' into NaN, so count both."""
    return s.isna() | (s.astype("object") == token)


def validate(raw, cfg):
    schema = cfg["schema"]
    token = schema["missing_token"]
    target, source_target = cfg["target"]["column"], cfg["target"]["source_column"]
    checks, violations = [], []

    def check(name, column, n_bad, detail="", severity="error", rows=None, values=None):
        checks.append({"check": name, "column": column, "violations": int(n_bad),
                       "status": "PASS" if n_bad == 0 else ("WARN" if severity == "warn" else "FAIL"),
                       "detail": detail})
        if rows is not None:
            for r, v in zip(rows, values):
                violations.append({"row": int(r), "column": column, "value": v, "check": name})

    # 1. Columns: names, count, order.
    df = raw
    if target not in raw.columns and source_target in raw.columns:
        check("column names", target, 1, severity="warn",
              detail=f"'{target}' is called '{source_target}' in the source; renamed in a copy "
                     f"(values are identical, raw snapshot untouched)")
        df = raw.rename(columns={source_target: target})
    expected = list(schema["types"])
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    check("column set", "*", len(missing_cols) + len(extra_cols),
          detail=f"missing {missing_cols} extra {extra_cols}" if missing_cols or extra_cols
          else f"{len(expected)} columns as in schema")
    check("column order", "*", int(list(df.columns) != expected), severity="warn",
          detail="same order as schema" if list(df.columns) == expected else str(list(df.columns)))

    # 2. Types.
    for col, typ in schema["types"].items():
        s = df[col]
        if typ == "int":
            ok_dtype = pdt.is_integer_dtype(s)
            # A float column is still acceptable data if every value is whole; count the rest.
            n_bad = 0 if ok_dtype else (len(s) if not pdt.is_numeric_dtype(s)
                                        else int((s.notna() & (s != np.floor(s))).sum()) or 1)
            check("type int", col, n_bad, detail=f"dtype {s.dtype}")
        else:
            ok_dtype = isinstance(s.dtype, pd.CategoricalDtype) or pdt.is_object_dtype(s)
            check("type category", col, int(not ok_dtype), detail=f"dtype {s.dtype}")

    # 3. Ranges (inclusive).
    for col, (lo, hi) in schema["ranges"].items():
        s = df[col]
        bad = s[(s < lo) | (s > hi)]
        check("range", col, len(bad), rows=bad.index, values=bad.tolist(),
              detail=f"[{lo}, {hi}], observed [{s.min()}, {s.max()}]")

    # 4. Missing values. '?' is allowed (and expected) in categorical columns only.
    for col, typ in schema["types"].items():
        s = df[col]
        n_nan, n_tok = int(s.isna().sum()), int((s.astype("object") == token).sum())
        n_miss = n_nan + n_tok
        if typ == "category" and col != target:
            checks.append({"check": "missing ('?')", "column": col, "violations": 0,
                           "status": "INFO" if n_miss else "PASS",
                           "detail": f"{n_miss} missing ({n_miss / len(df):.2%}): {n_tok} literal '?', "
                                     f"{n_nan} NaN (parsed from '?') -- allowed by null policy"})
        else:
            bad = s[is_missing(s, token)]
            check("missing not allowed", col, len(bad), rows=bad.index, values=bad.tolist(),
                  detail="numeric/target column must be complete" if not len(bad) else "")

    # 5. Categories outside the data dictionary.
    for col, allowed in schema["allowed_categories"].items():
        s = df[col].astype("object")
        bad = s[~is_missing(s, token) & ~s.isin(allowed)]
        check("allowed categories", col, len(bad), rows=bad.index, values=bad.tolist(),
              detail=f"{len(allowed)} allowed + '{token}'; observed {s.nunique(dropna=True)} distinct"
                     + (f"; unknown {sorted(set(bad))[:MAX_EXAMPLES]}" if len(bad) else ""))
        unused = sorted(set(allowed) - set(s.dropna()))
        if unused:
            checks.append({"check": "unused categories", "column": col, "violations": 0,
                           "status": "INFO", "detail": f"never observed: {unused}"})

    # 6. Duplicates (reported, not dropped; Step 4 drops them from train only).
    features = [c for c in expected if c != target]
    dup_rows = df.duplicated(keep="first")
    check("exact duplicate rows", "*", dup_rows.sum(), severity="warn",
          rows=df.index[dup_rows], values=["duplicate of earlier row"] * int(dup_rows.sum()),
          detail=f"{int(df.duplicated(keep=False).sum())} rows involved in "
                 f"{int(df[df.duplicated(keep=False)].drop_duplicates().shape[0])} groups")
    dup_feat = df.duplicated(subset=features, keep=False)
    groups = df[dup_feat].groupby(features, observed=True, dropna=False)[target].nunique()
    n_conf = int((groups > 1).sum())
    check("duplicate features, conflicting label", "*", n_conf, severity="warn",
          detail=f"{n_conf} feature groups carry both labels (audited in step 6)")

    # 7. Cross-field consistency from the data dictionary: education <-> education-num is 1:1.
    pairs = df.groupby("education", observed=True)["education-num"].agg(["nunique", "min"])
    n_bad = int((pairs["nunique"] > 1).sum())
    n_num = df.groupby("education-num")["education"].nunique()
    check("education <-> education-num 1:1", "education", n_bad + int((n_num > 1).sum()),
          detail=f"{len(pairs)} education levels map to {df['education-num'].nunique()} codes")

    return checks, violations, df


def render_md(checks, violations, n_rows, n_cols):
    status = pd.Series([c["status"] for c in checks]).value_counts()
    lines = [
        "# Validation report",
        "",
        f"Input: raw snapshot, {n_rows} rows x {n_cols} columns. **No rows were dropped or changed.**",
        "",
        "Summary: " + ", ".join(f"{k} {status.get(k, 0)}" for k in ["PASS", "INFO", "WARN", "FAIL"]),
        f"Row-level violations listed in `validation_violations.csv`: {len(violations)}",
        "",
        "| # | Check | Column | Violations | Status | Detail |",
        "|---:|---|---|---:|---|---|",
    ]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c['check']} | `{c['column']}` | {c['violations']} | {c['status']} | "
                     f"{c['detail'].replace('|', '/')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = load_config()
    raw = load_raw()
    checks, violations, _ = validate(raw, cfg)
    md = render_md(checks, violations, *raw.shape)
    print(md)
    write_result("validation_report.md", md)
    write_result("validation_report.json", json.dumps(
        {"rows": raw.shape[0], "columns": raw.shape[1], "checks": checks,
         "row_violations": len(violations)}, indent=2, default=str))
    pd.DataFrame(violations, columns=["row", "column", "value", "check"]).to_csv(
        RESULTS_DIR / "validation_violations.csv", index=False)
