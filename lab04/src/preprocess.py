"""Step 4: split, de-duplicate train, fit imputer + clipper on train only, transform both, save.

Order matters for leakage: the split happens on the raw data, before any statistic is computed.
Test rows are never dropped or reordered; they are only transformed with train statistics.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (PREPROCESSOR_PATH, PROCESSED_DIR, categorical_columns, feature_columns,
                    load_config, load_named, numeric_columns, sha256_frame, write_result)
from cleaning import build_preprocessor


def split(df, cfg):
    y = df[cfg["target"]["column"]]
    return train_test_split(df, test_size=cfg["split"]["test_size"], random_state=cfg["seed"],
                            stratify=y if cfg["split"]["stratify"] else None)


def run(cfg):
    target = cfg["target"]["column"]
    feats, num, cat = feature_columns(cfg), numeric_columns(cfg), categorical_columns(cfg)
    df = load_named(cfg)
    train_raw, test_raw = split(df, cfg)
    lines = [f"split: train {len(train_raw)}  test {len(test_raw)}  "
             f"(test_size {cfg['split']['test_size']}, stratified, seed {cfg['seed']})",
             f"train/test index overlap: {len(set(train_raw.index) & set(test_raw.index))}"]
    for name, part in [("full", df), ("train", train_raw), ("test", test_raw)]:
        lines.append(f"  {name:<5} {target} >50K share {(part[target] == cfg['target']['positive']).mean():.4f}")

    # Duplicates: exact (all 15 columns) duplicates dropped from train only.
    n_before = len(train_raw)
    if cfg["preprocessing"]["drop_train_duplicates"]:
        train_raw = train_raw[~train_raw.duplicated(keep="first")]
    test_dups = int(test_raw.duplicated(keep="first").sum())
    cross = test_raw.merge(train_raw.drop_duplicates(), on=list(test_raw.columns), how="inner")
    lines += [
        f"train exact duplicates dropped: {n_before - len(train_raw)}  -> train {len(train_raw)}",
        f"test exact duplicates kept (test untouched): {test_dups}  -> test {len(test_raw)}",
        f"test rows identical to a train row (all 15 columns): {len(cross)}",
    ]

    # Fit on train only.
    pre = build_preprocessor(num, cat, cfg)
    train_x = pre.fit_transform(train_raw[feats])
    test_x = pre.transform(test_raw[feats])
    assert list(test_x.index) == list(test_raw.index), "test rows must not be dropped or reordered"

    imp = pre.named_steps["impute"]
    clip = pre.named_steps["clip"]
    stats = dict(zip(num + cat, list(imp.named_transformers_["num"].statistics_)
                     + list(imp.named_transformers_["cat"].statistics_)))
    train_n = train_raw[feats].copy()
    lines += ["", "imputation values (fitted on train) vs the same statistic on full data:"]
    for c in num:
        lines.append(f"  {c:<16} median train {stats[c]:>10g}   full {df[c].median():>10g}")
    for c in cat:
        full_mode = df[c].mode().iloc[0]
        n_tr, n_te = int(train_n[c].isna().sum()), int(test_raw[c].isna().sum())
        lines.append(f"  {c:<16} mode   train {stats[c]:<18} full {full_mode:<18} "
                     f"filled train {n_tr}  test {n_te}")

    lines += ["", f"clipping bounds = train percentiles {cfg['preprocessing']['clip_percentiles']}; "
                  "values clipped (low / high):",
              f"  {'column':<16}{'p1':>10}{'p99':>10}{'train low':>11}{'high':>7}{'test low':>10}{'high':>7}"]
    for c in num:
        lo, hi = clip.lower_[c], clip.upper_[c]
        tr, te = train_raw[c], test_raw[c]
        lines.append(f"  {c:<16}{lo:>10g}{hi:>10g}{(tr < lo).sum():>11}{(tr > hi).sum():>7}"
                     f"{(te < lo).sum():>10}{(te > hi).sum():>7}")

    lines += ["", f"missing after transform: train {int(train_x.isna().sum().sum())}  "
                  f"test {int(test_x.isna().sum().sum())}",
              f"output columns ({train_x.shape[1]}): {list(train_x.columns)}"]

    PREPROCESSOR_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pre, PREPROCESSOR_PATH)
    reloaded = joblib.load(PREPROCESSOR_PATH)
    pd.testing.assert_frame_equal(reloaded.transform(test_raw[feats]), test_x)
    lines.append(f"saved {PREPROCESSOR_PATH.name}; reloaded transform of test identical: OK")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, x, raw in [("train", train_x, train_raw), ("test", test_x, test_raw)]:
        frame = x.assign(**{target: raw[target].astype("object")}).rename_axis("raw_row").reset_index()
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, lineterminator="\n")
        out[name] = frame
        lines.append(f"processed {name}: {frame.shape}  content sha256 {sha256_frame(frame)[:16]}...")
    return "\n".join(lines), pre, out


if __name__ == "__main__":
    text, *_ = run(load_config())
    print(text)
    write_result("preprocessing.txt", text)
