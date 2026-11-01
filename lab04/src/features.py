"""Step 5: derived features. LogisticRegression baseline with and without capital_net / hours_per_age.

Both variants use the same train/test split, the same cleaning fitted on train, and the same model.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cleaning import build_preprocessor
from common import (RESULTS_DIR, categorical_columns, feature_columns, load_config, numeric_columns,
                    write_result)
from preprocess import load_split


def build_model(cfg, derived):
    num, cat = numeric_columns(cfg), categorical_columns(cfg)
    encode = ColumnTransformer([
        ("num", StandardScaler(), num + list(derived)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([
        ("clean", build_preprocessor(num, cat, cfg, derived=derived)),
        ("encode", encode),
        ("model", LogisticRegression(max_iter=cfg["model"]["max_iter"], C=cfg["model"]["C"])),
    ])


def evaluate(cfg):
    _, train_raw, test_raw = load_split(cfg)
    feats, target, pos = feature_columns(cfg), cfg["target"]["column"], cfg["target"]["positive"]
    x_tr, x_te = train_raw[feats], test_raw[feats]
    y_tr, y_te = (train_raw[target] == pos).astype(int), (test_raw[target] == pos).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])

    variants = {
        "without derived": [],
        "+ capital_net": ["capital_net"],
        "+ hours_per_age": ["hours_per_age"],
        "with both derived": list(cfg["features"]["derived"]),
    }
    rows, folds = [], {}
    for name, derived in variants.items():
        model = build_model(cfg, derived).fit(x_tr, y_tr)
        pred = model.predict(x_te)
        # 5-fold CV on train only: the spread shows whether a test difference is larger than noise.
        scores = cross_validate(build_model(cfg, derived), x_tr, y_tr, cv=cv, scoring=["accuracy", "f1"])
        folds[name] = scores
        base = folds["without derived"]
        # Same folds for every variant, so per-fold differences are paired.
        d_acc = scores["test_accuracy"] - base["test_accuracy"]
        d_f1 = scores["test_f1"] - base["test_f1"]
        coefs = dict(zip(model.named_steps["encode"].get_feature_names_out(),
                         model.named_steps["model"].coef_[0]))
        rows.append({
            "variant": name, "n_numeric": len(numeric_columns(cfg)) + len(derived),
            "test_accuracy": accuracy_score(y_te, pred), "test_f1": f1_score(y_te, pred),
            "cv_accuracy_mean": scores["test_accuracy"].mean(), "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1_mean": scores["test_f1"].mean(), "cv_f1_std": scores["test_f1"].std(),
            "cv_paired_d_accuracy": d_acc.mean(), "cv_paired_d_f1": d_f1.mean(),
            "cv_paired_d_f1_std": d_f1.std(), "folds_f1_better": f"{int((d_f1 > 0).sum())}/5",
            "coef_capital_net": coefs.get("num__capital_net"),
            "coef_hours_per_age": coefs.get("num__hours_per_age"),
        })
    table = pd.DataFrame(rows)

    clean = build_preprocessor(numeric_columns(cfg), categorical_columns(cfg), cfg).fit(x_tr)
    d_tr = clean.transform(x_tr)[list(cfg["features"]["derived"])]
    lines = [
        f"train {len(x_tr)} (after dedup)  test {len(x_te)}  positive class '{pos}'",
        "model: clean (fit on train) -> StandardScaler(numeric) + OneHotEncoder(categorical) -> "
        f"LogisticRegression(C={cfg['model']['C']}, max_iter={cfg['model']['max_iter']})",
        "majority-class baseline on test: accuracy "
        f"{1 - y_te.mean():.4f}  F1 0.0000",
        "",
        table.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "cv_paired_d_*: mean over the 5 folds of (variant - without derived) on the same fold",
        "",
        "derived features on train (computed after imputation and clipping):",
        d_tr.describe().T.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
        "correlation on train: corr(capital_net, capital-gain) "
        f"{d_tr['capital_net'].corr(clean.transform(x_tr)['capital-gain']):.3f}; "
        "capital_net is an exact linear combination of two existing inputs",
    ]
    return "\n".join(lines), table


if __name__ == "__main__":
    text, table = evaluate(load_config())
    print(text)
    write_result("features.txt", text)
    table.to_csv(RESULTS_DIR / "features.csv", index=False)
