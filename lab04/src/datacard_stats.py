"""Numbers for data_card.md: group shares and positive rates (bias), missingness by group, re-identification risk."""
import pandas as pd

from common import load_config, load_named, write_result

QUASI_IDENTIFIERS = ["age", "sex", "race", "native-country", "education", "marital-status", "occupation"]


def run(cfg):
    df = load_named(cfg)
    target, pos = cfg["target"]["column"], cfg["target"]["positive"]
    y = df[target] == pos
    lines = [f"rows {len(df)}  overall {pos} rate {y.mean():.3f}", ""]
    for col in ["sex", "race"]:
        t = pd.DataFrame({"share": df[col].value_counts(normalize=True),
                          f"{pos} rate": y.groupby(df[col], observed=True).mean()})
        lines += [f"{col}:", t.sort_values("share", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"), ""]
    us = df["native-country"] == "United-States"
    lines += [f"native-country United-States {us.mean():.3f}, missing {df['native-country'].isna().mean():.3f}, "
              f"other {df['native-country'].nunique() - 1} countries {1 - us.mean() - df['native-country'].isna().mean():.3f}",
              ""]
    age_band = pd.cut(df["age"], [16, 24, 64, 90], labels=["17-24", "25-64", "65-90"])
    miss = df["workclass"].isna().groupby(age_band, observed=True).mean()
    lines += ["workclass/occupation missing rate by age band (missing not at random):",
              miss.to_string(float_format=lambda v: f"{v:.3f}"),
              f"{pos} rate when workclass missing {y[df['workclass'].isna()].mean():.3f} "
              f"vs present {y[df['workclass'].notna()].mean():.3f}", ""]
    sizes = df.groupby(QUASI_IDENTIFIERS, observed=True, dropna=False).size()
    lines += [f"re-identification: quasi-identifiers {QUASI_IDENTIFIERS}",
              f"  equivalence classes {len(sizes)};  rows unique on them (k = 1): {int((sizes == 1).sum())} "
              f"({(sizes == 1).sum() / len(df):.1%});  k < 5: {int(sizes[sizes < 5].sum())} rows "
              f"({sizes[sizes < 5].sum() / len(df):.1%})"]
    return "\n".join(lines)


if __name__ == "__main__":
    text = run(load_config())
    print(text)
    write_result("data_card_stats.txt", text)
