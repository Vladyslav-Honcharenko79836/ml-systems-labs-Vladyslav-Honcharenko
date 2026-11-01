"""Preprocessor building blocks. Kept in their own module so results/preprocessor.joblib can be loaded."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

MISSING_TOKEN = "?"


def normalize_missing(X):
    """Null policy: '?' (or NaN from the OpenML parser) means missing. Categories become plain strings."""
    X = X.copy()
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object:
            X[col] = X[col].astype("object").replace(MISSING_TOKEN, np.nan)
    return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [p_low, p_high] percentiles learned in fit (i.e. from the training split)."""

    def __init__(self, columns, percentiles=(1, 99)):
        self.columns = columns
        self.percentiles = percentiles

    def fit(self, X, y=None):
        lo, hi = self.percentiles
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.lower_ = X[self.columns].quantile(lo / 100)
        self.upper_ = X[self.columns].quantile(hi / 100)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].clip(self.lower_, self.upper_, axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)


DERIVED = {
    "capital_net": lambda X: X["capital-gain"] - X["capital-loss"],
    "hours_per_age": lambda X: X["hours-per-week"] / (X["age"] - 16),  # age >= 17, so never / 0
}


def add_derived(X, names):
    """Derived features, computed row by row from already imputed and clipped columns (nothing is fitted)."""
    return X.assign(**{n: DERIVED[n](X) for n in names})


def derived_names_out(transformer, input_features):
    return np.asarray(list(input_features) + list(transformer.kw_args["names"]), dtype=object)


def build_preprocessor(numeric, categorical, cfg, derived=None):
    """Raw feature frame -> clean feature frame: '?' -> NaN, impute (median / mode), clip to
    percentiles, add derived features. `derived` overrides cfg['features']['derived'] (e.g. [])."""
    pp = cfg["preprocessing"]
    derived = cfg["features"]["derived"] if derived is None else derived
    impute = ColumnTransformer(
        [("num", SimpleImputer(strategy=pp["numeric_impute"]), numeric),
         ("cat", SimpleImputer(strategy=pp["categorical_impute"]), categorical)],
        verbose_feature_names_out=False,
    )
    steps = [
        ("missing", FunctionTransformer(normalize_missing, feature_names_out="one-to-one")),
        ("impute", impute),
        ("clip", PercentileClipper(numeric, tuple(pp["clip_percentiles"]))),
    ]
    if derived:
        steps.append(("derive", FunctionTransformer(add_derived, kw_args={"names": list(derived)},
                                                    feature_names_out=derived_names_out)))
    return Pipeline(steps).set_output(transform="pandas")
