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


def build_preprocessor(numeric, categorical, cfg):
    """Raw feature frame -> clean feature frame: '?' -> NaN, impute (median / mode), clip to percentiles."""
    pp = cfg["preprocessing"]
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
    return Pipeline(steps).set_output(transform="pandas")
