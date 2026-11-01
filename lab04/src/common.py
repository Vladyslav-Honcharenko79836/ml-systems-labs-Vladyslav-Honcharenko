"""Shared paths, config loading and hashing for the Lab 4 pipeline."""
import hashlib
from pathlib import Path

import pandas as pd
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
CONFIG_PATH = LAB_DIR / "config.yaml"
DATA_DIR = LAB_DIR.parent / "data" / "lab04"  # git-ignored
OPENML_DIR = DATA_DIR / "openml"               # fetch_openml cache
RAW_PATH = DATA_DIR / "raw" / "adult_raw.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSOR_PATH = RESULTS_DIR / "preprocessor.joblib"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_frame(df):
    """Content hash: independent of file format, depends only on column names, order and values."""
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def load_raw():
    """The ingested snapshot, exactly as fetched (OpenML column names, NaN for '?')."""
    return pd.read_parquet(RAW_PATH)


def load_named(cfg=None):
    """Raw snapshot with the target column renamed to the schema name (a copy; raw is not touched)."""
    cfg = cfg or load_config()
    return load_raw().rename(columns={cfg["target"]["source_column"]: cfg["target"]["column"]})


def feature_columns(cfg):
    return [c for c in cfg["schema"]["types"] if c != cfg["target"]["column"]]


def numeric_columns(cfg):
    return [c for c, t in cfg["schema"]["types"].items() if t == "int"]


def categorical_columns(cfg):
    return [c for c, t in cfg["schema"]["types"].items()
            if t == "category" and c != cfg["target"]["column"]]


def write_result(name, text):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / name).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
