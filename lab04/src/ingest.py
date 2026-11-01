"""Step 2: ingestion. Fetch UCI Adult from OpenML and store it unmodified, with hashes and a log."""
import logging

import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml

from common import OPENML_DIR, RAW_PATH, load_config, sha256_file, sha256_frame, write_result

log = logging.getLogger("ingest")


def ingest(cfg):
    ds = cfg["dataset"]
    bunch = fetch_openml(name=ds["name"], version=ds["version"], as_frame=True,
                         data_home=str(OPENML_DIR), parser="auto")
    raw = bunch.frame  # no cleaning, renaming or casting here

    n_rows, n_cols = raw.shape
    log.info("rows %d  columns %d", n_rows, n_cols)
    assert (n_rows, n_cols) == (ds["expected_rows"], ds["expected_cols"]), (n_rows, n_cols)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_PATH, index=False)
    # The stored snapshot must be identical to what was fetched: values, dtypes, categories.
    pd.testing.assert_frame_equal(pd.read_parquet(RAW_PATH), raw)

    # Only the data file: the JSON metadata files are gzipped locally by sklearn with the download
    # time in the gzip header, so their hashes change on every fresh download.
    source_files = sorted(p for p in OPENML_DIR.rglob("data/v1/download/*") if p.is_file())
    target = cfg["target"]["source_column"]
    lines = [
        f"source: OpenML '{ds['name']}' version {ds['version']} (id {bunch.details['id']}), "
        f"licence {bunch.details.get('licence')}, sklearn {sklearn.__version__}",
        f"url: https://www.openml.org/d/{bunch.details['id']}",
        f"rows {n_rows}  columns {n_cols}  (expected {ds['expected_rows']} x {ds['expected_cols']}: OK)",
        f"snapshot: {RAW_PATH.relative_to(RAW_PATH.parents[3]).as_posix()}  "
        f"round-trip identical to fetched frame: OK",
        f"sha256 content (CSV of the frame): {sha256_frame(raw)}",
        f"sha256 snapshot file:              {sha256_file(RAW_PATH)}",
        "downloaded OpenML data file (ARFF, gzip; sha256):",
        *[f"  {sha256_file(p)}  {p.relative_to(OPENML_DIR).as_posix()}" for p in source_files],
        "",
        f"{'column':<16}{'dtype':<10}{'non-null':>9}{'NaN':>6}",
        *[f"{c:<16}{str(raw[c].dtype):<10}{raw[c].notna().sum():>9}{raw[c].isna().sum():>6}"
          for c in raw.columns],
        "",
        f"target '{target}': " + "  ".join(f"{k} {v} ({v / n_rows:.2%})"
                                         for k, v in raw[target].value_counts().items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    text = ingest(load_config())
    print(text)
    write_result("ingest_log.txt", text)
