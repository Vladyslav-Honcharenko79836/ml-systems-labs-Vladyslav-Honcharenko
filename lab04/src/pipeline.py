"""Step 7: the whole Lab 4 pipeline in one command, ending with the processed-data hash.

    .venv\\Scripts\\python lab04\\src\\pipeline.py        (from the repository root)

ingest -> validate (stops on FAIL) -> split/clean/save preprocessor -> derived features + baseline
-> label-quality audit -> data-card numbers -> SHA256 of processed data, stored with the config in
results/data_version.yaml. No timestamps are written, so a rerun must reproduce every file.
"""
import hashlib
import json
import logging
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import yaml

import datacard_stats
import features
import ingest
import label_quality
import preprocess
import validate
from common import (CONFIG_PATH, PREPROCESSOR_PATH, PROCESSED_DIR, RAW_PATH, RESULTS_DIR,
                    load_config, load_raw, sha256_file, sha256_frame, write_result)

log = logging.getLogger("pipeline")


def data_version(cfg, feature_table):
    files = {name: PROCESSED_DIR / f"{name}.csv" for name in ["train", "test"]}
    frames = {name: pd.read_csv(path) for name, path in files.items()}
    combined = hashlib.sha256()
    for name in ["train", "test"]:
        combined.update(files[name].read_bytes())
    best = feature_table.set_index("variant").loc["with both derived"]
    return {
        "dataset": f"OpenML {cfg['dataset']['name']} v{cfg['dataset']['version']} (id 1590)",
        "sha256": {
            "processed_dataset": combined.hexdigest(),  # train.csv bytes followed by test.csv bytes
            "processed_train_csv": sha256_file(files["train"]),
            "processed_test_csv": sha256_file(files["test"]),
            "raw_content": sha256_frame(load_raw()),
            "config_yaml": sha256_file(CONFIG_PATH),
        },
        "processed_files": {name: {"path": f"data/lab04/processed/{name}.csv",
                                   "rows": len(frames[name]), "columns": frames[name].shape[1],
                                   "bytes": files[name].stat().st_size} for name in files},
        "baseline_logreg_test": {"accuracy": round(float(best["test_accuracy"]), 4),
                                 "f1": round(float(best["test_f1"]), 4)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "config": cfg,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()

    log.info("1/7 ingest")
    write_result("ingest_log.txt", ingest.ingest(cfg))

    log.info("2/7 validate")
    raw = load_raw()
    checks, violations, _ = validate.validate(raw, cfg)
    write_result("validation_report.md", validate.render_md(checks, violations, *raw.shape))
    write_result("validation_report.json", json.dumps(
        {"rows": raw.shape[0], "columns": raw.shape[1], "checks": checks,
         "row_violations": len(violations)}, indent=2, default=str))
    pd.DataFrame(violations, columns=["row", "column", "value", "check"]).to_csv(
        RESULTS_DIR / "validation_violations.csv", index=False)
    failed = [c for c in checks if c["status"] == "FAIL"]
    if failed:
        log.error("validation failed: %s", failed)
        sys.exit(1)
    log.info("validation: %d checks, 0 FAIL, %d row-level findings (reported, not dropped)",
             len(checks), len(violations))

    log.info("3/7 split, clean, save preprocessor")
    write_result("preprocessing.txt", preprocess.run(cfg)[0])

    log.info("4/7 derived features + LogisticRegression baseline")
    text, table = features.evaluate(cfg)
    write_result("features.txt", text)
    table.to_csv(RESULTS_DIR / "features.csv", index=False)

    log.info("5/7 label-quality audit")
    write_result("label_quality.txt", label_quality.run(cfg)[0])

    log.info("6/7 data-card numbers")
    write_result("data_card_stats.txt", datacard_stats.run(cfg))

    log.info("7/7 hash processed data")
    version = data_version(cfg, table)
    write_result("data_version.yaml", yaml.safe_dump(version, sort_keys=False, allow_unicode=True))
    log.info("processed dataset sha256 %s", version["sha256"]["processed_dataset"])
    log.info("done: raw %s, preprocessor %s", RAW_PATH.name, PREPROCESSOR_PATH.name)


if __name__ == "__main__":
    main()
