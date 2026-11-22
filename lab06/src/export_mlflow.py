import mlflow

from common import RESULTS_DIR, load_config, resolve, tee_results

PARAMS = ["device", "data.batch_size", "data.num_workers", "optim.name", "optim.lr", "sched.name",
          "train.epochs", "train.accum_steps"]
METRICS = ["test_acc", "best_val_acc", "val_acc", "peak_mem_mb", "samples_per_s", "epoch_s"]

if __name__ == "__main__":
    cfg = load_config()
    mlflow.set_tracking_uri("sqlite:///" + (resolve(cfg["paths"]["mlflow_dir"]) / "mlflow.db").as_posix())
    df = mlflow.search_runs(experiment_names=[cfg["experiment"]], order_by=["start_time ASC"])
    df = df[df["tags.mlflow.runName"] != "smoke"].drop_duplicates("tags.mlflow.runName", keep="last")
    cols = {"tags.mlflow.runName": "run", "status": "status"}
    cols |= {f"params.{p}": p for p in PARAMS if f"params.{p}" in df}
    cols |= {f"metrics.{m}": m for m in METRICS if f"metrics.{m}" in df}
    out = df[list(cols) + ["run_id"]].rename(columns=cols)
    out.to_csv(RESULTS_DIR / "mlflow_runs.csv", index=False)
    with tee_results("mlflow_runs.txt"):
        print(f"experiment '{cfg['experiment']}', tracking store {cfg['paths']['mlflow_dir']}/mlflow.db, "
              f"{len(out)} runs")
        print(out.drop(columns=["run_id"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
