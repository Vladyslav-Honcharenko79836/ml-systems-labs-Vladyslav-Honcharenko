import argparse

import torch

from common import DEFAULT_CONFIG, env_line, load_config, pick_device, tee_results
from pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-output", default=None)
    args = ap.parse_args()

    if args.resume:
        cfg = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        cfg["train"]["resume"] = args.resume
    else:
        cfg = load_config(args.config, args.set)

    def go():
        print(env_line(pick_device(cfg["device"])))
        res = run(cfg)
        print(f"mlflow run id {res['run_id']}")

    if args.save_output:
        with tee_results(args.save_output):
            go()
    else:
        go()


if __name__ == "__main__":
    main()
