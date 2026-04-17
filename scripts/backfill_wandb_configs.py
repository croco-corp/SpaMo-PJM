"""Backfill W&B run configs from saved project YAML files.

For each log directory that has both a W&B run and a saved *-project.yaml,
this script loads the model/data params and uploads them to the corresponding
W&B run using the Public API.
"""

import glob
import os
import sys

import wandb
from omegaconf import OmegaConf

ENTITY = "croco-corp"
PROJECT = "lightning_logs"
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def build_experiment_config(project_yaml: str) -> dict:
    cfg = OmegaConf.load(project_yaml)
    experiment_config: dict = {}

    if "model" in cfg and "params" in cfg.model:
        experiment_config.update(
            OmegaConf.to_container(cfg.model.params, resolve=True)  # type: ignore[arg-type]
        )
    if "data" in cfg and "params" in cfg.data:
        experiment_config["data"] = OmegaConf.to_container(
            cfg.data.params, resolve=True
        )

    return experiment_config


def main():
    api = wandb.Api()

    log_dirs = sorted(glob.glob(os.path.join(LOGS_DIR, "*/")))
    updated = 0
    skipped = 0
    failed = 0

    for log_dir in log_dirs:
        run_id = os.path.basename(log_dir.rstrip("/"))

        # Find saved project config
        config_files = sorted(
            glob.glob(os.path.join(log_dir, "configs", "*-project.yaml"))
        )
        if not config_files:
            print(f"  SKIP (no config): {run_id}")
            skipped += 1
            continue

        project_yaml = config_files[-1]  # use most recent if multiple

        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        except wandb.errors.CommError:
            print(f"  SKIP (not in W&B): {run_id}")
            skipped += 1
            continue

        try:
            experiment_config = build_experiment_config(project_yaml)
            run.config.update(experiment_config)
            run.update()
            print(f"  OK: {run_id}  ({len(experiment_config)} keys)")
            updated += 1
        except Exception as e:
            print(f"  FAIL: {run_id} — {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
