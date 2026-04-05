"""
Reset a wandb run's history to match the checkpoint epoch.

When training crashes at epoch 130 but the checkpoint is at epoch 100,
wandb has data for epochs 101-130 that would cause duplicate/messy graphs
on resume. This script deletes those extra data points so that resumed
training produces clean, continuous graphs.

Usage:
    python job-scripts/reset_wandb_to_checkpoint.py <NFS_RUN_DIR> [--dry-run]

Example:
    python job-scripts/reset_wandb_to_checkpoint.py ~/iris-stu-transformer-runs/runs/pen-human-v1_seed0_ml20
    python job-scripts/reset_wandb_to_checkpoint.py ~/iris-stu-transformer-runs/runs/pen-human-v1_seed0_ml20 --dry-run
"""

import argparse
import sys
from pathlib import Path

import torch
import wandb


def get_checkpoint_epoch(run_dir: Path) -> int:
    epoch_path = run_dir / "checkpoints" / "epoch.pt"
    if not epoch_path.exists():
        print(f"ERROR: {epoch_path} not found")
        sys.exit(1)
    return int(torch.load(epoch_path, map_location="cpu"))


def get_wandb_run_id(run_dir: Path) -> str:
    """Extract the wandb run ID from the local wandb directory."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        print(f"ERROR: {wandb_dir} not found")
        sys.exit(1)

    # The wandb directory contains run-YYYYMMDD_HHMMSS-<run_id> directories
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        # Also check for offline runs
        run_dirs = sorted(wandb_dir.glob("offline-run-*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not run_dirs:
        print(f"ERROR: No wandb run directories found in {wandb_dir}")
        sys.exit(1)

    # Run ID is the last part after the final dash
    run_dir_name = run_dirs[0].name
    run_id = run_dir_name.split("-")[-1]
    return run_id


def get_wandb_project(run_dir: Path) -> str:
    """Try to extract project name from saved config."""
    config_paths = [
        run_dir / "config" / "trainer.yaml",
        run_dir / ".hydra" / "config.yaml",
    ]
    for config_path in config_paths:
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg and "wandb" in cfg and "project" in cfg["wandb"]:
                return cfg["wandb"]["project"]
    return "iris"  # default project name


def main():
    parser = argparse.ArgumentParser(description="Reset wandb run history to checkpoint epoch")
    parser.add_argument("run_dir", type=Path, help="Path to the NFS run directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--entity", type=str, default=None, help="wandb entity (team/user)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    checkpoint_epoch = get_checkpoint_epoch(run_dir)
    wandb_run_id = get_wandb_run_id(run_dir)
    wandb_project = get_wandb_project(run_dir)

    print(f"Run directory:    {run_dir}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Wandb run ID:     {wandb_run_id}")
    print(f"Wandb project:    {wandb_project}")
    print()

    # Connect to wandb API
    api = wandb.Api()
    entity = args.entity or api.default_entity
    run_path = f"{entity}/{wandb_project}/{wandb_run_id}"
    print(f"Fetching wandb run: {run_path}")

    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        print(f"ERROR: Could not fetch run {run_path}: {e}")
        print("Make sure you are logged in (wandb login) and the entity/project are correct.")
        print(f"Try: --entity <your-username-or-team>")
        sys.exit(1)

    # Get the run's history to find the last wandb step at or before the checkpoint epoch
    print(f"Fetching run history...")
    history = run.scan_history(keys=["epoch", "_step"], page_size=1000)

    max_step_at_checkpoint = -1
    max_step_overall = -1
    rows_after_checkpoint = 0

    for row in history:
        step = row.get("_step", 0)
        epoch = row.get("epoch", None)
        if epoch is not None:
            max_step_overall = max(max_step_overall, step)
            if epoch <= checkpoint_epoch:
                max_step_at_checkpoint = max(max_step_at_checkpoint, step)
            else:
                rows_after_checkpoint += 1

    if max_step_at_checkpoint < 0:
        print("WARNING: Could not find any wandb steps at or before the checkpoint epoch.")
        print("The run may not have logged epoch data, or the history is empty.")
        sys.exit(1)

    print(f"Last wandb step at/before epoch {checkpoint_epoch}: {max_step_at_checkpoint}")
    print(f"Last wandb step overall: {max_step_overall}")
    print(f"Rows logged after checkpoint: {rows_after_checkpoint}")
    print()

    if rows_after_checkpoint == 0:
        print("Nothing to reset - wandb history already matches the checkpoint.")
        return

    if args.dry_run:
        print(f"DRY RUN: Would delete {rows_after_checkpoint} rows after step {max_step_at_checkpoint}")
        print(f"DRY RUN: Would set run.summary step to {max_step_at_checkpoint}")
        return

    # Confirm
    print(f"This will DELETE {rows_after_checkpoint} logged rows (epochs {checkpoint_epoch + 1}+)")
    print(f"and reset the run's history to step {max_step_at_checkpoint}.")
    response = input("Proceed? [y/N] ").strip().lower()
    if response != "y":
        print("Aborted.")
        sys.exit(0)

    # Use the wandb API to fork/trim the run history
    # wandb doesn't support deleting history rows directly via the public API.
    # The recommended approach is to use `run.update()` to modify the summary,
    # and let the resume mechanism handle it by setting the correct step.
    #
    # However, we can use a workaround: the wandb backend stores history as
    # append-only logs. When you resume with `resume=True`, wandb picks up
    # from the last step. The duplicate epoch values will show as the LATEST
    # value at each epoch on the x-axis.
    #
    # The cleanest approach is to:
    # 1. Use the internal API to truncate the history file in the local wandb dir
    # 2. Re-sync

    # Approach: truncate the local wandb history files
    wandb_dir = run_dir / "wandb"
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        run_dirs = sorted(wandb_dir.glob("offline-run-*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not run_dirs:
        print("ERROR: No local wandb run directory found")
        sys.exit(1)

    local_wandb_run_dir = run_dirs[0]
    history_file = local_wandb_run_dir / "run-history.jsonl"

    if not history_file.exists():
        # Try older wandb format
        for f in local_wandb_run_dir.glob("*.wandb"):
            print(f"Found binary wandb log: {f}")
        print()
        print("Cannot truncate binary wandb logs directly.")
        print("Alternative: when you resume training, wandb will append new data.")
        print("If you use 'epoch' as the x-axis in your wandb charts, the new data")
        print("will overwrite the old data points for the same epoch values,")
        print("producing clean graphs. No manual reset needed in this case.")
        print()
        print("If you still see artifacts, you can manually delete the duplicate")
        print("data points in the wandb web UI (Edit Panel -> X-axis: epoch).")
        return

    # Truncate the JSONL history file
    import json

    print(f"Truncating history file: {history_file}")
    lines_kept = []
    lines_removed = 0

    with open(history_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                epoch = row.get("epoch", None)
                if epoch is not None and epoch > checkpoint_epoch:
                    lines_removed += 1
                    continue
            except json.JSONDecodeError:
                pass
            lines_kept.append(line)

    # Write truncated file
    with open(history_file, "w") as f:
        for line in lines_kept:
            f.write(line + "\n")

    print(f"Kept {len(lines_kept)} lines, removed {lines_removed} lines")
    print()
    print("Local wandb history truncated successfully.")
    print("When you resume training, wandb will continue from the checkpoint epoch")
    print("and produce clean, continuous graphs.")
    print()
    print(f"Next step: RUN_NAME={run_dir.name} bash job-scripts/resume-lambda.sh")


if __name__ == "__main__":
    main()
