#!/usr/bin/env python3
"""Plan or explicitly run the shared two-condition scBeacon training recipe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Explicitly trust and reuse this run's own saved state")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = json.loads(config_path.read_text())
    required = ["condition1_input", "condition2_input", "condition1_name", "condition2_name", "label_key_1", "label_key_2", "seed", "output_dir", "compatibility_profile"]
    for key in required:
        if key not in settings:
            parser.error(f"Missing configuration field: {key}")
    if settings["compatibility_profile"] not in {"ipf09", "covid09"}:
        parser.error("Unknown compatibility profile")
    if int(settings.get("iterations", 33)) != 33:
        parser.error("This preserved manuscript entry targets iteration33; other schedules are outside this recipe")
    bundle = args.bundle_root.resolve()
    core = bundle / "01_tool/01_shared_scbeacon_core"
    runner_path = bundle / "01_tool/02_training/run_balanced_seed_task.py"
    def local_path(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else config_path.parent / path).resolve()
    output = args.output.resolve() if args.output else local_path(settings["output_dir"])
    environment = {
        "SCBEACON_DATA_FILE_1": str(local_path(settings["condition1_input"])),
        "SCBEACON_DATA_FILE_2": str(local_path(settings["condition2_input"])),
        "SCBEACON_CONDITION_NAME_1": str(settings["condition1_name"]),
        "SCBEACON_CONDITION_NAME_2": str(settings["condition2_name"]),
        "SCBEACON_LABEL_KEY_1": str(settings["label_key_1"]),
        "SCBEACON_LABEL_KEY_2": str(settings["label_key_2"]),
        "SCBEACON_RANDOM_SEED": str(int(settings["seed"])),
        "SCBEACON_OUTPUT_ROOT": str(output),
        "SCBEACON_WORKSPACE_ROOT": str(config_path.parent),
        "SCBEACON_DATASET_ID": str(settings.get("dataset_id", "user_condition_pair")),
        "SCBEACON_INPUT_TRANSFORM": "log1p_raw",
        "SCBEACON_USE_RAW_DECODER_TARGET": "true",
        "SCBEACON_APPLY_LOGNORM": "false",
        "SCBEACON_COMBINE_PREPROCESS": "true",
        "SCBEACON_USE_HVG_SUBSET": "false",
        "SCBEACON_FILTER_MIN_GENES": "0",
        "SCBEACON_FILTER_MIN_CELLS": "0",
        "SCBEACON_MAX_CELLS_PER_CONDITION": "0",
        "SCBEACON_BATCH_SIZE": str(int(settings.get("batch_size", 512))),
        "SCBEACON_PRETRAIN_EPOCH": str(int(settings.get("pretrain_epochs", 40))),
        "SCBEACON_ANNO_EPOCH": str(int(settings.get("annotation_epochs", 20))),
        "SCBEACON_CONT_EPOCH": str(int(settings.get("contrastive_epochs", 5))),
        "SCBEACON_NUM_ITERATION": "33",
        "SCBEACON_USE_CUDA": "true" if settings.get("use_cuda", True) else "false",
        "SCBEACON_NUM_WORKERS": str(int(settings.get("num_workers", 1))),
        "SCBEACON_COMPATIBILITY_PROFILE": settings["compatibility_profile"],
        "SCBEACON_PRETRAIN_ONLY": "false",
    }
    print(json.dumps({"core": str(core), "historical_recipe": str(runner_path), "environment": environment, "execution": "requested" if args.execute else "plan_only"}, indent=2), flush=True)
    if not args.execute:
        return
    if not os.environ.get("SLURM_JOB_ID") and not os.environ.get("PBS_JOBID") and os.environ.get("SCBEACON_ALLOW_LOCAL_COMPUTE") != "1":
        raise RuntimeError("Use a compute allocation; analytical execution is not allowed on a login node")
    for alias in [str(bundle)]:
        if output.is_relative_to(Path(alias).resolve()):
            raise ValueError("Choose a new output directory outside the protected original project")
    for key in ["SCBEACON_DATA_FILE_1", "SCBEACON_DATA_FILE_2"]:
        if not Path(environment[key]).is_file():
            raise FileNotFoundError(environment[key])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty run directory; use a new directory or explicitly request --resume")
    output.mkdir(parents=True, exist_ok=True)
    for key in list(os.environ):
        if key.startswith("SCBEACON_"):
            del os.environ[key]
    os.environ.update(environment)
    runner = load_module(runner_path, "scbeacon_preserved_seed_recipe")
    seed = int(settings["seed"])
    summary = output / "05_selection_summary"
    summary.mkdir(exist_ok=True)
    config, embedding = runner.run_pretrain(seed, core, output)
    selected, _ = runner.load_or_compute_selection(config, embedding, seed, summary)
    runner.run_iteration_training(core, selected)
    target_dir = output / "03_contrastive/contrastive_gene_gene_sets"
    source = target_dir / "constractive_optimal_33.npy"
    alias = target_dir / "iter_33_contrastive_final.npy"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not alias.exists():
        shutil.copy2(source, alias)
    print(json.dumps({"selected_training_resolution": selected, "embedding": str(alias)}, indent=2))


if __name__ == "__main__":
    main()
