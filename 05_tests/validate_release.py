#!/usr/bin/env python3
"""Validate copied inputs, interfaces and cached results without scientific reruns."""

from __future__ import annotations

import ast
import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_tool/03_runtime"))
from workflow import compute_only, config_for, settings, validate_inputs
from results import export_panels, folder, summary


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-generated-panels", action="store_true")
    args = parser.parse_args()
    compute_only()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "training_executed": False, "deg_recomputed": False, "live_api_called": False}
    parsed = []
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text())
        parsed.append(str(path.relative_to(ROOT)))
    report["python_files_parsed"] = len(parsed)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "05_tests"), pattern="test_*.py")
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (ROOT / "06_metadata/unit_tests.txt").write_text(log.getvalue())
    report["unit_tests"] = {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors)}
    if not tests.wasSuccessful():
        raise RuntimeError(log.getvalue())
    report["inputs"] = {d: validate_inputs(settings(config_for(d))) for d in ("ipf", "covid", "lung")}
    report["cli"] = []
    for d in ("ipf", "covid", "lung"):
        for command in (["inspect", "--dataset", d], ["plan", "--dataset", d, "--output", "/tmp/scbeacon_example_" + d], ["train", "--config", str(config_for(d)), "--output", "/tmp/scbeacon_train_example_" + d]):
            result = subprocess.run([sys.executable, str(ROOT / "scbeacon.py"), *command], text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            json.loads(result.stdout)
            report["cli"].append({"dataset": d, "command": command[0], "mode": "no_execution", "returncode": result.returncode})
    copies = pd.read_csv(ROOT / "06_metadata/source_copy_map.tsv", sep="\t")
    for row in copies.itertuples():
        if row.copy_mode == "independent_file" and not row.source_project_relative.startswith("release_staging/") and digest(ROOT / row.release_relative) != row.release_sha256:
            raise ValueError(f"Copied artifact hash mismatch: {row.release_relative}")
    report["independent_copy_hashes_checked"] = int(((copies.copy_mode == "independent_file") & ~copies.source_project_relative.str.startswith("release_staging/")).sum())
    report["panels"] = {}
    for d in ("ipf", "covid", "lung"):
        out = folder(d) / "06_individual_panels"
        if out.exists() and not args.refresh_generated_panels:
            raise FileExistsError(f"Reviewed panels already exist; use a new render destination: {out}")
        report["panels"][d] = export_panels(d, out)
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "anndata", "h5py", "scanpy", "torch", "scikit-learn", "nbformat", "nbclient", "ipykernel", "leidenalg", "igraph", "umap-learn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    environment = {"python": platform.python_version(), "packages": packages, "purpose": "cached notebook execution and interface/input validation; not the certified historical training environment"}
    (ROOT / "06_metadata/notebook_environment.json").write_text(json.dumps(environment, indent=2))
    report["execution_location"] = "compute_allocation" if os.environ.get("SLURM_JOB_ID") else "explicit_local_workstation"
    (ROOT / "06_metadata/validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
