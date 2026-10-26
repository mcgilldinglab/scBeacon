"""Explicit, non-destructive interfaces for prepared-input scBeacon workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"ipf": "01_ipf", "covid": "02_covid", "lung": "03_lung"}

# Keep JIT cache writes out of the immutable source tree and shared environments.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / f"scbeacon_numba_{os.getuid()}"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config_for(dataset):
    return ROOT / "03_reproduction" / DATASETS[dataset] / "01_configuration/dataset.json"


def settings(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for key in ("condition1_input", "condition2_input", "combined_lognorm", "embedding", "reference_dir"):
        if key in cfg:
            value = Path(cfg[key]).expanduser()
            cfg[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    return cfg


def compute_only():
    if os.environ.get("SCBEACON_ALLOW_LOCAL_COMPUTE") == "1":
        return
    if not os.environ.get("SLURM_JOB_ID") and not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Use a compute allocation, or explicitly set SCBEACON_ALLOW_LOCAL_COMPUTE=1 on your own workstation. Never enable it on a login node.")


def fresh_output(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Outputs must be outside the immutable release directory")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_inputs(cfg):
    import anndata as ad
    import numpy as np
    from scipy import sparse

    matrices = [ad.read_h5ad(cfg[k]) for k in ("condition1_input", "condition2_input", "combined_lognorm")]
    first, second, combined = matrices
    if not first.var_names.equals(second.var_names) or not first.var_names.equals(combined.var_names):
        raise ValueError("Ordered feature identifiers must be identical in all three matrices")
    expected_ids = []
    for matrix, name, label in zip(matrices[:2], (cfg["condition1_name"], cfg["condition2_name"]), (cfg["label_key_1"], cfg["label_key_2"])):
        if not matrix.obs_names.is_unique or not matrix.var_names.is_unique:
            raise ValueError("Cell and feature identifiers must be unique within each input")
        if label not in matrix.obs:
            raise KeyError(f"Missing annotation column used by the preserved training resolution recipe: {label}")
        values = matrix.X.data if sparse.issparse(matrix.X) else np.asarray(matrix.X)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values)):
            raise ValueError("Training inputs must contain nonnegative integer raw counts in .X")
        expected_ids.extend(x if x.startswith(name + "::") else name + "::" + x for x in matrix.obs_names.astype(str))
    if combined.obs_names.astype(str).tolist() != expected_ids:
        raise ValueError("combined_lognorm must preserve condition1 then condition2 cell order, with condition:: prefixes")
    embedding = np.load(cfg["embedding"], mmap_mode="r") if cfg.get("embedding") else None
    if embedding is not None and (embedding.ndim != 2 or len(embedding) != len(expected_ids) or not np.isfinite(embedding).all()):
        raise ValueError("Embedding must be finite, two-dimensional, and aligned to all input cells")
    return {"condition1_cells": first.n_obs, "condition2_cells": second.n_obs, "n_genes": first.n_vars, "embedding_shape": list(embedding.shape) if embedding is not None else None, "ordered_cell_ids_verified": True}


def pairs_and_deg(cfg, output):
    import anndata as ad
    import numpy as np

    module = load_module(ROOT / "01_tool/04_downstream/formal_pairs.py", "formal_pairs")
    first = ad.read_h5ad(cfg["condition1_input"])
    second = ad.read_h5ad(cfg["condition2_input"])
    combined = ad.read_h5ad(cfg["combined_lognorm"])
    combined.obs["condition"] = [cfg["condition1_name"]] * first.n_obs + [cfg["condition2_name"]] * second.n_obs
    z = np.load(cfg["embedding"])
    a = module.cluster_embedding(z[:first.n_obs], 1.0, 0)
    b = module.cluster_embedding(z[first.n_obs:], 1.0, 0)
    assignments = module.write_assignments(output, module.prefixed_cell_ids(first, cfg["condition1_name"]), module.prefixed_cell_ids(second, cfg["condition2_name"]), a, b, first, second, cfg["label_key_1"], cfg["label_key_2"], cfg["condition1_name"], cfg["condition2_name"])
    assignments["mode"] = f"scbeacon_seed{cfg['seed']}_iteration33_cluster_resolution1"
    assignments.to_csv(output / "combined_condition_assignments.tsv.gz", sep="\t", index=False)
    _, _, _, formal, _ = module.build_pair_maps(z[:first.n_obs], a, z[first.n_obs:], b)
    formal.to_csv(output / "formal_pairs.tsv", sep="\t", index=False)
    (output / "run.json").write_text(json.dumps(cfg, indent=2))
    module.rank_wilcoxon_for_pairs(combined, assignments, formal, output / "02_deg", cfg["condition1_name"], cfg["condition2_name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "plan", "render"):
        child = commands.add_parser(name)
        child.add_argument("--dataset", choices=DATASETS, required=True)
        if name == "inspect":
            child.add_argument("--inputs", action="store_true")
        else:
            child.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--execute", action="store_true")
    plot_run = commands.add_parser("plot-run")
    plot_run.add_argument("--config", type=Path, required=True)
    plot_run.add_argument("--pair-run", type=Path, required=True)
    plot_run.add_argument("--qc-run", type=Path, required=True)
    plot_run.add_argument("--marker-run", type=Path, required=True)
    plot_run.add_argument("--enrichment-run", type=Path, required=True)
    plot_run.add_argument("--output", type=Path, required=True)
    plot_run.add_argument("--execute", action="store_true")
    for name in ("prepare", "pairs-deg", "qc", "markers", "enrich"):
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--execute", action="store_true")
        if name == "pairs-deg":
            child.add_argument("--embedding", type=Path)
        elif name != "prepare":
            child.add_argument("--pair-run", type=Path, required=True)
        if name in ("markers", "enrich"):
            child.add_argument("--clean-pairs", type=Path, required=True)
        if name == "enrich":
            child.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        cfg = settings(config_for(args.dataset))
        if args.inputs:
            compute_only()
            print(json.dumps(validate_inputs(cfg), indent=2))
        else:
            from results import summary
            print(json.dumps(summary(args.dataset), indent=2))
        return
    if args.command == "plan":
        c = str(config_for(args.dataset))
        out = str(args.output.resolve())
        base = [sys.executable, str(ROOT / "scbeacon.py")]
        steps = [
            base + ["train", "--config", c, "--output", out + "/01_train", "--execute"],
            base + ["pairs-deg", "--config", c, "--embedding", out + "/01_train/03_contrastive/contrastive_gene_gene_sets/iter_33_contrastive_final.npy", "--output", out + "/02_pairs_deg", "--execute"],
            base + ["qc", "--config", c, "--pair-run", out + "/02_pairs_deg", "--output", out + "/03_qc", "--execute"],
            base + ["markers", "--config", c, "--pair-run", out + "/02_pairs_deg", "--clean-pairs", out + "/03_qc/clean_pairs.tsv", "--output", out + "/04_markers", "--execute"],
            base + ["enrich", "--config", c, "--pair-run", out + "/02_pairs_deg", "--clean-pairs", out + "/03_qc/clean_pairs.tsv", "--output", out + "/05_enrichment", "--execute", "--allow-network"],
            base + ["plot-run", "--config", c, "--pair-run", out + "/02_pairs_deg", "--qc-run", out + "/03_qc", "--marker-run", out + "/04_markers", "--enrichment-run", out + "/05_enrichment", "--output", out + "/06_figures", "--execute"],
        ]
        print(json.dumps({"mode": "plan_only", "commands": steps, "note": "Live enrichment changes with ToppGene. Frozen notebook tables reproduce archived numerical results."}, indent=2))
        return
    if args.command == "render":
        compute_only()
        from results import export_panels
        export_panels(args.dataset, fresh_output(args.output))
        return
    cfg = settings(args.config)
    if getattr(args, "embedding", None):
        cfg["embedding"] = str(args.embedding.resolve())
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "stage": args.command, "settings": cfg, "output": str(args.output.resolve())}, indent=2))
        return
    compute_only()
    if args.command == "train":
        child = ROOT / "01_tool/02_training/train_scbeacon.py"
        subprocess.run([sys.executable, str(child), "--config", str(args.config.resolve()), "--output", str(args.output.resolve()), "--bundle-root", str(ROOT), "--execute"], check=True)
        return
    if args.command == "enrich" and not args.allow_network:
        raise ValueError("Live ToppGene calls require --allow-network; use notebooks for offline archived results")
    if args.command == "plot-run":
        from new_results import plot_run_results
        plot_run_results(cfg, args, fresh_output(args.output))
    elif args.command == "pairs-deg":
        validate_inputs(cfg)
        pairs_and_deg(cfg, fresh_output(args.output))
    elif args.command == "prepare":
        from prepare import prepare_inputs
        prepare_inputs(cfg, fresh_output(args.output))
    else:
        from downstream import run_stage
        run_stage(args.command, cfg, args.pair_run.resolve(), fresh_output(args.output), getattr(args, "clean_pairs", None))


if __name__ == "__main__":
    main()
