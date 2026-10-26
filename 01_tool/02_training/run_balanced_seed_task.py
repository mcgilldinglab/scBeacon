#!/usr/bin/env python3
"""Run one dataset/seed scBeacon task with pretrain-selected train resolution."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score, silhouette_score


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def load_manifest_row(manifest: Path, task_id: int) -> dict[str, str]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if int(row["task_id"]) == int(task_id):
            return row
    raise ValueError(f"No task_id={task_id} in {manifest}")


def apply_base_env(row: dict[str, str], output_root: Path) -> None:
    env = parse_env_file(Path(row["workspace_env"]))
    for key, value in env.items():
        os.environ[key] = value
    seed = int(row["seed"])
    os.environ["SCBEACON_WORKSPACE_ROOT"] = str(Path(row["workspace_env"]).parent)
    os.environ["SCBEACON_OUTPUT_ROOT"] = str(output_root)
    os.environ["SCBEACON_DATASET_ID"] = f"{row['dataset']}_balanced_seed_{seed:02d}_iteration33"
    os.environ["SCBEACON_RANDOM_SEED"] = str(seed)
    os.environ["SCBEACON_BATCH_SIZE"] = "512"
    os.environ["SCBEACON_NUM_ITERATION"] = "33"
    os.environ["SCBEACON_USE_CUDA"] = "true"
    os.environ["SCBEACON_PRETRAIN_ONLY"] = "false"
    if row["dataset"] == "covid":
        os.environ["SCBEACON_LABEL_KEY"] = "Cell_type_annotation_level1"


def set_seed(seed: int, use_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_core(core_root: Path):
    for candidate in [core_root, core_root / "src", core_root / "unifan"]:
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    sys.modules.pop("config_scbeacon_pair", None)
    from config_scbeacon_pair import Config
    from unifan.autoencoderV2 import autoencoder
    from unifan.trainer import Trainer

    return Config, autoencoder, Trainer


def import_train(core_root: Path):
    for candidate in [core_root, core_root / "src", core_root / "unifan"]:
        candidate_str = str(candidate.resolve())
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    sys.modules.pop("config_scbeacon_pair", None)
    sys.modules.pop("train_scbeacon_pair", None)
    from train_scbeacon_pair import Train

    return Train


def to_numpy(value):
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        pass
    try:
        return value.cpu().numpy()
    except AttributeError:
        pass
    return np.asarray(value)


def run_pretrain(seed: int, core_root: Path, output_root: Path):
    Config, autoencoder, Trainer = import_core(core_root)
    set_seed(seed)
    cfg = Config()
    model_autoencoder = autoencoder(
        input_dim=cfg.G,
        z_dim=cfg.z_dim,
        encoder_dim=cfg.z_encoder_dim,
        emission_dim=cfg.z_decoder_dim,
        num_layers_encoder=cfg.z_encoder_layers,
        num_layers_decoder=cfg.z_decoder_layers,
        reconstruction_network="gaussian",
        decoding_network="gaussian",
        use_cuda=cfg.use_cuda,
    )
    runtime_root = output_root / "_runtime_state"
    runtime_root.mkdir(parents=True, exist_ok=True)
    resume_state_path = runtime_root / "pretrain_z_ZINB_resume_state_1.pickle"
    trainer = Trainer(
        dataset=cfg.expression_integrated,
        model=model_autoencoder,
        learning_rate=cfg.alpha,
        model_name="pretrain_z_ZINB",
        batch_size=cfg.batch_size,
        num_epochs=cfg.pretrain_epoch,
        save_infer=True,
        output_folder=cfg.pretrain_z_folder,
        num_workers=cfg.num_workers,
        use_cuda=cfg.use_cuda,
        iteration=1,
        resume_state_path=str(resume_state_path),
        save_epoch_embeddings=False,
    )
    completed = trainer.resume_state is not None and bool(trainer.resume_state.get("completed", False))
    if completed:
        print("[pretrain] completed resume state found; rebuild embedding", flush=True)
    else:
        print(f"[pretrain] start/resume from epoch {getattr(trainer, 'start_epoch', 0)}", flush=True)
        trainer.train()
    z_init = to_numpy(trainer.infer_z_ZINB())
    Path(cfg.pretrain_z_folder).mkdir(parents=True, exist_ok=True)
    np.save(cfg.input_z_ZINB_path, z_init)
    np.save(Path(cfg.pretrain_z_folder) / "pretrain_final_for_resolution_selection.npy", z_init)
    return cfg, z_init


def build_neighbor_adata(embedding: np.ndarray, random_seed: int):
    adata = sc.AnnData(X=np.zeros((embedding.shape[0], 1), dtype=np.float32))
    adata.obsm["X_eval"] = embedding.astype(np.float32, copy=False)
    sc.pp.neighbors(adata, use_rep="X_eval", random_state=random_seed)
    return adata


def cluster_cached(adata_base, labels: np.ndarray, resolution: float, random_seed: int) -> dict[str, object]:
    tmp = adata_base.copy()
    sc.tl.leiden(tmp, resolution=float(resolution), random_state=random_seed)
    pred = tmp.obs["leiden"].astype(str).to_numpy()
    return {
        "pred": pred,
        "n_clusters": int(pd.Series(pred).nunique()),
        "ari": float(adjusted_rand_score(labels, pred)),
    }


def label_codes(labels: np.ndarray) -> np.ndarray:
    return pd.Categorical(labels.astype(str)).codes.astype(int)


def safe_silhouette(embedding: np.ndarray, labels: np.ndarray, random_seed: int, max_cells: int) -> float:
    labels = np.asarray(labels).astype(str)
    if pd.Series(labels).nunique() < 2:
        return float("nan")
    if embedding.shape[0] <= pd.Series(labels).nunique():
        return float("nan")
    if max_cells > 0 and embedding.shape[0] > max_cells:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(embedding.shape[0], size=max_cells, replace=False))
        embedding = embedding[idx]
        labels = labels[idx]
        if pd.Series(labels).nunique() < 2:
            return float("nan")
    return float(silhouette_score(embedding, label_codes(labels), metric="euclidean"))


def centroids_from_labels(embedding: np.ndarray, labels: np.ndarray):
    df = pd.DataFrame(embedding)
    df["cluster"] = labels.astype(str)
    grouped = df.groupby("cluster", sort=True)
    centroids = grouped.mean().to_numpy(dtype=np.float32)
    cluster_names = grouped.mean().index.astype(str).tolist()
    sizes = df["cluster"].value_counts().sort_index()
    size_map = {str(k): int(v) for k, v in sizes.items()}
    return cluster_names, centroids, size_map


def cluster_size_stats(size_map: dict[str, int]) -> dict[str, int]:
    sizes = np.asarray(list(size_map.values()), dtype=int)
    if sizes.size == 0:
        return {"cluster_count": 0, "min_cluster_cells": 0, "max_cluster_cells": 0}
    return {
        "cluster_count": int(sizes.size),
        "min_cluster_cells": int(sizes.min()),
        "max_cluster_cells": int(sizes.max()),
    }


def get_kl_div_pair(centroids_1: np.ndarray, centroids_2: np.ndarray) -> dict[int, int]:
    pair_map: dict[int, int] = {}
    c1 = torch.tensor(centroids_1, dtype=torch.float32)
    c2 = torch.tensor(centroids_2, dtype=torch.float32)
    for i in range(c1.shape[0]):
        dists = []
        for j in range(c2.shape[0]):
            dists.append(
                F.kl_div(
                    c1[i].softmax(dim=-1).log(),
                    c2[j].softmax(dim=-1),
                    reduction="sum",
                ).item()
            )
        pair_map[i] = int(np.argmin(dists))
    return pair_map


def pair_reciprocity(z1: np.ndarray, labels_1: np.ndarray, z2: np.ndarray, labels_2: np.ndarray) -> dict[str, object]:
    names_1, centroids_1, size_map_1 = centroids_from_labels(z1, labels_1)
    names_2, centroids_2, size_map_2 = centroids_from_labels(z2, labels_2)
    stats_1 = cluster_size_stats(size_map_1)
    stats_2 = cluster_size_stats(size_map_2)
    pair_1 = get_kl_div_pair(preprocessing.scale(centroids_1), preprocessing.scale(centroids_2))
    pair_2 = get_kl_div_pair(preprocessing.scale(centroids_2), preprocessing.scale(centroids_1))
    mutual_pairs = []
    reciprocal_cells_1 = 0
    reciprocal_cells_2 = 0
    for i, j in pair_1.items():
        if pair_2.get(j) == i:
            mutual_pairs.append((i, j))
            reciprocal_cells_1 += size_map_1[names_1[i]]
            reciprocal_cells_2 += size_map_2[names_2[j]]
    reciprocity_control = float(np.mean([pair_2.get(j) == i for i, j in pair_1.items()])) if pair_1 else np.nan
    reciprocity_ipf = float(np.mean([pair_1.get(i) == j for j, i in pair_2.items()])) if pair_2 else np.nan
    many_to_one_control = 1.0 - (len(set(pair_1.values())) / max(1, len(pair_1)))
    many_to_one_ipf = 1.0 - (len(set(pair_2.values())) / max(1, len(pair_2)))
    return {
        "control_n_clusters": len(names_1),
        "ipf_n_clusters": len(names_2),
        "cluster_count_ratio_ipf_over_control": float(stats_2["cluster_count"] / max(1, stats_1["cluster_count"])),
        "control_min_cluster_cells": stats_1["min_cluster_cells"],
        "ipf_min_cluster_cells": stats_2["min_cluster_cells"],
        "overall_min_cluster_cells": int(min(stats_1["min_cluster_cells"], stats_2["min_cluster_cells"])),
        "reciprocal_pair_count": len(mutual_pairs),
        "reciprocal_pair_ratio": len(mutual_pairs) / max(1, min(len(pair_1), len(pair_2))),
        "reciprocity_control": reciprocity_control,
        "reciprocity_ipf": reciprocity_ipf,
        "reciprocity_mean": float(np.nanmean([reciprocity_control, reciprocity_ipf])),
        "many_to_one_control": many_to_one_control,
        "many_to_one_ipf": many_to_one_ipf,
        "many_to_one_mean": 0.5 * (many_to_one_control + many_to_one_ipf),
        "reciprocal_cell_fraction": (reciprocal_cells_1 + reciprocal_cells_2) / float(z1.shape[0] + z2.shape[0]),
    }


def rank_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(
        method="average",
        ascending=not higher_is_better,
        na_option="bottom",
    )


def add_selection_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    numeric_cols = [
        "resolution",
        "reciprocal_pair_ratio",
        "reciprocity_mean",
        "many_to_one_mean",
        "overall_min_cluster_cells",
        "cluster_count_ratio_ipf_over_control",
        "reciprocal_cell_fraction",
        "silhouette_mean",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["cluster_balance_abs_log_ratio"] = np.abs(np.log(work["cluster_count_ratio_ipf_over_control"]))
    work["passes_quality_boundary"] = (
        (work["overall_min_cluster_cells"] >= 10)
        & (work["cluster_count_ratio_ipf_over_control"] >= 0.5)
        & (work["cluster_count_ratio_ipf_over_control"] <= 2.0)
        & (work["reciprocal_pair_ratio"] >= 0.5)
        & (work["many_to_one_mean"] <= 0.5)
    )
    quality = work[work["passes_quality_boundary"]].copy()
    candidate_status = "quality_boundary"
    if quality.empty:
        quality = work.copy()
        candidate_status = "fallback_all_candidates"
    quality["rank_reciprocal_pair_ratio"] = rank_series(quality["reciprocal_pair_ratio"], True)
    quality["rank_reciprocity_mean"] = rank_series(quality["reciprocity_mean"], True)
    quality["rank_many_to_one_mean"] = rank_series(quality["many_to_one_mean"], False)
    quality["rank_min_cluster_cells"] = rank_series(quality["overall_min_cluster_cells"], True)
    quality["rank_cluster_balance"] = rank_series(quality["cluster_balance_abs_log_ratio"], False)
    quality["rank_reciprocal_cell_fraction"] = rank_series(quality["reciprocal_cell_fraction"], True)
    rank_cols = [
        "rank_reciprocal_pair_ratio",
        "rank_reciprocity_mean",
        "rank_many_to_one_mean",
        "rank_min_cluster_cells",
        "rank_cluster_balance",
        "rank_reciprocal_cell_fraction",
    ]
    quality["balanced_no_silhouette_rank_score"] = quality[rank_cols].mean(axis=1)
    work = work.merge(
        quality[["resolution", "balanced_no_silhouette_rank_score"]],
        on="resolution",
        how="left",
    )
    balanced = quality.sort_values(
        ["balanced_no_silhouette_rank_score", "reciprocal_pair_ratio", "many_to_one_mean", "resolution"],
        ascending=[True, False, True, True],
    ).iloc[0].copy()
    balanced["selector_metric"] = "balanced_no_silhouette_rank"
    balanced["candidate_status"] = candidate_status
    silhouette = work.sort_values(["silhouette_mean", "resolution"], ascending=[False, True]).iloc[0].copy()
    silhouette["selector_metric"] = "silhouette_mean_max"
    silhouette["candidate_status"] = "all_candidates"
    return work, pd.DataFrame([balanced, silhouette])


def compute_pretrain_resolution_metrics(
    cfg,
    embedding: np.ndarray,
    seed: int,
    outdir: Path,
    max_silhouette_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    z1 = embedding[: cfg.N_project1]
    z2 = embedding[cfg.N_project1 :]
    labels1 = cfg.clusters_true1.astype(str)
    labels2 = cfg.clusters_true2.astype(str)
    joint_labels = np.concatenate(
        [
            np.char.add(f"{cfg.condition_name_1}:", labels1),
            np.char.add(f"{cfg.condition_name_2}:", labels2),
        ]
    )
    control_adata = build_neighbor_adata(z1, seed)
    ipf_adata = build_neighbor_adata(z2, seed)
    joint_adata = build_neighbor_adata(embedding, seed)
    rows = []
    metrics_path = outdir / "pretrain_resolution_metrics.tsv"
    for resolution in np.round(np.arange(0.1, 1.0 + 0.05, 0.1), 1):
        control_stats = cluster_cached(control_adata, labels1, float(resolution), seed)
        ipf_stats = cluster_cached(ipf_adata, labels2, float(resolution), seed)
        joint_stats = cluster_cached(joint_adata, joint_labels, float(resolution), seed)
        pair_stats = pair_reciprocity(z1, control_stats["pred"], z2, ipf_stats["pred"])
        silhouette_control = safe_silhouette(z1, control_stats["pred"], seed, max_silhouette_cells)
        silhouette_ipf = safe_silhouette(z2, ipf_stats["pred"], seed, max_silhouette_cells)
        rows.append(
            {
                "seed": seed,
                "resolution": float(resolution),
                "control_ari": float(control_stats["ari"]),
                "ipf_ari": float(ipf_stats["ari"]),
                "joint_ari": float(joint_stats["ari"]),
                "joint_n_clusters": int(joint_stats["n_clusters"]),
                "silhouette_control": float(silhouette_control),
                "silhouette_ipf": float(silhouette_ipf),
                "silhouette_mean": float(np.nanmean([silhouette_control, silhouette_ipf])),
                **pair_stats,
            }
        )
        pd.DataFrame(rows).to_csv(metrics_path, sep="\t", index=False)
    scored, selection = add_selection_scores(pd.DataFrame(rows).sort_values("resolution"))
    scored.to_csv(metrics_path, sep="\t", index=False)
    selection.to_csv(outdir / "pretrain_resolution_selection_summary.tsv", sep="\t", index=False)
    return scored, selection


def load_or_compute_selection(cfg, embedding: np.ndarray, seed: int, outdir: Path) -> tuple[float, pd.DataFrame]:
    selection_path = outdir / "pretrain_resolution_selection_summary.tsv"
    metrics_path = outdir / "pretrain_resolution_metrics.tsv"
    if selection_path.exists() and metrics_path.exists():
        selection = pd.read_csv(selection_path, sep="\t")
    else:
        _, selection = compute_pretrain_resolution_metrics(cfg, embedding, seed, outdir, max_silhouette_cells=10000)
    balanced = selection[selection["selector_metric"].astype(str) == "balanced_no_silhouette_rank"].iloc[0]
    selected_resolution = float(balanced["resolution"])
    (outdir / "selected_train_resolution.txt").write_text(f"{selected_resolution:.1f}\n", encoding="utf-8")
    return selected_resolution, selection


def run_iteration_training(core_root: Path, selected_resolution: float) -> None:
    os.environ["SCBEACON_LEIDEN_RESOLUTION"] = f"{selected_resolution:.1f}"
    os.environ["SCBEACON_NUM_ITERATION"] = "33"
    os.environ["SCBEACON_PRETRAIN_ONLY"] = "false"
    Train = import_train(core_root)
    old_cwd = Path.cwd()
    try:
        os.chdir(core_root / "src")
        train = Train()
        train.run()
    finally:
        os.chdir(old_cwd)


def final_iteration33_ari(core_root: Path, seed: int, selected_resolution: float, summary_dir: Path) -> pd.DataFrame:
    Config, _, _ = import_core(core_root)
    cfg = Config()
    embedding_path = Path(cfg.model_contrastive_folder) / "constractive_optimal_33.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(embedding_path)
    embedding = np.load(embedding_path)
    alias_path = Path(cfg.model_contrastive_folder) / "iter_33_contrastive_final.npy"
    if not alias_path.exists():
        np.save(alias_path, embedding)
    z1 = embedding[: cfg.N_project1]
    z2 = embedding[cfg.N_project1 :]
    labels1 = cfg.clusters_true1.astype(str)
    labels2 = cfg.clusters_true2.astype(str)
    joint_labels = np.concatenate(
        [
            np.char.add(f"{cfg.condition_name_1}:", labels1),
            np.char.add(f"{cfg.condition_name_2}:", labels2),
        ]
    )
    control_stats = cluster_cached(build_neighbor_adata(z1, seed), labels1, 1.0, seed)
    ipf_stats = cluster_cached(build_neighbor_adata(z2, seed), labels2, 1.0, seed)
    joint_stats = cluster_cached(build_neighbor_adata(embedding, seed), joint_labels, 1.0, seed)
    out = pd.DataFrame(
        [
            {
                "seed": seed,
                "selected_train_resolution": selected_resolution,
                "iteration": 33,
                "cluster_resolution": 1.0,
                "condition1_ari": control_stats["ari"],
                "condition2_ari": ipf_stats["ari"],
                "joint_ari": joint_stats["ari"],
                "condition1_n_clusters": control_stats["n_clusters"],
                "condition2_n_clusters": ipf_stats["n_clusters"],
                "joint_n_clusters": joint_stats["n_clusters"],
                "embedding_path": str(embedding_path),
                "embedding_alias_path": str(alias_path),
                "label_key_1": cfg.label_key_1,
                "label_key_2": cfg.label_key_2,
            }
        ]
    )
    out.to_csv(summary_dir / "iteration33_cluster_resolution1_ari.tsv", sep="\t", index=False)
    return out


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    row = load_manifest_row(Path(args.manifest), args.task_id)
    seed = int(row["seed"])
    core_root = Path(row["core_root"]).resolve()
    dataset_run_root = Path(row["dataset_run_root"]).resolve()
    output_base = dataset_run_root / "01_model" / "03_output" / "01_seed1_10_balanced_array"
    seed_output = output_base / "02_output" / f"seed_{seed:02d}" / "full_run"
    seed_summary = output_base / "03_summary" / f"seed_{seed:02d}"
    seed_summary.mkdir(parents=True, exist_ok=True)
    completion_marker = seed_summary / "task_completed.json"
    if completion_marker.exists():
        print(f"[skip] completed marker exists: {completion_marker}", flush=True)
        return

    start = time.time()
    apply_base_env(row, seed_output)
    print(f"[task] dataset={row['dataset']} seed={seed} output={seed_output}", flush=True)
    cfg, embedding = run_pretrain(seed, core_root, seed_output)
    selected_resolution, selection = load_or_compute_selection(cfg, embedding, seed, seed_summary)
    print(f"[selection] balanced_no_silhouette_rank selected train resolution={selected_resolution:.1f}", flush=True)
    write_json(
        seed_summary / "selected_train_resolution.json",
        {
            "dataset": row["dataset"],
            "seed": seed,
            "selected_train_resolution": selected_resolution,
            "selector": "balanced_no_silhouette_rank",
            "selection_rows": selection.to_dict(orient="records"),
        },
    )
    run_iteration_training(core_root, selected_resolution)
    final_ari = final_iteration33_ari(core_root, seed, selected_resolution, seed_summary)
    elapsed = time.time() - start
    payload = {
        "dataset": row["dataset"],
        "seed": seed,
        "selected_train_resolution": selected_resolution,
        "iteration": 33,
        "elapsed_seconds": round(elapsed, 3),
        "final_ari": final_ari.to_dict(orient="records")[0],
        "output_root": str(seed_output),
        "summary_dir": str(seed_summary),
    }
    write_json(completion_marker, payload)
    pd.DataFrame([payload | {"final_joint_ari": final_ari["joint_ari"].iloc[0]}]).to_csv(
        seed_summary / "task_completed.tsv",
        sep="\t",
        index=False,
    )
    print(
        f"[done] dataset={row['dataset']} seed={seed} selected_res={selected_resolution:.1f} "
        f"joint_ari={final_ari['joint_ari'].iloc[0]:.6f} elapsed_min={elapsed / 60.0:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
