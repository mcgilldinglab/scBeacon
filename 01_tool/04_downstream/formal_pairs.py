#!/usr/bin/env python3
"""Build scBeacon resolution=1 KL pairs and Wilcoxon DEG tables.

This is the resolution=1 companion of the seed23 KL-pair downstream:
the model/embedding stays fixed, but the condition-specific clusters
used for KL pairing are recomputed from the final embedding at Leiden
resolution 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-resolution", type=float, required=True)
    parser.add_argument("--cluster-resolution", type=float, default=1.0)
    parser.add_argument("--embedding-npy", required=True)
    parser.add_argument("--combined-lognorm-h5ad", required=True)
    parser.add_argument("--condition1-h5ad", required=True)
    parser.add_argument("--condition2-h5ad", required=True)
    parser.add_argument("--condition1-name", required=True)
    parser.add_argument("--condition2-name", required=True)
    parser.add_argument("--label-key-1", required=True)
    parser.add_argument("--label-key-2", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prefixed_cell_ids(adata: ad.AnnData, condition: str) -> list[str]:
    out = []
    for cell_id in adata.obs_names.astype(str):
        if cell_id.startswith(f"{condition}::"):
            out.append(cell_id)
        else:
            out.append(f"{condition}::{cell_id}")
    return out


def cluster_embedding(embedding: np.ndarray, resolution: float, random_seed: int) -> np.ndarray:
    adata = ad.AnnData(X=np.zeros((embedding.shape[0], 1), dtype=np.float32))
    adata.obsm["X_eval"] = np.asarray(embedding, dtype=np.float32)
    sc.pp.neighbors(adata, use_rep="X_eval", random_state=random_seed)
    sc.tl.leiden(adata, resolution=float(resolution), random_state=random_seed)
    return adata.obs["leiden"].astype(str).to_numpy()


def centroids_from_labels(embedding: np.ndarray, labels: np.ndarray) -> tuple[list[str], np.ndarray, dict[str, int]]:
    df = pd.DataFrame(np.asarray(embedding, dtype=np.float32))
    df["cluster"] = labels.astype(str)
    grouped = df.groupby("cluster", sort=True)
    centroids = grouped.mean().to_numpy(dtype=np.float32)
    cluster_names = grouped.mean().index.astype(str).tolist()
    sizes = df["cluster"].value_counts().sort_index()
    size_map = {str(k): int(v) for k, v in sizes.items()}
    return cluster_names, centroids, size_map


def get_kl_div_pair(centroids_1: np.ndarray, centroids_2: np.ndarray) -> dict[int, int]:
    pair_map: dict[int, int] = {}
    c1 = torch.tensor(centroids_1, dtype=torch.float32)
    c2 = torch.tensor(centroids_2, dtype=torch.float32)
    for i in range(c1.shape[0]):
        dists = []
        for j in range(c2.shape[0]):
            z_dist = F.kl_div(
                c1[i].softmax(dim=-1).log(),
                c2[j].softmax(dim=-1),
                reduction="sum",
            ).item()
            dists.append(z_dist)
        pair_map[i] = int(np.argmin(dists))
    return pair_map


def build_pair_maps(
    z1: np.ndarray,
    labels_1: np.ndarray,
    z2: np.ndarray,
    labels_2: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cluster_names_1, centroids_1, size_map_1 = centroids_from_labels(z1, labels_1)
    cluster_names_2, centroids_2, size_map_2 = centroids_from_labels(z2, labels_2)
    scaled_centroids_1 = preprocessing.scale(centroids_1)
    scaled_centroids_2 = preprocessing.scale(centroids_2)

    c_to_i_idx = get_kl_div_pair(scaled_centroids_1, scaled_centroids_2)
    i_to_c_idx = get_kl_div_pair(scaled_centroids_2, scaled_centroids_1)

    c_to_i_rows = []
    for source_idx, target_idx in c_to_i_idx.items():
        c_to_i_rows.append(
            {
                "control_cluster": int(cluster_names_1[source_idx]),
                "ipf_cluster": int(cluster_names_2[target_idx]),
                "control_cluster_size": int(size_map_1[cluster_names_1[source_idx]]),
                "ipf_cluster_size": int(size_map_2[cluster_names_2[target_idx]]),
                "is_reciprocal": bool(i_to_c_idx.get(target_idx) == source_idx),
            }
        )
    i_to_c_rows = []
    for source_idx, target_idx in i_to_c_idx.items():
        i_to_c_rows.append(
            {
                "ipf_cluster": int(cluster_names_2[source_idx]),
                "control_cluster": int(cluster_names_1[target_idx]),
                "ipf_cluster_size": int(size_map_2[cluster_names_2[source_idx]]),
                "control_cluster_size": int(size_map_1[cluster_names_1[target_idx]]),
                "is_reciprocal": bool(c_to_i_idx.get(target_idx) == source_idx),
            }
        )

    reciprocal_rows = []
    for row in c_to_i_rows:
        if row["is_reciprocal"]:
            reciprocal_rows.append(
                {
                    "pair_id": f"A_pair_{len(reciprocal_rows):02d}",
                    "pair_version": "A_reciprocal_cluster_resolution1",
                    "pair_type": "reciprocal",
                    "control_cluster": row["control_cluster"],
                    "ipf_cluster": row["ipf_cluster"],
                    "control_cluster_size": row["control_cluster_size"],
                    "ipf_cluster_size": row["ipf_cluster_size"],
                }
            )

    directed_rows = []
    for row in i_to_c_rows:
        directed_rows.append(
            {
                "pair_id": f"B_pair_{len(directed_rows):02d}",
                "pair_version": "B_ipf_to_control_directed_cluster_resolution1",
                "pair_type": "ipf_to_control_directed",
                "control_cluster": row["control_cluster"],
                "ipf_cluster": row["ipf_cluster"],
                "control_cluster_size": row["control_cluster_size"],
                "ipf_cluster_size": row["ipf_cluster_size"],
                "is_reciprocal": row["is_reciprocal"],
            }
        )

    reciprocal_cells_1 = sum(row["control_cluster_size"] for row in reciprocal_rows)
    reciprocal_cells_2 = sum(row["ipf_cluster_size"] for row in reciprocal_rows)
    stats = {
        "n_control_clusters": len(cluster_names_1),
        "n_ipf_clusters": len(cluster_names_2),
        "control_to_ipf_pair_count": len(c_to_i_rows),
        "ipf_to_control_pair_count": len(i_to_c_rows),
        "mutual_pair_count": len(reciprocal_rows),
        "reciprocity_control": float(np.mean([row["is_reciprocal"] for row in c_to_i_rows])) if c_to_i_rows else np.nan,
        "reciprocity_ipf": float(np.mean([row["is_reciprocal"] for row in i_to_c_rows])) if i_to_c_rows else np.nan,
        "reciprocal_cell_fraction": (reciprocal_cells_1 + reciprocal_cells_2) / float(z1.shape[0] + z2.shape[0]),
    }
    return pd.DataFrame(c_to_i_rows), pd.DataFrame(i_to_c_rows), pd.DataFrame(reciprocal_rows), pd.DataFrame(directed_rows), stats


def write_assignments(
    output_dir: Path,
    control_cells: list[str],
    ipf_cells: list[str],
    control_labels: np.ndarray,
    ipf_labels: np.ndarray,
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    label_key_1: str,
    label_key_2: str,
    condition1: str,
    condition2: str,
) -> pd.DataFrame:
    rows = []
    label_values_1 = adata1.obs[label_key_1].astype(str).tolist()
    label_values_2 = adata2.obs[label_key_2].astype(str).tolist()
    for cell_id, true_label, cluster in zip(control_cells, label_values_1, control_labels, strict=True):
        rows.append(
            {
                "cell_id": cell_id,
                "condition": condition1,
                "true_label": true_label,
                "predicted_cluster": int(cluster),
                "mode": "scbeacon_seed23_iteration33_cluster_resolution1",
                "method": "scbeacon_kl_pair",
            }
        )
    for cell_id, true_label, cluster in zip(ipf_cells, label_values_2, ipf_labels, strict=True):
        rows.append(
            {
                "cell_id": cell_id,
                "condition": condition2,
                "true_label": true_label,
                "predicted_cluster": int(cluster),
                "mode": "scbeacon_seed23_iteration33_cluster_resolution1",
                "method": "scbeacon_kl_pair",
            }
        )
    assignments = pd.DataFrame(rows)
    assignments.to_csv(output_dir / "combined_condition_assignments.tsv.gz", sep="\t", index=False, compression="gzip")
    return assignments


def pair_cell_ids(assignments: pd.DataFrame, pair_row: pd.Series, condition1: str, condition2: str) -> list[str]:
    control_cluster = int(pair_row["control_cluster"])
    ipf_cluster = int(pair_row["ipf_cluster"])
    mask = (
        (assignments["condition"].astype(str) == condition1)
        & (assignments["predicted_cluster"].astype(int) == control_cluster)
    ) | (
        (assignments["condition"].astype(str) == condition2)
        & (assignments["predicted_cluster"].astype(int) == ipf_cluster)
    )
    return assignments.loc[mask, "cell_id"].astype(str).tolist()


def rank_wilcoxon_for_pairs(
    combined: ad.AnnData,
    assignments: pd.DataFrame,
    pair_table: pd.DataFrame,
    output_dir: Path,
    condition1: str,
    condition2: str,
) -> pd.DataFrame:
    ranked_dir = ensure_dir(output_dir / "01_full_ranked_tables")
    rows = []
    for _, pair in pair_table.iterrows():
        pair_id = str(pair["pair_id"])
        cell_ids = [cell_id for cell_id in pair_cell_ids(assignments, pair, condition1, condition2) if cell_id in combined.obs_names]
        adata_pair = combined[cell_ids].copy()
        if "condition" not in adata_pair.obs:
            adata_pair.obs["condition"] = [str(idx).split("::", 1)[0] for idx in adata_pair.obs_names]
        n_control = int((adata_pair.obs["condition"].astype(str) == condition1).sum())
        n_ipf = int((adata_pair.obs["condition"].astype(str) == condition2).sum())
        out_path = ranked_dir / f"{pair_id}_ipf_vs_control.tsv"
        if n_control == 0 or n_ipf == 0:
            pd.DataFrame().to_csv(out_path, sep="\t", index=False)
            status = "skipped_missing_condition"
        else:
            sc.tl.rank_genes_groups(
                adata_pair,
                groupby="condition",
                groups=[condition2],
                reference=condition1,
                method="wilcoxon",
                use_raw=False,
                pts=True,
            )
            deg = sc.get.rank_genes_groups_df(adata_pair, group=condition2)
            deg.to_csv(out_path, sep="\t", index=False)
            status = "completed"
        rows.append(
            {
                "pair_id": pair_id,
                "pair_type": pair["pair_type"],
                "control_cluster": int(pair["control_cluster"]),
                "ipf_cluster": int(pair["ipf_cluster"]),
                "n_cells": int(adata_pair.n_obs),
                "n_control_cells": n_control,
                "n_ipf_cells": n_ipf,
                "n_genes_ranked": int(combined.n_vars) if status == "completed" else 0,
                "status": status,
                "ranked_table": str(out_path),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "deg_unit_summary.tsv", sep="\t", index=False)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    pair_dir = ensure_dir(output_dir / "00_pair_definitions")
    wilcox_a = ensure_dir(output_dir / "01_wilcoxon_full_deg/01_version_a_reciprocal_pairs")
    wilcox_b = ensure_dir(output_dir / "01_wilcoxon_full_deg/02_version_b_ipf_to_control_directed_pairs")

    adata1 = ad.read_h5ad(args.condition1_h5ad)
    adata2 = ad.read_h5ad(args.condition2_h5ad)
    combined = ad.read_h5ad(args.combined_lognorm_h5ad)
    if "condition" not in combined.obs:
        combined.obs["condition"] = [str(idx).split("::", 1)[0] for idx in combined.obs_names]

    embedding = np.load(args.embedding_npy)
    n1 = adata1.n_obs
    n2 = adata2.n_obs
    if embedding.shape[0] != n1 + n2:
        raise ValueError(f"Embedding rows {embedding.shape[0]} != n1+n2 {n1+n2}")
    z1 = embedding[:n1]
    z2 = embedding[n1:]

    labels1 = cluster_embedding(z1, args.cluster_resolution, args.random_seed)
    labels2 = cluster_embedding(z2, args.cluster_resolution, args.random_seed)
    control_cells = prefixed_cell_ids(adata1, args.condition1_name)
    ipf_cells = prefixed_cell_ids(adata2, args.condition2_name)
    assignments = write_assignments(
        pair_dir,
        control_cells,
        ipf_cells,
        labels1,
        labels2,
        adata1,
        adata2,
        args.label_key_1,
        args.label_key_2,
        args.condition1_name,
        args.condition2_name,
    )

    control_to_ipf, ipf_to_control, reciprocal_pairs, directed_pairs, stats = build_pair_maps(z1, labels1, z2, labels2)
    control_to_ipf.to_csv(pair_dir / "control_to_ipf_kl_pair_map.tsv", sep="\t", index=False)
    ipf_to_control.to_csv(pair_dir / "ipf_to_control_kl_pair_map.tsv", sep="\t", index=False)
    reciprocal_pairs.to_csv(pair_dir / "version_a_reciprocal_pairs.tsv", sep="\t", index=False)
    directed_pairs.to_csv(pair_dir / "version_b_ipf_to_control_directed_pairs.tsv", sep="\t", index=False)
    pd.DataFrame([stats]).to_csv(pair_dir / "resolution1_pairing_summary.tsv", sep="\t", index=False)

    rank_wilcoxon_for_pairs(combined, assignments, reciprocal_pairs, wilcox_a, args.condition1_name, args.condition2_name)
    rank_wilcoxon_for_pairs(combined, assignments, directed_pairs, wilcox_b, args.condition1_name, args.condition2_name)

    manifest = {
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "iteration": args.iteration,
        "train_resolution": args.train_resolution,
        "cluster_resolution_for_pairing": args.cluster_resolution,
        "embedding_npy": str(Path(args.embedding_npy).resolve()),
        "combined_lognorm_h5ad": str(Path(args.combined_lognorm_h5ad).resolve()),
        "condition1_h5ad": str(Path(args.condition1_h5ad).resolve()),
        "condition2_h5ad": str(Path(args.condition2_h5ad).resolve()),
        "condition1_name": args.condition1_name,
        "condition2_name": args.condition2_name,
        "label_key_1": args.label_key_1,
        "label_key_2": args.label_key_2,
        "n_condition1_cells": n1,
        "n_condition2_cells": n2,
        "n_condition1_resolution1_clusters": int(pd.Series(labels1).nunique()),
        "n_condition2_resolution1_clusters": int(pd.Series(labels2).nunique()),
        "pairing_stats": stats,
        "kl_pairing_notes": {
            "centroids": "mean final embedding per condition-specific Leiden cluster",
            "scaling": "each condition centroid matrix standardized with sklearn.preprocessing.scale before KL pairing",
            "directed_pair": "for each source centroid i, choose target centroid j minimizing KL(softmax(source_i) || softmax(target_j))",
            "reciprocal_pair": "control->IPF and IPF->control choices agree",
        },
    }
    with open(output_dir / "deg_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Wrote resolution=1 pair definitions and Wilcoxon DEG to {output_dir}")


if __name__ == "__main__":
    main()
