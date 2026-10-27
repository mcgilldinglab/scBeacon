"""Portable I/O around the archived pair, edge-test and ToppGene methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow import ROOT, load_module


def directed_edges(source, target, source_labels, target_labels):
    from sklearn.neighbors import NearestNeighbors

    if min(len(source), len(target)) < 30:
        raise ValueError("The fixed k=30 recipe requires at least 30 cells in each condition")
    # Count directed cross-condition cell-neighbor edges, not UMAP distances.
    neighbors = NearestNeighbors(n_neighbors=30, metric="euclidean", n_jobs=1).fit(target).kneighbors(source, return_distance=False)
    rows = sorted(set(source_labels), key=int)
    cols = sorted(set(target_labels), key=int)
    ri = {x: i for i, x in enumerate(rows)}
    ci = {x: i for i, x in enumerate(cols)}
    counts = np.zeros((len(rows), len(cols)), dtype=np.int64)
    np.add.at(counts, (np.repeat([ri[x] for x in source_labels], 30), np.array([ci[x] for x in target_labels])[neighbors.ravel()]), 1)
    return pd.DataFrame(counts, index=rows, columns=cols)


def clean_pairs(embedding, assignments, formal, reference, group):
    stats = load_module(ROOT / "01_tool/04_downstream/edge_statistics.py", "edge_statistics")
    if len(embedding) != len(assignments):
        raise ValueError("Assignments must have one ordered row per embedding cell")
    conditions = assignments["condition"].astype(str).to_numpy()
    labels = assignments["predicted_cluster"].astype(str).to_numpy()
    ref, dis = conditions == reference, conditions == group
    if not np.all(ref | dis):
        raise ValueError("Unexpected condition in assignments")
    forward = directed_edges(embedding[dis], embedding[ref], labels[dis], labels[ref])
    reverse = directed_edges(embedding[ref], embedding[dis], labels[ref], labels[dis])
    rows = []
    for _, pair in formal.iterrows():
        r, g = str(int(pair["control_cluster"])), str(int(pair["ipf_cluster"]))
        f = stats.fisher_from_matrix(forward, g, r)
        b = stats.fisher_from_matrix(reverse, r, g)
        row = pair.to_dict()
        for direction, values in (("forward", f), ("reverse", b)):
            for label, value in zip(("a", "b", "c", "d", "fisher_pvalue"), values):
                row[f"{direction}_{label}"] = value
        row["reference_cluster_size"] = int((labels[ref] == r).sum())
        row["group_cluster_size"] = int((labels[dis] == g).sum())
        row["pair_cluster_label"] = f"C{r}->D{g}"
        rows.append(row)
    tests = pd.DataFrame(rows)
    # The BH family is all formal pairs, before the cluster-size exclusion.
    for direction in ("forward", "reverse"):
        tests[f"{direction}_fisher_qvalue_bh"] = stats.bh_qvalues(tests[f"{direction}_fisher_pvalue"])
    tests["min_cluster_size"] = tests[["reference_cluster_size", "group_cluster_size"]].min(axis=1)
    keep = (tests.min_cluster_size >= 100) & (tests.forward_fisher_qvalue_bh < 0.05) & (tests.reverse_fisher_qvalue_bh < 0.05)
    return tests, tests.loc[keep].copy(), forward, reverse


def ranked_table(pair_run, pair_id):
    files = list((pair_run / "02_deg/01_full_ranked_tables").glob(pair_id + "_*.tsv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one ranked DEG table for {pair_id}: {files}")
    return pd.read_csv(files[0], sep="\t")


def gene_symbols(cfg):
    import anndata as ad

    matrix = ad.read_h5ad(cfg["combined_lognorm"], backed="r")
    lookup = {str(x): str(x) for x in matrix.var_names}
    for key in ("gene_symbol", "gene_symbols", "symbol", "features"):
        if key in matrix.var:
            for name, symbol in zip(matrix.var_names.astype(str), matrix.var[key].astype(str)):
                if symbol and symbol not in {"nan", "None"}:
                    lookup[name] = symbol
    matrix.file.close()
    annotation = cfg.get("gene_symbol_map")
    if not annotation and cfg.get("reference_dir"):
        annotation = str(Path(cfg["reference_dir"]) / "gene_symbols.tsv")
    if annotation and Path(annotation).is_file():
        frame = pd.read_csv(annotation, sep="\t")
        lookup.update(zip(frame.gene_id.astype(str), frame.gene_symbol.astype(str)))
    return lookup


def passing_genes(deg):
    return deg.loc[(pd.to_numeric(deg.pvals_adj) < 0.05) & (pd.to_numeric(deg.logfoldchanges) > 1.0)].copy()


def markers(cfg, pair_run, pairs, output):
    import anndata as ad
    from scipy import sparse

    matrix = ad.read_h5ad(cfg["combined_lognorm"])
    assignments = pd.read_csv(pair_run / "combined_condition_assignments.tsv.gz", sep="\t")
    symbols = gene_symbols(cfg)
    rows, columns, indices = [], [], []
    pairs = pairs.sort_values(["control_cluster", "ipf_cluster"])
    for _, pair in pairs.iterrows():
        deg = passing_genes(ranked_table(pair_run, pair.pair_id))
        deg["gene_symbol"] = deg.names.astype(str).map(symbols).fillna(deg.names)
        if cfg.get("exclude_viral_markers", False):
            deg = deg.loc[~deg.names.astype(str).str.startswith("VIRAL_")]
        deg = deg.loc[~deg.gene_symbol.str.startswith("ENSG")].sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False]).drop_duplicates("gene_symbol").head(5)
        for rank, (_, gene) in enumerate(deg.iterrows(), 1):
            columns.append({"column_id": f"{pair.pair_id}|rank{rank}|{gene['names']}", "pair_id": pair.pair_id, "gene_id": gene["names"], "gene_symbol": gene.gene_symbol, "pvals_adj": gene.pvals_adj, "logfoldchanges": gene.logfoldchanges})
        for condition, cluster in ((cfg["condition1_name"], int(pair.control_cluster)), (cfg["condition2_name"], int(pair.ipf_cluster))):
            cells = assignments.loc[(assignments.condition == condition) & (assignments.predicted_cluster.astype(int) == cluster), "cell_id"].astype(str)
            loc = matrix.obs_names.get_indexer(cells)
            if (loc < 0).any() or not len(loc):
                raise ValueError("Matched cells missing from normalized expression matrix")
            indices.append(loc)
            rows.append({"row_id": f"{pair.pair_id}|{condition}", "pair_id": pair.pair_id, "condition": condition, "cluster": cluster, "row_label": f"C{int(pair.control_cluster)}->D{int(pair.ipf_cluster)}_{condition}"})
    cols = pd.DataFrame(columns)
    if cols.empty:
        raise ValueError("No eligible markers; full DEG remains available")
    gene_indices = matrix.var_names.get_indexer(cols.gene_id)
    if (gene_indices < 0).any():
        raise ValueError("Marker genes missing from matrix")
    means = []
    for loc in indices:
        x = matrix.X[loc, :][:, gene_indices]
        means.append(np.asarray(x.mean(axis=0)).ravel() if sparse.issparse(x) else np.mean(x, axis=0))
    values = np.asarray(means)
    sd = values.std(axis=0)
    sd[sd == 0] = 1
    pd.DataFrame(rows).to_csv(output / "heatmap_rows.tsv", sep="\t", index=False)
    cols.to_csv(output / "heatmap_columns.tsv", sep="\t", index=False)
    for name, x in (("mean_expression", values), ("heatmap_zscore", (values - values.mean(axis=0)) / sd)):
        pd.DataFrame(x, index=[r["row_id"] for r in rows], columns=cols.column_id).rename_axis("row_id").to_csv(output / (name + ".tsv"), sep="\t")


def run_stage(stage, cfg, pair_run, output, clean_path=None):
    if stage == "qc":
        run = json.loads((pair_run / "run.json").read_text())
        assignments = pd.read_csv(pair_run / "combined_condition_assignments.tsv.gz", sep="\t")
        formal = pd.read_csv(pair_run / "formal_pairs.tsv", sep="\t")
        tests, clean, forward, reverse = clean_pairs(np.load(run["embedding"]), assignments, formal, cfg["condition1_name"], cfg["condition2_name"])
        for name, table in (("all_tests", tests), ("clean_pairs", clean)):
            table.to_csv(output / (name + ".tsv"), sep="\t", index=False)
        forward.to_csv(output / "forward_edge_counts.tsv", sep="\t")
        reverse.to_csv(output / "reverse_edge_counts.tsv", sep="\t")
        return
    pairs = pd.read_csv(clean_path, sep="\t")
    if stage == "markers":
        markers(cfg, pair_run, pairs, output)
        return
    api = load_module(ROOT / "01_tool/04_downstream/toppgene_api.py", "toppgene_api")
    symbols = gene_symbols(cfg)
    collection = api.DegCollection("wilcoxon_version_b", "wilcoxon", "version_b", "filtered_kl_pair", pair_run)
    opts = argparse.Namespace(min_mapped_genes=2, max_retries=3, sleep_seconds=0.5, request_timeout_seconds=30, fdr_threshold=0.05)
    failures = []
    for pair in pairs.itertuples():
        deg = passing_genes(ranked_table(pair_run, pair.pair_id))
        genes = [symbols.get(str(name), str(name)) for name in deg.names]
        result, _ = api.run_toppgene(genes, collection, pair.pair_id, "filtered_kl_pair", output / pair.pair_id, opts)
        if "failed" in result["status"]:
            failures.append(result)
    if failures:
        raise RuntimeError(f"ToppGene requests failed; inspect saved summaries: {failures}")
