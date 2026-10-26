"""Render newly computed analysis stages; never substitutes frozen result values."""

import json
from pathlib import Path

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scanpy as sc

from results import export_panels
from workflow import ROOT, load_module


def plot_run_results(cfg, args, output):
    run = json.loads((args.pair_run / "run.json").read_text())
    embedding = np.load(run["embedding"])
    assignments = pd.read_csv(args.pair_run / "combined_condition_assignments.tsv.gz", sep="\t")
    adata = sc.AnnData(X=np.zeros((len(embedding), 1), dtype=np.float32))
    adata.obsm["X_scbeacon"] = embedding
    sc.pp.neighbors(adata, use_rep="X_scbeacon", n_neighbors=10, metric="cosine", random_state=0)
    sc.tl.umap(adata, min_dist=.8, spread=1.0, init_pos="spectral", random_state=0)
    cells = assignments[["cell_id", "condition"]].copy()
    cells["cell_type"] = assignments.true_label.astype(str)
    previous = Path(cfg.get("reference_dir", ".")) / "umap.tsv.gz"
    if previous.is_file():
        # Transfer only archived display annotations by barcode, never coordinates.
        previous = pd.read_csv(previous, sep="\t").set_index("cell_id")
        cells["cell_type"] = cells.cell_id.map(previous.cell_type).fillna(cells.cell_type)
    prefixes = np.where(cells.condition == cfg["condition1_name"], "C", "D")
    cells["cluster"] = prefixes + assignments.predicted_cluster.astype(str).to_numpy()
    cells[["umap_1", "umap_2"]] = adata.obsm["X_umap"]
    tables = output / "01_source_values"
    tables.mkdir()
    cells.to_csv(tables / "umap.tsv.gz", sep="\t", index=False)
    palette = []
    for kind in ("cell_type", "cluster"):
        labels = cells[kind].drop_duplicates().tolist()
        for i, label in enumerate(labels):
            palette.append({"kind": kind, "label": label, "color": colors.to_hex(colors.hsv_to_rgb(((i * .61803398875) % 1, .55, .78)))})
    pd.DataFrame(palette).to_csv(tables / "colors.tsv", sep="\t", index=False)
    clean = pd.read_csv(args.qc_run / "clean_pairs.tsv", sep="\t")
    clean.to_csv(tables / "clean_pairs.tsv", sep="\t", index=False)
    for name in ("heatmap_rows.tsv", "heatmap_columns.tsv", "heatmap_zscore.tsv"):
        data = pd.read_csv(args.marker_run / name, sep="\t")
        if name == "heatmap_rows.tsv":
            data["pair_label"] = data.pair_id.map(clean.set_index("pair_id").pair_cluster_label)
        if name == "heatmap_columns.tsv":
            data["highlight"] = False
        data.to_csv(tables / name, sep="\t", index=False)
    selector = load_module(ROOT / "01_tool/04_downstream/pathway_display.py", "pathway_display")
    term_rows = []
    wanted = cfg.get("display_pair_labels")
    if not wanted and cfg.get("reference_dir") and Path(cfg["reference_dir"]).is_dir():
        frozen_terms = Path(cfg["reference_dir"]) / "selected_pathways.tsv"
        if frozen_terms.is_file():
            wanted = pd.read_csv(frozen_terms, sep="\t").pair_label.unique().tolist()
    for pair in clean.itertuples():
        if wanted and pair.pair_cluster_label not in wanted:
            continue
        path = args.enrichment_run / pair.pair_id / "06_toppgene_annotations.tsv"
        terms = pd.read_csv(path, sep="\t")
        terms = terms.loc[terms.Category.eq("Pathway") & terms.Source.isin(["Reactome Pathways", "WikiPathways"])].copy()
        terms["qvalue_fdr_bh"] = pd.to_numeric(terms.QValueFDRBH, errors="coerce")
        terms["pvalue"] = pd.to_numeric(terms.PValue, errors="coerce")
        terms = terms.loc[terms.qvalue_fdr_bh < .05].copy()
        terms["term_clean"] = terms.Name.map(selector.clean_term)
        selected = selector.select_nonoverlap_terms(terms)
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            item = row.to_dict()
            item.update({"pair_label": pair.pair_cluster_label, "term": selector.display_term(row.Name), "qvalue": row.qvalue_fdr_bh, "rank": rank, "highlight": False})
            term_rows.append(item)
    if wanted and set(wanted) - set(clean.pair_cluster_label):
        raise ValueError("A requested display pair is not retained in this new run. Inspect results; do not force its retention.")
    pd.DataFrame(term_rows, columns=list(term_rows[0]) if term_rows else ["pair_label", "term", "qvalue", "rank", "highlight"]).to_csv(tables / "selected_pathways.tsv", sep="\t", index=False)
    export_panels(cfg["dataset_id"], output / "02_individual_panels", reference=tables)
    (output / "run_manifest.json").write_text(json.dumps({"mode": "new_scientific_run", "umap_recomputed": True, "red_highlights": "none assigned automatically to new terms/genes", "note": "Current annotation/API versions may change terms and p-values; no frozen numerical results substituted"}, indent=2))
