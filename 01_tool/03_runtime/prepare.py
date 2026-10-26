"""Prepare user counts with the original joint QC/normalization/HVG ordering."""

import json

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def prepare_inputs(cfg, output):
    matrices = [ad.read_h5ad(cfg[k]) for k in ("condition1_input", "condition2_input")]
    if not matrices[0].var_names.equals(matrices[1].var_names):
        raise ValueError("Supply identical ordered feature identifiers in both full-feature raw inputs")
    names = [cfg["condition1_name"], cfg["condition2_name"]]
    for x, condition, label in zip(matrices, names, (cfg["label_key_1"], cfg["label_key_2"])):
        if not x.obs_names.is_unique or not x.var_names.is_unique or label not in x.obs:
            raise ValueError("Inputs need unique identifiers and the configured annotation column")
        values = x.X.data if sparse.issparse(x.X) else np.asarray(x.X)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values)):
            raise ValueError("Input .X must contain raw nonnegative integer counts")
        x.obs_names = [i if i.startswith(condition + "::") else condition + "::" + i for i in x.obs_names.astype(str)]
        x.obs["condition"] = condition
    combined = ad.concat(matrices, join="inner", merge="first")
    sc.pp.filter_cells(combined, min_genes=200)
    sc.pp.filter_genes(combined, min_cells=3)
    log = combined.copy()
    sc.pp.normalize_total(log, target_sum=10000)
    sc.pp.log1p(log)
    sc.pp.highly_variable_genes(log, n_top_genes=2000, inplace=True)
    mask = log.var["highly_variable"].to_numpy()
    if int(mask.sum()) != 2000:
        raise ValueError(f"Expected 2000 HVGs, observed {int(mask.sum())}; inspect your data, do not silently alter the recipe")
    raw = combined[:, mask].copy()
    log = log[:, mask].copy()
    for i, condition in enumerate(names, 1):
        raw[raw.obs.condition == condition].copy().write_h5ad(output / f"condition{i}_raw_top2000_hvg.h5ad", compression="gzip")
    raw.write_h5ad(output / "combined_raw_top2000_hvg.h5ad", compression="gzip")
    log.write_h5ad(output / "combined_lognorm_top2000_hvg.h5ad", compression="gzip")
    prepared = dict(cfg)
    prepared["condition1_input"] = "condition1_raw_top2000_hvg.h5ad"
    prepared["condition2_input"] = "condition2_raw_top2000_hvg.h5ad"
    prepared["combined_lognorm"] = "combined_lognorm_top2000_hvg.h5ad"
    for key in ("embedding", "reference_dir"):
        prepared.pop(key, None)
    (output / "prepared_config.json").write_text(json.dumps(prepared, indent=2))
