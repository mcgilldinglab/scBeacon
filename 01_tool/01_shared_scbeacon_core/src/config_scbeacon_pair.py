"""Dataset-local configuration for paired scBeacon runs."""

import json
import os
import sys

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
import scipy
import torch
from scipy.sparse import csc_matrix

sys.path.append(r"../")
from unifan.datasets1 import AnnDataset


def _env(name, default=None):
    return os.environ.get(name, default)


def _env_int(name, default):
    value = _env(name)
    return int(value) if value not in (None, "") else default


def _env_float(name, default):
    value = _env(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name, default):
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(path_value, base_dir):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


class Config(object):
    def __init__(self):
        self.core_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.workspace_root = os.path.abspath(
            _env("SCBEACON_WORKSPACE_ROOT", os.path.join(self.core_root, ".."))
        )

        self.dataset_id = _env("SCBEACON_DATASET_ID", "unnamed_dataset_pair")
        self.condition_name_1 = _env("SCBEACON_CONDITION_NAME_1", "condition_1")
        self.condition_name_2 = _env("SCBEACON_CONDITION_NAME_2", "condition_2")
        self.label_key_1 = _env("SCBEACON_LABEL_KEY_1", _env("SCBEACON_LABEL_KEY", "label"))
        self.label_key_2 = _env("SCBEACON_LABEL_KEY_2", _env("SCBEACON_LABEL_KEY", "label"))
        self.standard_label_key = "scbeacon_label"

        self.data_filepath1 = _resolve_path(_env("SCBEACON_DATA_FILE_1"), self.workspace_root)
        self.data_filepath2 = _resolve_path(_env("SCBEACON_DATA_FILE_2"), self.workspace_root)
        if self.data_filepath1 is None or self.data_filepath2 is None:
            raise ValueError("SCBEACON_DATA_FILE_1 and SCBEACON_DATA_FILE_2 must be set.")

        self.output_path = _resolve_path(
            _env("SCBEACON_OUTPUT_ROOT", "../03_run_records/02_output/full_run"),
            self.workspace_root,
        )
        os.makedirs(self.output_path, exist_ok=True)

        self.apply_lognorm = _env_bool("SCBEACON_APPLY_LOGNORM", True)
        self.use_raw_decoder_target = _env_bool("SCBEACON_USE_RAW_DECODER_TARGET", False)
        self.combine_preprocess = _env_bool("SCBEACON_COMBINE_PREPROCESS", False)
        self.use_hvg_subset = _env_bool("SCBEACON_USE_HVG_SUBSET", False)
        self.input_transform = _env("SCBEACON_INPUT_TRANSFORM", "").strip().lower()
        if not self.input_transform:
            self.input_transform = "lognorm" if self.apply_lognorm else "raw"
        if self.input_transform not in {"raw", "lognorm", "log1p_raw"}:
            raise ValueError(
                "SCBEACON_INPUT_TRANSFORM must be one of: raw, lognorm, log1p_raw."
            )
        self.max_cells_per_condition = _env_int("SCBEACON_MAX_CELLS_PER_CONDITION", 0)
        self.subset_seed = _env_int("SCBEACON_SUBSET_SEED", 0)
        self.leiden_resolution = _env_float("SCBEACON_LEIDEN_RESOLUTION", 1.0)
        self.target_sum = _env_float("SCBEACON_TARGET_SUM", 1e4)
        self.filter_min_genes = _env_int("SCBEACON_FILTER_MIN_GENES", 200)
        self.filter_min_cells = _env_int("SCBEACON_FILTER_MIN_CELLS", 3)
        self.hvg_min_mean = _env_float("SCBEACON_HVG_MIN_MEAN", 0.0)
        self.hvg_max_mean = _env_float("SCBEACON_HVG_MAX_MEAN", 1000.0)
        self.hvg_min_disp = _env_float("SCBEACON_HVG_MIN_DISP", 0.01)
        self.debug_stop_after_stage = _env("SCBEACON_DEBUG_STOP_AFTER_STAGE")
        self.debug_stop_after_epoch = _env_int("SCBEACON_DEBUG_STOP_AFTER_EPOCH", -1)

        requested_cuda = _env_bool("SCBEACON_USE_CUDA", True)
        self.use_cuda = requested_cuda and torch.cuda.is_available()
        self.num_workers = _env_int("SCBEACON_NUM_WORKERS", 1)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.pin_memory = self.use_cuda
        self.non_blocking = self.use_cuda
        self.random_seed = _env_int("SCBEACON_RANDOM_SEED", 0)

        self.prior_name = "c5.go.bp.v7.4.symbols.gmt+c2.cp.v7.4.symbols.gmt+TF-DNA"
        self.features_type = "gene_gene_sets"
        self.alpha = _env_float("SCBEACON_ALPHA", 1e-3)
        self.beta = _env_float("SCBEACON_BETA", 1e-5)
        self.weight_decay = _env_float("SCBEACON_WEIGHT_DECAY", 1e-5)
        self.tau = _env_float("SCBEACON_TAU", 10.0)
        self.z_dim = _env_int("SCBEACON_Z_DIM", 128)
        self.batch_size = _env_int("SCBEACON_BATCH_SIZE", 4096)
        self.num_iteration = _env_int("SCBEACON_NUM_ITERATION", 40)
        self.num_epochs_z = _env_int("SCBEACON_NUM_EPOCHS_Z", 20)
        self.pretrain_epoch = _env_int("SCBEACON_PRETRAIN_EPOCH", 20)
        self.anno_epoch = _env_int("SCBEACON_ANNO_EPOCH", 20)
        self.cont_epoch = _env_int("SCBEACON_CONT_EPOCH", 5)

        self.r_encoder_layers = 5
        self.r_decoder_layers = 1
        self.r_encoder_dim = 128
        self.r_decoder_dim = 128
        self.rnetwork = "non-negative"
        self.z_encoder_layers = _env_int("SCBEACON_Z_ENCODER_LAYERS", 3)
        self.z_decoder_layers = _env_int("SCBEACON_Z_DECODER_LAYERS", 2)
        self.z_encoder_dim = _env_int("SCBEACON_Z_ENCODER_DIM", self.z_dim * 4)
        self.z_decoder_dim = _env_int("SCBEACON_Z_DECODER_DIM", self.z_dim * 4)

        self.pre_project = "01_pretrain"
        self.anno_project = "02_annocluster"
        self.cont_project = "03_contrastive"
        self.pretrain_output_parent_path = os.path.join(self.output_path, self.pre_project)
        self.pretrain_z_folder = os.path.join(self.pretrain_output_parent_path, "pretrain_z")
        self.input_z_ZINB_path = os.path.join(self.pretrain_z_folder, "pretrain_z_ZINB_optimal_1.npy")
        self.input_ae_ZINB_path = os.path.join(
            self.pretrain_z_folder, "pretrain_z_ZINB_model_optimal.pickle"
        )
        self.anno_output_parent_path = os.path.join(self.output_path, self.anno_project)
        self.annocluster_ZINB_folder = os.path.join(
            self.anno_output_parent_path, f"annocluster_ZINB_{self.features_type}"
        )
        self.contrastive_output_parent_path = os.path.join(self.output_path, self.cont_project)
        self.model_contrastive_folder = os.path.join(
            self.contrastive_output_parent_path, f"contrastive_{self.features_type}"
        )

        self.adata1 = self._load_dataset_member(
            self.data_filepath1, self.label_key_1, self.condition_name_1
        )
        self.adata2 = self._load_dataset_member(
            self.data_filepath2, self.label_key_2, self.condition_name_2
        )
        if self.combine_preprocess:
            self._prepare_combined_expression_space()
        else:
            self._validate_gene_space()

            self.x_raw1 = self._to_dense_array(self.adata1.X).astype(np.float32, copy=False)
            self.x_raw2 = self._to_dense_array(self.adata2.X).astype(np.float32, copy=False)

            self._apply_input_transform(self.adata1)
            self._apply_input_transform(self.adata2)

            self.x1 = self._to_dense_array(self.adata1.X).astype(np.float32, copy=False)
            self.x2 = self._to_dense_array(self.adata2.X).astype(np.float32, copy=False)
            self.x = np.vstack((self.x1, self.x2))
            if self.use_raw_decoder_target:
                self.x_raw = np.vstack((self.x_raw1, self.x_raw2))
            else:
                self.x_raw1 = self.x1
                self.x_raw2 = self.x2
                self.x_raw = self.x

        self.N_project1 = self.adata1.n_obs
        self.N_project2 = self.adata2.n_obs
        self.N = self.N_project1 + self.N_project2
        self.G = self.adata1.n_vars
        self.clusters_true1 = self.adata1.obs[self.standard_label_key].to_numpy()
        self.clusters_true2 = self.adata2.obs[self.standard_label_key].to_numpy()
        self.expression_integrated = AnnDataset(
            self.x,
            raw_array=self.x_raw,
            return_raw_target=self.use_raw_decoder_target,
        )

        self._write_run_manifest(requested_cuda)

    def _load_dataset_member(self, path, label_key, condition_name):
        data = sc.read(path, dtype="float64", backed="r")
        obs = data.obs.copy()
        var = data.var.copy()
        indices = np.arange(data.n_obs)
        if self.max_cells_per_condition > 0 and data.n_obs > self.max_cells_per_condition:
            rng = np.random.default_rng(self.subset_seed)
            indices = np.sort(rng.choice(indices, size=self.max_cells_per_condition, replace=False))

        x_matrix = self._read_h5ad_matrix_subset(path, indices)
        adata = sc.AnnData(X=x_matrix, obs=obs.iloc[indices].copy(), var=var.copy())
        if hasattr(data, "file") and data.file is not None:
            data.file.close()

        if label_key not in adata.obs.columns:
            raise KeyError(f"Label key '{label_key}' not found in {path}")

        adata.obs[self.standard_label_key] = adata.obs[label_key].astype(str)
        adata.obs["scbeacon_condition"] = condition_name
        return adata

    def _read_h5ad_matrix_subset(self, path, row_indices):
        with h5py.File(path, "r") as handle:
            x_node = handle["X"]
            encoding = x_node.attrs.get("encoding-type")
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")

            if encoding == "csr_matrix":
                indptr = x_node["indptr"]
                indices_ds = x_node["indices"]
                data_ds = x_node["data"]
                shape = tuple(int(v) for v in x_node.attrs["shape"])

                new_indptr = [0]
                chunks_indices = []
                chunks_data = []
                for row in row_indices:
                    start = int(indptr[row])
                    end = int(indptr[row + 1])
                    chunks_indices.append(indices_ds[start:end])
                    chunks_data.append(data_ds[start:end])
                    new_indptr.append(new_indptr[-1] + (end - start))

                if chunks_data:
                    all_indices = np.concatenate(chunks_indices).astype(np.int64, copy=False)
                    all_data = np.concatenate(chunks_data)
                else:
                    all_indices = np.array([], dtype=np.int64)
                    all_data = np.array([], dtype=np.float64)

                return scipy.sparse.csr_matrix(
                    (
                        all_data,
                        all_indices,
                        np.asarray(new_indptr, dtype=np.int64),
                    ),
                    shape=(len(row_indices), shape[1]),
                )

            if isinstance(x_node, h5py.Dataset):
                dense = np.asarray(x_node[row_indices, :], dtype=np.float64)
                return scipy.sparse.csr_matrix(dense)

            raise NotImplementedError(
                f"Unsupported H5AD X encoding '{encoding}' for {path}"
            )

    def _validate_gene_space(self):
        if self.adata1.n_vars != self.adata2.n_vars:
            raise ValueError(
                f"Gene dimension mismatch: {self.adata1.n_vars} vs {self.adata2.n_vars}"
            )
        if not np.array_equal(self.adata1.var_names.to_numpy(), self.adata2.var_names.to_numpy()):
            raise ValueError("Gene ordering mismatch between the paired datasets.")

    def _apply_input_transform(self, adata):
        if self.input_transform == "raw":
            return
        if self.input_transform == "lognorm":
            sc.pp.normalize_total(adata, target_sum=self.target_sum, inplace=True)
            sc.pp.log1p(adata)
            return
        if self.input_transform == "log1p_raw":
            sc.pp.log1p(adata)
            return
        raise ValueError(f"Unsupported input transform: {self.input_transform}")

    def _prepare_combined_expression_space(self):
        combined = ad.concat([self.adata1, self.adata2], join="inner", merge="same")
        if self.filter_min_genes > 0:
            sc.pp.filter_cells(combined, min_genes=self.filter_min_genes)
        if self.filter_min_cells > 0:
            sc.pp.filter_genes(combined, min_cells=self.filter_min_cells)

        decoder_target = self._to_dense_array(combined.X).astype(np.float32, copy=False)

        self._apply_input_transform(combined)

        if self.use_hvg_subset:
            sc.pp.highly_variable_genes(
                combined,
                min_mean=self.hvg_min_mean,
                max_mean=self.hvg_max_mean,
                min_disp=self.hvg_min_disp,
                inplace=True,
            )
            hvg_mask = combined.var["highly_variable"].to_numpy(dtype=bool)
            if not np.any(hvg_mask):
                raise RuntimeError("No HVGs were selected during combined preprocessing.")
            combined = combined[:, hvg_mask].copy()
            decoder_target = decoder_target[:, hvg_mask]

        processed = self._to_dense_array(combined.X).astype(np.float32, copy=False)
        if not self.use_raw_decoder_target:
            decoder_target = processed.copy()

        condition_codes = combined.obs["scbeacon_condition"].to_numpy()
        mask1 = condition_codes == self.condition_name_1
        mask2 = condition_codes == self.condition_name_2

        self.adata1 = combined[mask1].copy()
        self.adata2 = combined[mask2].copy()
        self.x1 = processed[mask1]
        self.x2 = processed[mask2]
        self.x = processed
        self.x_raw1 = decoder_target[mask1]
        self.x_raw2 = decoder_target[mask2]
        self.x_raw = decoder_target

    def _to_dense_array(self, matrix):
        if scipy.sparse.issparse(matrix):
            return matrix.toarray()
        return np.asarray(matrix)

    def _write_run_manifest(self, requested_cuda):
        manifest = {
            "dataset_id": self.dataset_id,
            "condition_name_1": self.condition_name_1,
            "condition_name_2": self.condition_name_2,
            "label_key_1": self.label_key_1,
            "label_key_2": self.label_key_2,
            "data_filepath1": self.data_filepath1,
            "data_filepath2": self.data_filepath2,
            "output_path": self.output_path,
            "apply_lognorm": self.apply_lognorm,
            "input_transform": self.input_transform,
            "use_raw_decoder_target": self.use_raw_decoder_target,
            "combine_preprocess": self.combine_preprocess,
            "use_hvg_subset": self.use_hvg_subset,
            "max_cells_per_condition": self.max_cells_per_condition,
            "subset_seed": self.subset_seed,
            "leiden_resolution": self.leiden_resolution,
            "target_sum": self.target_sum,
            "filter_min_genes": self.filter_min_genes,
            "filter_min_cells": self.filter_min_cells,
            "hvg_min_mean": self.hvg_min_mean,
            "hvg_max_mean": self.hvg_max_mean,
            "hvg_min_disp": self.hvg_min_disp,
            "use_cuda_requested": requested_cuda,
            "use_cuda_effective": self.use_cuda,
            "n_obs_condition_1": int(self.N_project1),
            "n_obs_condition_2": int(self.N_project2),
            "n_vars": int(self.G),
        }
        with open(os.path.join(self.output_path, "run_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
