import os
import pickle
import random

from config_scbeacon_pair import Config
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import preprocessing

from unifan.annoclusterV4 import AnnoCluster
from unifan.autoencoderV2 import autoencoder
from unifan.KLCLRv4 import KLCLR
from unifan.datasets1 import ContrDataset
from unifan.trainer import DebugStopRequested, Trainer


class Train(Config):
    def _set_global_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.random_seed)

    def get_kl_div_pair(self, centroids_torch_project1, centroids_torch_project2):
        pair_1 = {}
        centroids_torch_project1 = torch.Tensor(centroids_torch_project1)
        centroids_torch_project2 = torch.Tensor(centroids_torch_project2)
        for i in range(centroids_torch_project1.shape[0]):
            z_dist_min = float("inf")
            for j in range(centroids_torch_project2.shape[0]):
                z_dist = F.kl_div(
                    centroids_torch_project1[i].softmax(dim=-1).log(),
                    centroids_torch_project2[j].softmax(dim=-1),
                    reduction="sum",
                )
                if z_dist < z_dist_min:
                    z_dist_min = z_dist
                    inx = j
            pair_1[i] = inx
        return pair_1

    def leiden_cluster(self, z_init):
        adata = sc.AnnData(X=z_init)
        adata.obsm["X_unifan"] = z_init
        sc.pp.neighbors(
            adata,
            n_pcs=self.z_dim,
            use_rep="X_unifan",
            random_state=self.random_seed,
        )
        sc.tl.leiden(
            adata,
            resolution=self.leiden_resolution,
            random_state=self.random_seed,
        )
        clusters_pre = adata.obs["leiden"].astype("int").values

        try:
            df_cluster = pd.DataFrame(z_init.detach().cpu().numpy())
        except AttributeError:
            df_cluster = pd.DataFrame(z_init)
        df_cluster["cluster"] = clusters_pre
        centroids = df_cluster.groupby("cluster").mean().values
        centroids_torch = torch.from_numpy(centroids)
        return clusters_pre, centroids_torch

    def _runtime_state_path(self, filename):
        runtime_root = os.path.join(self.output_path, "_runtime_state")
        os.makedirs(runtime_root, exist_ok=True)
        return os.path.join(runtime_root, filename)

    def _to_numpy(self, value):
        try:
            return value.detach().cpu().numpy()
        except AttributeError:
            pass
        try:
            return value.cpu().numpy()
        except AttributeError:
            pass
        return np.asarray(value)

    def _save_pickle(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def _save_cluster_state(self, path, clusters_pre1, centroids_1, clusters_pre2, centroids_2):
        payload = {
            "clusters_pre1": np.asarray(clusters_pre1),
            "clusters_pre2": np.asarray(clusters_pre2),
            "centroids_1": self._to_numpy(centroids_1),
            "centroids_2": self._to_numpy(centroids_2),
        }
        self._save_pickle(path, payload)

    def _load_cluster_state(self, path):
        payload = self._load_pickle(path)
        return (
            payload["clusters_pre1"],
            torch.from_numpy(payload["centroids_1"]),
            payload["clusters_pre2"],
            torch.from_numpy(payload["centroids_2"]),
        )

    def _resume_completed(self, resume_state):
        return resume_state is not None and bool(resume_state.get("completed", False))

    def _require_completed_resume(self, resume_state, resume_state_path, stage_name):
        if resume_state is None:
            raise RuntimeError(
                f"{stage_name} requires a completed resume state at {resume_state_path}. "
                f"Resume no longer falls back to optimal checkpoints."
            )
        if not self._resume_completed(resume_state):
            raise RuntimeError(
                f"{stage_name} resume state exists but is not marked completed at {resume_state_path}."
            )

    def _debug_stop_epoch(self, stage_name):
        return self.debug_stop_after_epoch if self.debug_stop_after_stage == stage_name else None

    def _env_bool_local(self, name, default=False):
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _compatibility_profile(self):
        profile = os.environ.get("SCBEACON_COMPATIBILITY_PROFILE", "covid09")
        if profile not in {"ipf09", "covid09"}:
            raise ValueError("SCBEACON_COMPATIBILITY_PROFILE must be ipf09 or covid09")
        return profile

    def _pretrain_output_option(self, name):
        profiles = {
            "ipf09": {
                "save_infer": True,
                "save_epoch_embeddings": True,
                "save_optimal_model_files": True,
                "save_stats_files": True,
            },
            "covid09": {
                "save_infer": False,
                "save_epoch_embeddings": False,
                "save_optimal_model_files": False,
                "save_stats_files": False,
            },
        }
        return profiles[self._compatibility_profile()][name]

    def run(self):
        self._set_global_seed()
        label_series = pd.concat(
            [
                self.adata1.obs[self.standard_label_key],
                self.adata2.obs[self.standard_label_key],
            ],
            axis=0,
        )
        sex_mapping = {label: idx for idx, label in enumerate(pd.Index(label_series).unique())}

        model_autoencoder = autoencoder(
            input_dim=self.G,
            z_dim=self.z_dim,
            encoder_dim=self.z_encoder_dim,
            emission_dim=self.z_decoder_dim,
            num_layers_encoder=self.z_encoder_layers,
            num_layers_decoder=self.z_decoder_layers,
            reconstruction_network="gaussian",
            decoding_network="gaussian",
            use_cuda=self.use_cuda,
        )

        pretrain_resume_state_path = self._runtime_state_path("pretrain_z_ZINB_resume_state_1.pickle")
        pretrain_cluster_state_path = self._runtime_state_path("pretrain_cluster_state.pickle")
        trainer = Trainer(
            dataset=self.expression_integrated,
            model=model_autoencoder,
            learning_rate=self.alpha,
            model_name="pretrain_z_ZINB",
            batch_size=self.batch_size,
            num_epochs=self.pretrain_epoch,
            save_infer=self._pretrain_output_option("save_infer"),
            output_folder=self.pretrain_z_folder,
            num_workers=self.num_workers,
            use_cuda=self.use_cuda,
            iteration=1,
            resume_state_path=pretrain_resume_state_path,
            debug_stop_after_epoch=self._debug_stop_epoch("pretrain"),
            debug_stage_name="pretrain",
            save_epoch_embeddings=self._pretrain_output_option("save_epoch_embeddings"),
            save_optimal_model_files=self._pretrain_output_option("save_optimal_model_files"),
            save_resume_state_files=True,
            save_stats_files=self._pretrain_output_option("save_stats_files"),
        )

        if os.path.isfile(pretrain_cluster_state_path):
            self._require_completed_resume(
                trainer.resume_state,
                pretrain_resume_state_path,
                "Pretrain cluster state",
            )
            (
                clusters_pre1,
                centroids_torch_project1,
                clusters_pre2,
                centroids_torch_project2,
            ) = self._load_cluster_state(pretrain_cluster_state_path)
        else:
            if self._resume_completed(trainer.resume_state):
                print("Pretrain training already completed. Rebuilding the initial clustering state.")
                z_init = trainer.infer_z_ZINB()
                np.save(self.input_z_ZINB_path, z_init)
            else:
                if trainer.resume_state is None:
                    print("Start pretraining the autoencoder model ... ")
                else:
                    print(
                        f"Resume pretraining the autoencoder model from epoch {trainer.start_epoch} ..."
                    )
                trainer.train()
                z_init = trainer.infer_z_ZINB()
                np.save(self.input_z_ZINB_path, z_init)

            z_init = self._to_numpy(z_init)
            z_init1 = z_init[: self.N_project1]
            z_init2 = z_init[self.N_project1 :]
            clusters_pre1, centroids_torch_project1 = self.leiden_cluster(z_init1)
            clusters_pre2, centroids_torch_project2 = self.leiden_cluster(z_init2)
            self._save_cluster_state(
                pretrain_cluster_state_path,
                clusters_pre1,
                centroids_torch_project1,
                clusters_pre2,
                centroids_torch_project2,
            )

        if self._compatibility_profile() == "covid09" and self._env_bool_local("SCBEACON_PRETRAIN_ONLY", False):
            print("SCBEACON_PRETRAIN_ONLY=true. Stop after pretrain state preparation.")
            return

        subjects = np.hstack((np.zeros(self.N_project1), np.ones(self.N_project2)))
        model_contrastive = None
        for iteration in range(1, self.num_iteration + 1):
            annocluster_ae_ZINB_path = os.path.join(
                self.annocluster_ZINB_folder, f"anno_ZINB_model_optimal_{iteration}.pickle"
            )
            annocluster_z_ZINB_path = os.path.join(
                self.annocluster_ZINB_folder, f"anno_z_ZINB_optimal_{iteration}.npy"
            )
            annocluster_resume_state_path = self._runtime_state_path(
                f"anno_ZINB_resume_state_{iteration}.pickle"
            )
            annocluster_postcluster_state_path = self._runtime_state_path(
                f"anno_ZINB_postcluster_state_{iteration}.pickle"
            )

            model_annocluster = AnnoCluster(
                input_dim=self.G,
                z_dim=self.z_dim,
                gene_set_dim=331,
                tau=self.tau,
                encoder_dim=self.z_encoder_dim,
                emission_dim=self.z_decoder_dim,
                num_layers_encoder=self.z_encoder_layers,
                num_layers_decoder=self.z_decoder_layers,
                use_t_dist=True,
                reconstruction_network="gaussian",
                decoding_network="gaussian",
                centroids_1=centroids_torch_project1,
                centroids_2=centroids_torch_project2,
                use_cuda=self.use_cuda,
            )

            if iteration == 1:
                pretrained_state_dict = model_autoencoder.state_dict()
            else:
                pretrained_state_dict = model_contrastive.state_dict()
            state_dict = model_annocluster.state_dict()
            for key in state_dict.keys():
                if key in pretrained_state_dict:
                    state_dict[key] = pretrained_state_dict[key]
            model_annocluster.load_state_dict(state_dict)

            clusters_pre = np.hstack((clusters_pre1, clusters_pre2))
            contr_expression = ContrDataset(
                self.x,
                clusters_pre,
                subjects,
                raw_array=self.x_raw,
                return_raw_target=self.use_raw_decoder_target,
            )
            trainer = Trainer(
                dataset=contr_expression,
                model=model_annocluster,
                learning_rate=self.alpha,
                model_name="anno_ZINB",
                batch_size=self.batch_size,
                num_epochs=self.anno_epoch,
                save_infer=True,
                output_folder=self.annocluster_ZINB_folder,
                num_workers=self.num_workers,
                use_cuda=self.use_cuda,
                iteration=iteration,
                resume_state_path=annocluster_resume_state_path,
                model_optimal_alias=annocluster_ae_ZINB_path,
                debug_stop_after_epoch=self._debug_stop_epoch("annocluster"),
                debug_stage_name=f"annocluster_iter_{iteration}",
            )
            annocluster_completed = self._resume_completed(trainer.resume_state)

            if os.path.isfile(annocluster_postcluster_state_path):
                self._require_completed_resume(
                    trainer.resume_state,
                    annocluster_resume_state_path,
                    f"Iteration {iteration} annocluster stage",
                )
                print(
                    f"Iteration {iteration}: annocluster stage already prepared. "
                    "Skipping annocluster training."
                )
                (
                    clusters_pre1,
                    centroids_torch_project1,
                    clusters_pre2,
                    centroids_torch_project2,
                ) = self._load_cluster_state(annocluster_postcluster_state_path)
            else:
                if annocluster_completed:
                    print(
                        f"Iteration {iteration}: annocluster training already completed. "
                        "Rebuilding post-annocluster clustering state."
                    )
                    z_init, _clusters = trainer.infer_anno_ZINB()
                    np.save(annocluster_z_ZINB_path, z_init)
                else:
                    if trainer.resume_state is None:
                        print("Start pretraining the annocluster model ... ")
                    else:
                        print(
                            f"Resume pretraining the annocluster model from epoch "
                            f"{trainer.start_epoch} ..."
                        )
                    trainer.train(weight_decay=self.weight_decay)
                    z_init, _clusters = trainer.infer_anno_ZINB()
                    np.save(annocluster_z_ZINB_path, z_init)

                z_init = self._to_numpy(z_init)
                z_init1 = z_init[: self.N_project1]
                z_init2 = z_init[self.N_project1 :]
                clusters_pre1, centroids_torch_project1 = self.leiden_cluster(z_init1)
                clusters_pre2, centroids_torch_project2 = self.leiden_cluster(z_init2)
                self._save_cluster_state(
                    annocluster_postcluster_state_path,
                    clusters_pre1,
                    centroids_torch_project1,
                    clusters_pre2,
                    centroids_torch_project2,
                )

            scaled_centroids_1 = preprocessing.scale(self._to_numpy(centroids_torch_project1))
            scaled_centroids_2 = preprocessing.scale(self._to_numpy(centroids_torch_project2))
            pair_1 = self.get_kl_div_pair(scaled_centroids_1, scaled_centroids_2)
            pair_2 = self.get_kl_div_pair(scaled_centroids_2, scaled_centroids_1)

            constractive_ae_ZINB_path = os.path.join(
                self.model_contrastive_folder, f"constractive_model_optimal_{iteration}.pickle"
            )
            constractive_z_ZINB_path = os.path.join(
                self.model_contrastive_folder, f"constractive_optimal_{iteration}.npy"
            )
            constractive_resume_state_path = self._runtime_state_path(
                f"constractive_resume_state_{iteration}.pickle"
            )
            constractive_postcluster_state_path = self._runtime_state_path(
                f"constractive_postcluster_state_{iteration}.pickle"
            )

            model_contrastive = KLCLR(
                input_dim=self.G,
                z_dim=self.z_dim,
                encoder_dim=self.z_encoder_dim,
                emission_dim=self.z_decoder_dim,
                num_layers_encoder=self.z_encoder_layers,
                num_layers_decoder=self.z_decoder_layers,
                reconstruction_network="gaussian",
                decoding_network="gaussian",
                use_cuda=self.use_cuda,
            )

            annocluster_pretrained_state_dict = model_annocluster.state_dict()
            state_dict_model_contrastive = model_contrastive.state_dict()
            for key in state_dict_model_contrastive.keys():
                if key in annocluster_pretrained_state_dict:
                    state_dict_model_contrastive[key] = annocluster_pretrained_state_dict[key]
            model_contrastive.load_state_dict(state_dict_model_contrastive)

            clusters_pre = np.hstack((clusters_pre1, clusters_pre2))
            contr_expression = ContrDataset(self.x, clusters_pre, subjects)
            trainer_contrastive = Trainer(
                dataset=contr_expression,
                model=model_contrastive,
                learning_rate=self.alpha,
                model_name="constractive",
                batch_size=self.batch_size,
                num_epochs=self.cont_epoch,
                save_infer=True,
                output_folder=self.model_contrastive_folder,
                num_workers=self.num_workers,
                use_cuda=self.use_cuda,
                contrastive_pair_1=pair_1,
                contrastive_pair_2=pair_2,
                iteration=iteration,
                resume_state_path=constractive_resume_state_path,
                model_optimal_alias=constractive_ae_ZINB_path,
                debug_stop_after_epoch=self._debug_stop_epoch("contrastive"),
                debug_stage_name=f"contrastive_iter_{iteration}",
            )
            constractive_completed = self._resume_completed(trainer_contrastive.resume_state)

            if os.path.isfile(constractive_postcluster_state_path):
                self._require_completed_resume(
                    trainer_contrastive.resume_state,
                    constractive_resume_state_path,
                    f"Iteration {iteration} contrastive stage",
                )
                print(
                    f"Iteration {iteration}: contrastive stage already prepared. "
                    "Moving to the next iteration."
                )
                (
                    clusters_pre1,
                    centroids_torch_project1,
                    clusters_pre2,
                    centroids_torch_project2,
                ) = self._load_cluster_state(constractive_postcluster_state_path)
            else:
                if constractive_completed:
                    print(
                        f"Iteration {iteration}: contrastive training already completed. "
                        "Rebuilding post-contrastive clustering state."
                    )
                    z_init = trainer_contrastive.infer_contrastive()
                    np.save(constractive_z_ZINB_path, z_init)
                else:
                    if trainer_contrastive.resume_state is None:
                        print("Start pretraining the KLCLR model ... ")
                    else:
                        print(
                            f"Resume pretraining the KLCLR model from epoch "
                            f"{trainer_contrastive.start_epoch} ..."
                        )
                    trainer_contrastive.train()
                    z_init = trainer_contrastive.infer_contrastive()
                    np.save(constractive_z_ZINB_path, z_init)

                z_init = self._to_numpy(z_init)
                z_init1 = z_init[: self.N_project1]
                z_init2 = z_init[self.N_project1 :]
                clusters_pre1, centroids_torch_project1 = self.leiden_cluster(z_init1)
                clusters_pre2, centroids_torch_project2 = self.leiden_cluster(z_init2)
                self._save_cluster_state(
                    constractive_postcluster_state_path,
                    clusters_pre1,
                    centroids_torch_project1,
                    clusters_pre2,
                    centroids_torch_project2,
                )


if __name__ == "__main__":
    try:
        train = Train()
        train.run()
        print("End")
    except DebugStopRequested as exc:
        print(f"[debug-stop] {exc}")
        raise SystemExit(85)
