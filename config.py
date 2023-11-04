"""
Configuration file
"""

import importlib
import numpy as np
import torch
import scanpy as sc
import os
import sys
from scipy.sparse import csc_matrix
sys.path.append(r"/data/liuchenyu/mcgill")
from unifan.datasets1 import AnnDataset, ContrDataset
class Config(object):
    def __init__(self,):
        # Define the project name for each steps
        # 1. pretrain
        # Loop:
        # 2. anno
        # 3. cont
        self.cont_project = "Contrast_GSE139324"
        self.anno_project = "Anno_GSE139324"
        self.pre_project = "Pretrain_GSE139324"


        # Define the data path.
        self.data_filepath1 = f"../GSE139324_HD.h5ad"
        self.data_filepath2 = f"../GSE139324_HNSSC.h5ad"

        # Define the output path
        self.output_path = "../example/output/"

        self.gene_sets_path = "../gene_sets/"

        self.use_cuda = True
        self.num_workers = 1

        # ------- no need to change the following -------
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.pin_memory = True
            self.non_blocking = True
        else:
            self.pin_memory = False
            self.non_blocking = False
        self.random_seed = 0

        # HyperParameter
        self.prior_name = "c5.go.bp.v7.4.symbols.gmt+c2.cp.v7.4.symbols.gmt+TF-DNA"
        self.features_type = "gene_gene_sets"
        self.alpha = 1e-5
        self.beta = 1e-5
        self.weight_decay = 1e-3
        self.tau = 10
        self.z_dim = 128
        self.batch_size = 512
        self.num_iteration = 10
        self.num_epochs_z = 20

        self.pretrain_epoch = 20
        self.anno_epoch = 20
        self.cont_epoch = 20

    
        self.r_encoder_layers = 5
        self.r_decoder_layers = 1  # need to fixed at 1 if using gene set matrix for decoder 
        self.r_encoder_dim = 128
        self.r_decoder_dim = 128  # actually not used if using gene set matrix for decoder 
        self.rnetwork = 'non-negative'

        self.z_encoder_layers = 3
        self.z_decoder_layers = 2
        self.z_encoder_dim = self.z_dim * 4
        self.z_decoder_dim = self.z_dim * 4

        # ------ prepare for output
        # ### Pretrain
        self.pretrain_output_parent_path = f"{self.output_path}{self.pre_project}/"
        self.pretrain_z_folder = f"{self.pretrain_output_parent_path}pretrain_z"
        self.input_z_ZINB_path = os.path.join(self.pretrain_z_folder, f"pretrain_z_ZINB_optimal_1.npy")
        self.input_ae_ZINB_path = os.path.join(self.pretrain_z_folder, f"pretrain_z_ZINB_model_optimal_1.pickle")

        # ### Anno
        self.anno_output_parent_path = f"{self.output_path}{self.anno_project}/"
        self.annocluster_ZINB_folder = f"{self.anno_output_parent_path}annocluster_ZINB_{self.features_type}"

        # ### Contrastive
        self.contrastive_output_parent_path = f"{self.output_path}{self.cont_project}/"
        self.model_contrastive_folder = f"{self.contrastive_output_parent_path}contrastive_{self.features_type}"
        self.input_constractive_cluster_path = os.path.join(self.model_contrastive_folder, f"cluster_project1.npy")
        self.input_constractive_cluster_path = os.path.join(self.model_contrastive_folder, f"cluster_project2.npy")
        self.input_constractive_centroids_path = os.path.join(self.model_contrastive_folder, f"centroids_project1.npy")
        self.input_constractive_centroids_path = os.path.join(self.model_contrastive_folder, f"centroids_project2.npy")

        # generate dataset
        self.data1 = sc.read(self.data_filepath1, dtype='float64', backed="r")
        self.data2 = sc.read(self.data_filepath2, dtype='float64', backed="r")
        self.x1=csc_matrix(self.data1.X[:]).toarray()
        self.x2=csc_matrix(self.data2.X[:]).toarray()
        self.x = np.vstack((self.x1, self.x2))
        self.N_project1 = self.data1.n_obs 
        self.N_project2 = self.data2.n_obs 
        self.N = self.N_project1 + self.N_project2
        self.G = self.data1.n_vars
        self.expression_integrated = AnnDataset(self.x)