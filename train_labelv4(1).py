import os
import sys
import gc
import itertools
from config_label import Config
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import umap
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import torch.nn.functional as F
import anndata as ad
from scipy.sparse import csc_matrix
# expression_only1.data
# expression_only1.data.concatenate(expression_only2.data)
# expression=[expression_only1.data,expression_only2.data]
import scanpy as sc
# from context import unifan 
sys.path.append(r"/data/liuchenyu/m")
from unifan.datasets1 import AnnDataset, ContrDataset

from unifan.annoclusterV4 import AnnoCluster
from unifan.autoencoderV2 import autoencoder
from unifan.KLCLRv4 import KLCLR
from unifan.classifier import classifier
from unifan.utils import getGeneSetMatrix
from unifan.trainer import Trainer






class Train(Config):
    def get_consine_pair(self, c_init_1, c_init_2):
        c_init_1 = torch.Tensor(c_init_1)
        c_init_2 = torch.Tensor(c_init_2)

        pair_1 = {}

        for i in range(0,c_init_1.shape[0]):
            score_max = -2
            for j in range(0,c_init_2.shape[0]):
        #         _z_dist = (centroids_torch_project2[j] - centroids_torch_project1[i]) ** 2
                similarity_score = F.cosine_similarity(c_init_1[i], c_init_2[j], dim=-1)

                if similarity_score > score_max:
                    score_max = similarity_score       
                    inx = j
                
            pair_1[i] = inx
        return pair_1

    def get_kl_div_pair(self, centroids_torch_project1, centroids_torch_project2):
        pair_1 = {}
        centroids_torch_project1 = torch.Tensor(centroids_torch_project1)
        centroids_torch_project2 = torch.Tensor(centroids_torch_project2)
        for i in range(0,centroids_torch_project1.shape[0]):
            z_dist_min = 9999
            for j in range(0,centroids_torch_project2.shape[0]):
        #         _z_dist = (centroids_torch_project2[j] - centroids_torch_project1[i]) ** 2
                z_dist = F.kl_div(centroids_torch_project1[i].softmax(dim=-1).log(), centroids_torch_project2[j].softmax(dim=-1), reduction='sum')
        #         z_dist = torch.sum(_z_dist, dim=-1)
                #@print(z_dist)
                if z_dist < z_dist_min:
                    z_dist_min = z_dist       
                    inx = j
           

            pair_1[i] = inx
        return pair_1

    def leiden_cluster(self, z_init, clusters_true):
        # initialize using leiden clustering
        adata = sc.AnnData(X=z_init)
        adata.obsm['X_unifan'] = z_init
        sc.pp.neighbors(adata, n_pcs=self.z_dim,  use_rep='X_unifan', random_state=self.random_seed)

        best_ari = 0
        best_resolution = 0
        for resolution in range (1,200):
            resolution = resolution / 1000
            sc.tl.leiden(adata, resolution=resolution)
            clusters_pre = adata.obs['leiden'].astype('int').values
            ari_smaller = adjusted_rand_score(clusters_true,
                                            clusters_pre)
            if ari_smaller>best_ari:
                best_ari = ari_smaller
                best_resolution = resolution


      

        sc.tl.leiden(adata, resolution=best_resolution, random_state=self.random_seed)
        clusters_pre = adata.obs['leiden'].astype('int').values  # original as string

        # initialize centroids
        try:
            df_cluster = pd.DataFrame(z_init.detach().cpu().numpy())
        except AttributeError:
            df_cluster = pd.DataFrame(z_init)

        cluster_labels = np.unique(clusters_pre)
        M = len(set(cluster_labels))  # set as number of clusters
        df_cluster['cluster'] = clusters_pre

        # get centroids
        centroids = df_cluster.groupby('cluster').mean().values
        centroids_torch = torch.from_numpy(centroids)
        return clusters_pre, centroids_torch

    def run(self):
        sex_mapping = {"Lymphoid": 0, "Myeloid": 1,"Multiplet": 2,"Stromal": 3,"Endothelial": 4,"Epithelial": 5,}
        clusters_true1 = self.adata1.obs['CellType_Category']
        clusters_true1 = clusters_true1.map(sex_mapping)
        clusters_true2 = self.adata2.obs['CellType_Category']
        clusters_true2 = clusters_true2.map(sex_mapping)
        model_autoencoder = autoencoder(input_dim=self.G, z_dim=self.z_dim, 
                                        encoder_dim=self.z_encoder_dim, emission_dim=self.z_decoder_dim,
                                        num_layers_encoder=self.z_encoder_layers, num_layers_decoder=self.z_decoder_layers,
                                        reconstruction_network='gaussian', decoding_network='gaussian',
                                        use_cuda=self.use_cuda)
        
        # ------ Pretrain autoencoder ------
        
        
        if os.path.isfile(self.input_z_ZINB_path) and os.path.isfile(self.input_ae_ZINB_path):
            print(f"Both pretrained autoencoder and inferred z exist. No need to pretrain the autoencoder model.")
            z_init = np.load(self.input_z_ZINB_path)
            model_autoencoder.load_state_dict(torch.load(self.input_ae_ZINB_path, map_location=self.device)['state_dict'])
        else:
           
            trainer = Trainer(dataset = self.expression_integrated, model=model_autoencoder,learning_rate=self.alpha, model_name="pretrain_z_ZINB", batch_size=self.batch_size,
                            num_epochs=self.pretrain_epoch, save_infer=True, output_folder=self.pretrain_z_folder, num_workers=0,
                            use_cuda=self.use_cuda)

            if os.path.isfile(self.input_ae_ZINB_path):
                print(f"Only pretrained autoencoder exists. Need to infer z.")
                model_autoencoder.load_state_dict(torch.load(self.input_ae_ZINB_path, map_location=self.device)['state_dict'])
                z_init = trainer.infer_z_ZINB()
                np.save(self.input_z_ZINB_path, z_init)
            else:
                print(f"Start pretraining the autoencoder model ... ")
                trainer.train()
                model_autoencoder.load_state_dict(torch.load(self.input_ae_ZINB_path, map_location=self.device)['state_dict'])
                z_init = np.load(self.input_z_ZINB_path)
        
        # ------ Find the Cluster ------
        try:
            z_init = z_init.numpy()
        except AttributeError:
            pass
        z_init1 = z_init[:self.N_project1]
        z_init2 = z_init[self.N_project1:]

        clusters_pre1, centroids_torch_project1 = self.leiden_cluster(z_init1, clusters_true1)
        clusters_pre2, centroids_torch_project2 = self.leiden_cluster(z_init2, clusters_true2)

        # ------ Define the Contrastive dataset ------
        subjects = np.hstack((np.zeros(self.N_project1), np.ones(self.N_project1)))
        clusters_pre = np.hstack((clusters_pre1, clusters_pre2))
        Contr_expression = ContrDataset(self.x, clusters_pre, subjects)   

        model_contrastive = KLCLR(input_dim=self.G, z_dim=self.z_dim, 
                                            encoder_dim=self.z_encoder_dim, emission_dim=self.z_decoder_dim,
                                            num_layers_encoder=self.z_encoder_layers, num_layers_decoder=self.z_decoder_layers,
                                            reconstruction_network='gaussian', decoding_network='gaussian', 
                                            use_cuda=self.use_cuda)
        for iteration in range(1, self.num_iteration+1):
            annocluster_ae_ZINB_path = os.path.join(self.annocluster_ZINB_folder, f"anno_ZINB_model_optimal.pickle")
            annocluster_z_ZINB_path = os.path.join(self.annocluster_ZINB_folder, f"anno_z_ZINB_optimal_{iteration}.npy")
            # ------ Train the anno model ------
            
            use_pretrain = True

            model_annocluster = AnnoCluster(input_dim=self.G, z_dim=self.z_dim, gene_set_dim=331, tau=self.tau,
                                            encoder_dim=self.z_encoder_dim, emission_dim=self.z_decoder_dim, 
                                            num_layers_encoder=self.z_encoder_layers, num_layers_decoder=self.z_decoder_layers, 
                                            use_t_dist=True, reconstruction_network='gaussian', decoding_network='gaussian', 
                                            centroids_1=centroids_torch_project1, centroids_2=centroids_torch_project2,use_cuda=self.use_cuda)

            # reload dataset, loading gene set activity scores together
            if os.path.isfile(annocluster_ae_ZINB_path) and False:
                model_annocluster.load_state_dict(torch.load(annocluster_ae_ZINB_path, map_location=self.device)['state_dict'])
            else:
                if use_pretrain:
                    if iteration == 1:
                        pretrained_state_dict = model_autoencoder.state_dict()
                    else:
                        pretrained_state_dict = model_contrastive.state_dict()
                    # load pretrained AnnoCluster model
                    state_dict = model_annocluster.state_dict()
                    for k, v in state_dict.items():
                        if k in pretrained_state_dict.keys():
                            state_dict[k] = pretrained_state_dict[k]
                        if (k[:8] + 'e' + k[11:]) in pretrained_state_dict.keys():
                                state_dict[k] = pretrained_state_dict[k[:8] + 'e' + k[11:]]

                    model_annocluster.load_state_dict(state_dict)
                
            trainer = Trainer(dataset=Contr_expression, model=model_annocluster, learning_rate=self.alpha,
                            model_name="anno_ZINB", batch_size=self.batch_size, num_epochs=self.anno_epoch,
                            save_infer=True, output_folder=self.annocluster_ZINB_folder, num_workers=0, use_cuda=self.use_cuda, iteration = iteration)
            
            if os.path.isfile(annocluster_ae_ZINB_path) and False:
                print(f"Only pretrained annocluster exists. Need to infer z and no need to pretrain the annocluster model.")
                z_init,_clusters = trainer.infer_anno_ZINB()
                np.save(annocluster_z_ZINB_path, z_init)
            else:
                print(f"Start pretraining the annocluster model ... ")
                trainer.train(weight_decay=self.weight_decay)
                model_annocluster.load_state_dict(torch.load(annocluster_ae_ZINB_path, map_location=self.device)['state_dict'])
                z_init,_clusters = trainer.infer_anno_ZINB()
                np.save(annocluster_z_ZINB_path, z_init)

    
            try:
                z_init = z_init.numpy()
            except AttributeError:
                pass
            z_init1 = z_init[:self.N_project1]
            z_init2 = z_init[self.N_project1:]
            clusters_pre1 = _clusters[:self.N_project1]
            clusters_pre2 = _clusters[self.N_project1:]
            # clusters_pre1, centroids_torch_project1 = self.leiden_cluster(z_init1, clusters_true1)
            # clusters_pre2, centroids_torch_project2 = self.leiden_cluster(z_init2, clusters_true2)

            
            # centroids_torch_project1 = preprocessing.scale(centroids_torch_project1)
            # centroids_torch_project2 = preprocessing.scale(centroids_torch_project2)
            centroids_torch_project1 = model_annocluster.embeddings_1.clone().detach()
            centroids_torch_project2 = model_annocluster.embeddings_2.clone().detach()
            #if iteration == 1:
            pair_1 = self.get_kl_div_pair(centroids_torch_project1, centroids_torch_project2)
            pair_2 = self.get_kl_div_pair(centroids_torch_project2, centroids_torch_project1)

            # ------ Train the KLCLR model ------
            clusters_pre = np.hstack((clusters_pre1, clusters_pre2))
            Contr_expression = ContrDataset(self.x, clusters_pre, subjects)   
            use_pretrain = True
            constractive_ae_ZINB_path = os.path.join(self.model_contrastive_folder, f"constractive_model_optimal.pickle")
            constractive_z_ZINB_path = os.path.join(self.model_contrastive_folder, f"constractive_optimal_{iteration}.npy")
     

            annocluster_pretrained_state_dict = model_annocluster.state_dict()

            # load pretrained AnnoCluster model
            state_dict_model_contrastive = model_contrastive.state_dict()
            for k, v in state_dict_model_contrastive.items():
                if k in annocluster_pretrained_state_dict.keys():
                    state_dict_model_contrastive[k] = annocluster_pretrained_state_dict[k]

            model_contrastive.load_state_dict(state_dict_model_contrastive)

            # reload dataset, loading gene set activity scores together
            if os.path.isfile(constractive_ae_ZINB_path) and False:
                model_contrastive.load_state_dict(torch.load(constractive_ae_ZINB_path, map_location=self.device)['state_dict'])

            trainer_contrastive = Trainer(dataset=Contr_expression, model=model_contrastive, learning_rate=self.alpha,
                            model_name="constractive", batch_size=self.batch_size, num_epochs=self.cont_epoch,
                            save_infer=True, output_folder=self.model_contrastive_folder, num_workers=0, use_cuda=self.use_cuda, 
                                contrastive_pair_1=pair_1, contrastive_pair_2=pair_2, iteration = iteration)

            if os.path.isfile(constractive_ae_ZINB_path) and False:
                print(f"Only pretrained KLCLR exists. Need to infer z and no need to pretrain the KLCLR model.")
                z_init = trainer_contrastive.infer_contrastive()
                np.save(constractive_z_ZINB_path, z_init)
            else:
                print(f"Start pretraining the KLCLR model ... ")
                trainer_contrastive.train()
                model_contrastive.load_state_dict(torch.load(constractive_ae_ZINB_path, map_location=self.device)['state_dict'])
                z_init = trainer_contrastive.infer_contrastive()

            # ------ Find the Cluster ------
            try:
                z_init = z_init.numpy()
            except AttributeError:
                pass
            z_init1 = z_init[:self.N_project1]
            z_init2 = z_init[self.N_project1:]

            clusters_pre1, centroids_torch_project1 = self.leiden_cluster(z_init1, clusters_true1)
            clusters_pre2, centroids_torch_project2 = self.leiden_cluster(z_init2, clusters_true2)


            # pair_1 = self.get_kl_div_pair(centroids_torch_project1, centroids_torch_project2)
            # pair_2 = self.get_kl_div_pair(centroids_torch_project2, centroids_torch_project1)


if __name__ == "__main__":

    train = Train()
    train.run()
    print("End")