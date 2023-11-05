import torch
import torch.nn as nn
import numpy as np
torch.backends.cudnn.benchmark = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from unifan.networks import Encoder, Decoder_ZINB, Set2Gene, LinearCoder, NonNegativeCoder, SigmoidCoder
from layers import ZINBLoss, MeanAct, DispAct, ZINB
import torch.nn.functional as F
class KLCLR(nn.Module):
    """

    Autoencoder used for pre-training.

    Parameters
    ----------
    input_dim: integer
        number of input features
    z_dim: integer
        number of low-dimensional features
    gene_set_dim: integer
        number of gene sets
    encoder_dim: integer
        dimension of hidden layer for encoders
    emission_dim: integer
        dimension of hidden layer for decoders
    num_layers_encoder: integer
        number of hidden layers  for encoder
    num_layers_decoder: integer
        number of hidden layers  for decoder
    dropout_rate: float
    gene_set_table: torch.Tensor
        gene set relationship table

    """

    def __init__(self, input_dim: int = 10000, z_dim: int = 32, gene_set_dim: int = 335, encoder_dim: int = 128,
                 emission_dim: int = 128, num_layers_encoder: int = 1, num_layers_decoder: int = 1,
                 dropout_rate: float = 0.1, reconstruction_network: str = "non-negative",centroids_1: torch.Tensor = None, centroids_2: torch.Tensor = None,
                 decoding_network: str = "geneSet", gene_set_table: torch.Tensor = None, use_cuda: bool = False):

        super().__init__()

        # initialize parameters
        self.z_dim = z_dim
        self.reconstruction_network = reconstruction_network
        self.decoding_network = decoding_network
        self.use_t_dist = True
        # initialize loss
        self.mse_loss = nn.MSELoss()
        
        # initialize centroids embeddings
        self.embeddings_1 = nn.Parameter(centroids_1, requires_grad=True)
        self.embeddings_2 = nn.Parameter(centroids_2, requires_grad=True)


        # initialize encoder and decoder
        if self.reconstruction_network == 'linear' and self.decoding_network == 'linear':
            self.encoder = LinearCoder(input_dim, z_dim)
            self.decoder_e = LinearCoder(z_dim, input_dim)
        else:

            if self.reconstruction_network == 'non-negative':
                # instantiate encoder for z
                self.encoder = NonNegativeCoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                                dropout_rate=dropout_rate)
            elif self.reconstruction_network == 'sigmoid':
                # instantiate encoder for z
                self.encoder = SigmoidCoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                            dropout_rate=dropout_rate)
            elif self.reconstruction_network == "gaussian":
                # instantiate encoder for z, using standard encoder
                self.encoder = Encoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                       dropout_rate=dropout_rate)


            else:
                raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                          f"'non-negative' or 'sigmoid' for encoder.")

            # instantiate decoder for emission
            if self.decoding_network == 'gaussian':
                self.decoder_e = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)

            elif self.decoding_network == 'geneSet':
                self.decoder_e = Set2Gene(gene_set_table)
            else:
                raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                          f"'geneSet' for emission decoder.")
        self.zinb_loss = ZINBLoss().cuda()
        self.criterion = ZINB().cuda()
        self.bceloss = nn.BCELoss()
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.proj_head = nn.Sequential(
                nn.Linear(z_dim, input_dim),
                #nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, z_dim),)
                #nn.BatchNorm1d(z_dim))
    def forward(self, data, subject):

        x = data
        if x.shape[1] == 64:
            # get encoding
            z_e = x
            z_c = self.proj_head(z_e)

        else:
            # get encoding
            z_e, _ = self.encoder(x)
            z_q = torch.Tensor(z_e.size()).cuda()
            
            # get the index of embedding closed to the encoding
            k, z_dist, dist_prob = self._get_clusters_1(z_e[subject == 0])

            # get embeddings (discrete representations)
            if len(k) > 0:
                z_q_1 = self._get_embeddings_1(k)
            # print(z_q_1.shape)
            # print(z_q.shape)
                z_q[subject == 0] = z_q_1
        
            # get the index of embedding closed to the encoding
            k, z_dist, dist_prob = self._get_clusters_2(z_e[subject == 1])

            # get embeddings (discrete representations)
            if len(k) > 0:
                z_q_2 = self._get_embeddings_2(k)
                z_q[subject == 1] = z_q_2

            z_c = self.proj_head(z_q)


        return  z_e ,z_c

    def _get_clusters_1(self, z_e):
        """

        Assign each sample to a cluster based on euclidean distances.

        Parameters
        ----------
        z_e: torch.Tensor
            low-dimensional encodings

        Returns
        -------
        k: torch.Tensor
            cluster assignments
        z_dist: torch.Tensor
            distances between encodings and centroids
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _z_dist = (z_e.unsqueeze(1) - self.embeddings_1.unsqueeze(0)) ** 2
        z_dist = torch.sum(_z_dist, dim=-1)

        if self.use_t_dist:
            dist_prob = self._t_dist_sim(z_dist, df=10)
            k = torch.argmax(dist_prob, dim=-1)
        else:
            k = torch.argmin(z_dist, dim=-1)
            dist_prob = None

        return k, z_dist, dist_prob
    def _get_clusters_2(self, z_e):
        """

        Assign each sample to a cluster based on euclidean distances.

        Parameters
        ----------
        z_e: torch.Tensor
            low-dimensional encodings

        Returns
        -------
        k: torch.Tensor
            cluster assignments
        z_dist: torch.Tensor
            distances between encodings and centroids
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _z_dist = (z_e.unsqueeze(1) - self.embeddings_2.unsqueeze(0)) ** 2
        z_dist = torch.sum(_z_dist, dim=-1)

        if self.use_t_dist:
            dist_prob = self._t_dist_sim(z_dist, df=10)
            k = torch.argmax(dist_prob, dim=-1)
        else:
            k = torch.argmin(z_dist, dim=-1)
            dist_prob = None

        return k, z_dist, dist_prob
    def _t_dist_sim(self, z_dist, df=10):
        """
        Transform distances using t-distribution kernel.

        Parameters
        ----------
        z_dist: torch.Tensor
            distances between encodings and centroids

        Returns
        -------
        dist_prob: torch.Tensor
            probability of closeness of encodings to centroids transformed by t-distribution

        """

        _factor = - ((df + 1) / 2)
        dist_prob = torch.pow((1 + z_dist / df), _factor)
        dist_prob = dist_prob / dist_prob.sum(axis=1).unsqueeze(1)

        return dist_prob

    def _get_embeddings_1(self, k):
        """

        Get the embeddings (discrete representations).

        Parameters
        ----------
        k: torch.Tensor
            cluster assignments

        Returns
        -------
        z_q: torch.Tensor
            low-dimensional embeddings (discrete representations)

        """

        k = k.long()
        _z_q = []
        for i in range(len(k)):
            _z_q.append(self.embeddings_1[k[i]])

        z_q = torch.stack(_z_q)

        return z_q
    def _get_embeddings_2(self, k):
        """

        Get the embeddings (discrete representations).

        Parameters
        ----------
        k: torch.Tensor
            cluster assignments

        Returns
        -------
        z_q: torch.Tensor
            low-dimensional embeddings (discrete representations)

        """

        k = k.long()
        _z_q = []
        for i in range(len(k)):
            _z_q.append(self.embeddings_2[k[i]])

        z_q = torch.stack(_z_q)

        return z_q

    def _loss_reconstruct(self, x, x_e):
        """
        Calculate reconstruction loss.

        Parameters
        ----------
        x: torch.Tensor
            original data
        x_e: torch.Tensor
            reconstructed data

        Returns
        -------
        mse_l: torch.Tensor
            reconstruction loss

        """
        l_e = self.mse_loss(x, x_e)
        mse_l = l_e

        return mse_l

    def loss(self, z_e, clusters_pre, contrastive_pair_1, contrastive_pair_2, subject, device, non_blocking):
        l1 = 0.

        for i in range(0,len(z_e)):
            z_1 = z_e[i]
            
            z_2 = torch.cat([z_e[:i], z_e[i+1:]],dim=0)
            sub_1 = subject[i]
            sub_2 = torch.cat([subject[:i], subject[i+1:]], dim=0)
            pre_1 = clusters_pre[i].int().item()
            pre_2 = torch.cat([clusters_pre[:i], clusters_pre[i+1:]], dim=0).int()
            loss_1 = 0.
            loss_2 = 0.
            if sub_1 == 0:
                ## compare subjects 0 and 0       
                ind_1 = np.where(sub_2 == 0)
                if len(ind_1) is not 0:
                    similarity_score_1 = F.cosine_similarity(z_1, z_2[ind_1], dim=-1)
                    similarity_score_1 = (similarity_score_1 + 1) / 2
                    similarity_score_1 = torch.clamp(similarity_score_1,0,1)
                    label_1 = torch.tensor(pre_2[ind_1]==pre_1).to(device).float()
                    #loss_1 = self.bceloss(similarity_score_1.float(),label_1)
                    try:
                        backup_similarity_score, backup_label = similarity_score_1.float().cpu(), label_1.cpu()
                        loss_1 = self.bceloss(similarity_score_1,label_1)
                    except Exception as e :
                        print(1)
                        print(backup_similarity_score,backup_label)
                        print(backup_similarity_score.shape,backup_label.shape)
                        #print(torch.isnan(backup_similarity_score),torch.isnan(backup_label))
                        print(e)
                        sys.exit()
                ## compare subjects 0 and 1
                ind_2 = np.where(sub_2 == 1)
                if len(ind_2) is not 0:
                    # get the corresponding cluster based on pairs
                    try:
                        j = contrastive_pair_1[pre_1]
                    except:
                        print(2)
                        print(pre_1)
                        print(contrastive_pair_1)
                        sys.exit()
                    similarity_score_2 = F.cosine_similarity(z_1, z_2[ind_2], dim=-1)
                    similarity_score_2 = (similarity_score_2 + 1) / 2
                    similarity_score_2 = torch.clamp(similarity_score_2,0,1)
                    label_2 = torch.tensor(j==pre_2[ind_2]).to(device).float()
                    #loss_2 = self.bceloss(similarity_score_2,label_2)
                
                #print(similarity_score_2, label_2)
                    try :
                        backup_similarity_score, backup_label = similarity_score_2.float().cpu(), label_2.cpu()
                        loss_2 = self.bceloss(similarity_score_2,label_2)
                    except Exception as e :
                        print(3)
                        print(backup_similarity_score,backup_label)
                        print(backup_similarity_score.shape,backup_label.shape)
                        #print(torch.isnan(backup_similarity_score),torch.isnan(backup_label))
                        print(e)
                        sys.exit()

            elif sub_1 == 1:
                ind_1 = np.where(sub_2 == 1)
                if len(ind_1) is not 0:
                    similarity_score_1 = F.cosine_similarity(z_1, z_2[ind_1], dim=-1)
                    similarity_score_1 = (similarity_score_1 + 1) / 2
                    similarity_score_1 = torch.clamp(similarity_score_1,0,1)
                    label_1 = torch.tensor(pre_2[ind_1]==pre_1).to(device).float()
                    #loss_1 = self.bceloss(similarity_score_1.float(),label_1)
                    try:
                        backup_similarity_score, backup_label = similarity_score_1.float().cpu(), label_1.cpu()
                        loss_1 = self.bceloss(similarity_score_1,label_1)
                    except Exception as e :
                        print(4)
                        print(backup_similarity_score,backup_label)
                        print(backup_similarity_score.shape,backup_label.shape)
                        #print(torch.isnan(backup_similarity_score),torch.isnan(backup_label))
                        print(e)
                        sys.exit()
                ind_2 = np.where(sub_2 == 0)
                if len(ind_2) is not 0:
                    try:
                        j = contrastive_pair_2[pre_1]
                    except:
                        print(5)
                        print(pre_1)
                        print(contrastive_pair_2)
                        sys.exit()
                    similarity_score_2 = F.cosine_similarity(z_1, z_2[ind_2], dim=-1)
                    similarity_score_2 = (similarity_score_2 + 1) / 2
                    similarity_score_2 = torch.clamp(similarity_score_2,0,1)
                    label_2 = torch.tensor(j==pre_2[ind_2]).to(device).float()
                    #loss_2 = self.bceloss(similarity_score_2.float(),label_2)
                #print(similarity_score_2, label_2)
                    try :
                        backup_similarity_score, backup_label = similarity_score_2.float().cpu(), label_2.cpu()
                        loss_2 = self.bceloss(similarity_score_2,label_2)
                    except Exception as e :
                        print(6)
                        print(backup_similarity_score,backup_label)
                        print(backup_similarity_score.shape,backup_label.shape)
                        #print(torch.isnan(backup_similarity_score),torch.isnan(backup_label))
                        print(e)
                        sys.exit()



            loss = loss_1 + loss_2
            l1 = l1 + loss
        return l1/len(z_e)
