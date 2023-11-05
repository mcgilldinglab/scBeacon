import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import sys
from unifan.networks import Encoder, Decoder_ZINB, Set2Gene
from layers import ZINBLoss, MeanAct, DispAct, ZINB

class AnnoCluster(nn.Module):

    """
    Clustering with annotator.

    Parameters
    ----------
    input_dim: integer
        number of input features
    z_dim: integer
        number of low-dimensional features
    gene_set_dim: integer
        number of gene sets
    tau: float
        hyperparameter to weight the annotator loss
    zeta: float
        hyperparameter to weight the reconstruction loss from embeddings (discrete representations)
    encoder_dim: integer
        dimension of hidden layer for encoders
    emission_dim: integer
        dimension of hidden layer for decoders
    num_layers_encoder: integer
        number of hidden layers  for encoder
    num_layers_decoder: integer
        number of hidden layers  for decoder
    dropout_rate: float
    use_t_dist: boolean
        if using t distribution kernel to transform the euclidean distances between encodings and centroids
    regulating_probability: string
        the type of probability to regulating the clustering (by distance) results
    centroids: torch.Tensor
        embeddings in the low-dimensional space for the cluster centroids
    gene_set_table: torch.Tensor
        gene set relationship table

    """

    def __init__(self, input_dim: int = 10000, z_dim: int = 32, gene_set_dim: int = 335,
                 tau: float = 1.0, zeta: float = 1.0, n_clusters: int = 16,
                 encoder_dim: int = 128, emission_dim: int = 128, num_layers_encoder: int = 1,
                 num_layers_decoder: int = 1, dropout_rate: float = 0.1, use_t_dist: bool = True,
                 reconstruction_network: str = "gaussian", decoding_network: str = "gaussian",
                 regulating_probability: str = "classifier", centroids_1: torch.Tensor = None, centroids_2: torch.Tensor = None,
                 gene_set_table: torch.Tensor = None, use_cuda: bool = False):

        super().__init__()

        # initialize parameters
        self.z_dim = z_dim
        self.reconstruction_network = reconstruction_network
        self.decoding_network = decoding_network
        self.tau = tau
        self.zeta = zeta
        self.n_clusters = n_clusters
        self.use_t_dist = use_t_dist
        self.regulating_probability = regulating_probability

        if regulating_probability not in ["classifier"]:
            raise NotImplementedError(f"The current implementation only support 'classifier', "
                                      f" for regulating probability.")

        # initialize centroids embeddings
        if centroids_1 is not None:
            self.embeddings_1 = nn.Parameter(centroids_1, requires_grad=True)
        else:
            print("Warning! the centroids embedding is not activate")
            self.embeddings_1 = nn.Parameter(torch.randn(self.n_clusters, self.z_dim) * 0.05, requires_grad=True)
        if centroids_2 is not None:
            self.embeddings_2 = nn.Parameter(centroids_2, requires_grad=True)
        else:
            print("Warning! the centroids embedding is not activate")
            self.embeddings_2 = nn.Parameter(torch.randn(self.n_clusters, self.z_dim) * 0.05, requires_grad=True)

        # initialize loss
        self.mse_loss = nn.MSELoss()
        self.nLL_loss = nn.NLLLoss()

        # instantiate encoder for z
        if self.reconstruction_network == "gaussian":
            self.encoder = Encoder(input_dim, z_dim, num_layers=num_layers_encoder, hidden_dim=encoder_dim,
                                   dropout_rate=dropout_rate)
        else:
            raise NotImplementedError(f"The current implementation only support 'gaussian' for encoder.")

        # instantiate decoder for emission
        if self.decoding_network == 'gaussian':
            self.decoder_e_1 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            self.decoder_q_1 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            self.decoder_e_2 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            self.decoder_q_2 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)

        elif self.decoding_network == 'geneSet':
            self.decoder_e = Set2Gene(gene_set_table)
            self.decoder_q = Set2Gene(gene_set_table)
        else:
            raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                      f"'geneSet' for emission decoder.")
        self.zinb_loss = ZINBLoss().cuda()
        self.criterion = ZINB().cuda()
        self.bceloss = nn.BCELoss()
        self.proj_head_1 = nn.Sequential(
                nn.Linear(z_dim, input_dim),
                #nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, z_dim),)
                #nn.BatchNorm1d(z_dim))
        self.proj_head_2 = nn.Sequential(
                nn.Linear(z_dim, input_dim),
                #nn.BatchNorm1d(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, z_dim),)
                #nn.BatchNorm1d(z_dim))
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, x, subject):

        # get encoding
        z_e, _ = self.encoder(x)
        if subject == 0:
            # get the index of embedding closed to the encoding
            k, z_dist, dist_prob = self._get_clusters_1(z_e)

            # get embeddings (discrete representations)
            z_q = self._get_embeddings_1(k)
            e_mean, e_disp, e_pi= self.decoder_e_1(z_e)
            q_mean, q_disp, q_pi= self.decoder_q_1(z_q)
            z_c = self.proj_head_1(z_e)
            z_q_c = self.proj_head_1(z_q)
        else:
            # get the index of embedding closed to the encoding
            k, z_dist, dist_prob = self._get_clusters_2(z_e)

            # get embeddings (discrete representations)
            z_q = self._get_embeddings_2(k)
            e_mean, e_disp, e_pi= self.decoder_e_2(z_e)
            q_mean, q_disp, q_pi= self.decoder_q_2(z_q)
            z_c = self.proj_head_2(z_e)
            z_q_c = self.proj_head_2(z_q)
        # decode embedding (discrete representation) and encoding
        
        
        return e_mean, e_disp, e_pi, q_mean, q_disp, q_pi, z_e, z_q, k, torch.mean(z_dist), dist_prob, z_c, z_q_c

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


    def klclr_loss(self, z_e, clusters_pre, contrastive_pair_1, contrastive_pair_2, subject, device, non_blocking):
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


    def loss(self, x_raw_tensor, e_mean, e_disp, e_pi, q_mean, q_disp, q_pi):

        l_e = self.criterion(e_pi, e_disp, x_raw_tensor, e_mean)
        l_q = self.criterion(q_pi, q_disp, x_raw_tensor, q_mean)
        #l_dist = torch.mean(z_dist)
        # l_e = self.zinb_loss(x=x_raw_tensor, mean=e_mean, disp=e_disp, pi=e_pi)
        # l_q = self.zinb_loss(x=x_raw_tensor, mean=q_mean, disp=q_disp, pi=q_pi)
        l = l_e + l_q * self.zeta
        return l, l_e, l_q


