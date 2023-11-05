import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

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
            self.decoder_e = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            self.decoder_q_1 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            # self.decoder_e_2 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)
            # self.decoder_q_2 = Decoder_ZINB(z_dim, input_dim, num_layers=num_layers_decoder, hidden_dim=emission_dim)

        elif self.decoding_network == 'geneSet':
            self.decoder_e = Set2Gene(gene_set_table)
            self.decoder_q = Set2Gene(gene_set_table)
        else:
            raise NotImplementedError(f"The current implementation only support 'gaussian', "
                                      f"'geneSet' for emission decoder.")
        self.zinb_loss = ZINBLoss().cuda()
        self.criterion = ZINB().cuda()
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
            e_mean, e_disp, e_pi= self.decoder_e(z_e)
            q_mean, q_disp, q_pi= self.decoder_q_1(z_q)
        else:
            # get the index of embedding closed to the encoding
            k, z_dist, dist_prob = self._get_clusters_2(z_e)

            # get embeddings (discrete representations)
            z_q = self._get_embeddings_2(k)
            e_mean, e_disp, e_pi= self.decoder_e(z_e)
            q_mean, q_disp, q_pi= self.decoder_q_1(z_q)
        # decode embedding (discrete representation) and encoding
        
        return e_mean, e_disp, e_pi, q_mean, q_disp, q_pi, z_e, z_q, k, torch.mean(z_dist), dist_prob

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

    def _loss_reconstruct(self, x, x_e, x_q):
        """
        Calculate reconstruction loss.

        Parameters
        -----------
        x: torch.Tensor
            original observation in full-dimension
        x_e: torch.Tensor
            reconstructed observation encodings
        x_q: torch.Tensor
            reconstructed observation from  embeddings (discrete representations)
        """

        l_e = self.mse_loss(x, x_e)
        l_q = self.mse_loss(x, x_q)
        mse_l = l_e + l_q * self.zeta
        return mse_l



    def loss(self, x_raw_tensor, e_mean, e_disp, e_pi, q_mean, q_disp, q_pi, z_dist):

        l_e = self.criterion(e_pi, e_disp, x_raw_tensor, e_mean)
        l_q = self.criterion(q_pi, q_disp, x_raw_tensor, q_mean)

        # l_e = self.zinb_loss(x=x_raw_tensor, mean=e_mean, disp=e_disp, pi=e_pi)
        # l_q = self.zinb_loss(x=x_raw_tensor, mean=q_mean, disp=q_disp, pi=q_pi)
        l = l_e + l_q * self.zeta + l_dist
        return l, l_e, l_q, l_dist


