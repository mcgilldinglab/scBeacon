import os
import gc
import pickle

from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import umap
from layers import ZINBLoss,weightedZINBLoss
import random
torch.backends.cudnn.benchmark = True


class Trainer(nn.Module):
    """

    Train NN models.


    Parameters
    ----------
    dataset: PyTorch Dataset
    model: Pytorch NN model
    model_2nd: another Pytorch NN model to be trained together
    model_name: string
        name of the model, any of "r", "pretrain_z", "annocluster", "pretrain_annotator"
    percent_training: float
        percentage of data used for training, the rest used for validation
    checkpoint_freq: integer
        frequency of saving models during training
    val_freq: integer
        frequency of conducting evaluation (both training and validation set will be evaluated)
    visualize_freq: integer
        frequency of inferring low-dimensional representation and visualize it using UMAP
    save_visual: boolean
        if conduct visualization and save the figures to the output folder
    save_checkpoint: boolean
        if saving checkpoint models during training
    save_infer: boolean
        if conduct inference (low-dimensional representation for autoencoders; clusters for clustering model;
        predicted results for classifiers) when the training finished
    output_folder: string
        folder to save all outputs

    """
    def __init__(self, dataset, model, model_2nd=None, model_name: str = None, batch_size: int = 128,
                 num_epochs: int = 50, percent_training: float = 1.0, learning_rate: float = 0.0005,
                 decay_factor: float = 0.5, num_workers: int = 8, use_cuda: bool = False, checkpoint_freq: int = 20,
                 val_freq: int = 10, visualize_freq: int = 10, save_visual: bool = False, save_checkpoint: bool = False,
                 save_infer: bool = False, output_folder: str = None, second_data=None, contrastive_pair_1=None, contrastive_pair_2=None, iteration=1):

        super().__init__()

        # device
        self.num_workers = num_workers
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.pin_memory = True if use_cuda else False
        self.non_blocking = True if use_cuda else False

        # model
        _support_models = ["r", "pretrain_z", "annocluster", "pretrain_annotator","anno","pretrain_z_ZINB","anno_ZINB","constractive"]
        if model_name not in _support_models:
            raise NotImplementedError(f"The current implementation only support training "
                                      f"for {','.join(_support_models)}.")
        self.model_name = model_name
        self.model = model.to(self.device)
        self.model_2nd = model_2nd.to(self.device) if model_2nd is not None else None

        # optimization
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, self.decay_factor)
        if model_2nd is not None:
            self.optimizer_2nd = torch.optim.Adam(self.model_2nd.parameters(), lr=self.learning_rate)
            self.scheduler_2nd = torch.optim.lr_scheduler.StepLR(self.optimizer_2nd, 5, self.decay_factor)
        else:
            self.optimizer_2nd, self.scheduler_2nd = None, None
        self.iteration = iteration
        # evaluation
        self.checkpoint_freq = checkpoint_freq
        self.val_freq = val_freq
        self.visual_frequency = visualize_freq
        self.output_folder = output_folder
        self.percent_training = percent_training
        self.save_checkpoint = save_checkpoint
        self.save_visual = save_visual
        self.save_infer = save_infer

        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # data
        self.dataset = dataset
        self.second_data = second_data
        self.contrastive_pair_1 = contrastive_pair_1
        self.contrastive_pair_2 = contrastive_pair_2
        # prepare for data loader
        train_length = int(self.dataset.N * self.percent_training)
        val_length = self.dataset.N - train_length

        train_data, val_data = torch.utils.data.random_split(self.dataset, (train_length, val_length))
        self.dataloader_all = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.percent_training != 1:
            self.dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True,
                                                         num_workers=self.num_workers, pin_memory=self.pin_memory)
        else:
            self.dataloader_val = None


        if self.second_data is not None:
            second_train_length = int(self.second_data.N * self.percent_training)
            second_val_length = self.second_data.N - second_train_length

            second_train_data, second_val_data = torch.utils.data.random_split(self.second_data, (second_train_length, second_val_length))
            self.second_dataloader_all = torch.utils.data.DataLoader(self.second_data, batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.num_workers, pin_memory=self.pin_memory)
            self.second_dataloader_train = torch.utils.data.DataLoader(second_train_data, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.num_workers, pin_memory=self.pin_memory)
            if self.percent_training != 1:
                self.second_dataloader_val = torch.utils.data.DataLoader(second_val_data, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=self.num_workers, pin_memory=self.pin_memory)
            else:
                self.second_dataloader_val = None

        # initialize functions for each model
        self.train_functions = {"r": self.train_epoch_r, "pretrain_z": self.train_epoch_z,
                                       "pretrain_annotator": self.train_epoch_annotator,
                                       "annocluster": self.train_epoch_cluster,
                                       "anno": self.train_epoch_anno,
                                       "pretrain_z_ZINB": self.train_epoch_z_ZINB,
                                       "anno_ZINB": self.train_epoch_anno_ZINB,
                                       "constractive": self.train_epoch_contrastive}
        self.train_stats_dicts = {"r": {k: [] for k in ['train_loss', 'val_loss', 'train_sparsity', 'train_set_cover']},
                                  "pretrain_z": {k: [] for k in ['train_loss', 'val_loss']},
                                  "pretrain_annotator": {k: [] for k in ['train_loss', 'val_loss']},
                                  "annocluster": {k: [] for k in ['train_loss', 'val_loss', 'train_mse', 'train_mse_e',
                                                                  'train_mse_q', 'train_prob_z_l', 'ARI', 'NMI']},
                                  "anno": {k: [] for k in ['train_loss', 'val_loss', 'train_mse', 'train_mse_e',
                                                                  'train_mse_q', 'ARI', 'NMI']},
                                  "pretrain_z_ZINB": {k: [] for k in ['train_loss', 'val_loss']},
                                  "anno_ZINB": {k: [] for k in ['train_loss', 'val_loss', 'train_mse', 'train_mse_e',
                                                                  'train_mse_q', 'ARI', 'NMI']},
                                  "constractive": {k: [] for k in ['train_loss', 'val_loss']}}

        self.evaluate_functions = {"r": self.evaluate_r, "pretrain_z": self.evaluate_z,
                                "pretrain_annotator": self.evaluate_annotator,
                                "annocluster": self.evaluate_cluster,
                                "anno": self.evaluate_anno,
                                "pretrain_z_ZINB": self.evaluate_z_ZINB,
                                "anno_ZINB": self.evaluate_anno_ZINB,
                                "constractive": self.evaluate_contrastive}

        self.infer_functions = {"r": self.infer_r, "pretrain_z": self.infer_z,
                                   "pretrain_annotator": self.infer_annotator,
                                   "annocluster": self.infer_cluster,
                                   "anno": self.infer_anno,
                                   "pretrain_z_ZINB": self.infer_z_ZINB,
                                   "anno_ZINB": self.infer_anno_ZINB,
                                   "constractive": self.infer_contrastive}

        # initialize the parameter list (used for regularization)
        if self.model_name in ["pretrain_annotator", "annocluster"]:
            self.annotator_param_list = nn.ParameterList()
            if self.model_name == "pretrain_annotator":
                for p in self.model.named_parameters():
                    self.annotator_param_list.append(p[1])
            else:
                assert self.model_2nd is not None
                for p in self.model_2nd.named_parameters():
                    self.annotator_param_list.append(p[1])

        if self.model_name == "pretrain_annotator" and self.save_visual:
            raise NotImplementedError(f"The current implementation only support visualizing "
                                      f"for {','.join(_support_models)[:-1]}.")

        if self.model_name != "annocluster" and self.model_2nd is not None:
            raise NotImplementedError(f"The current implementation only support two models training for annocluster")
        self.zinb_loss = ZINBLoss().cuda()

    def train(self, **kwargs):

        """
        Train the model. Will save the trained model & evaluation results as default.

        Parameters
        ----------
        kwargs: keyword arguements specific to each model (e.g. alpha for gene set activity scores model)

        """
        optimal_loss = float("inf")
        for epoch in range(self.num_epochs):
            self.model.train()
            if self.model_2nd is not None:
                self.model_2nd.train()

            loss = self.train_functions[self.model_name](**kwargs)

            if epoch % self.val_freq == 0:
                with torch.no_grad():
                    self.model.eval()
                    if self.model_2nd is not None:
                        self.model_2nd.eval()

                    self.evaluate_functions[self.model_name](**kwargs)

            if self.save_visual and epoch % self.visual_frequency == 0:
                with torch.no_grad():
                    self.model.eval()
                    if self.model_2nd is not None:
                        self.model_2nd.eval()
                    if self.model_name == "annocluster" or self.model_name == "anno" or self.model_name == "anno_ZINB":
                        _X, _clusters = self.infer_functions[self.model_name](**kwargs)
                    else:
                        _X = self.infer_functions[self.model_name](**kwargs)
                        _clusters = None
                    self.visualize_UMAP(_X, epoch, self.output_folder, clusters_true=self.dataset.clusters_true,
                                        clusters_pre=_clusters)

            # save model & inference
            if optimal_loss > loss:
                # save model
                _state = {'epoch': epoch,
                          'state_dict': self.model.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
                if self.model_2nd is not None:
                    _state = {'epoch': epoch,
                              'state_dict': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict(),
                              'state_dict_2': self.model_2nd.state_dict(),
                              'optimizer_2': self.optimizer_2nd.state_dict()}

                # default (no epoch identifier) model are trained with 20 epochs
                torch.save(_state, os.path.join(self.output_folder, f"{self.model_name}_model_optimal_{self.iteration}.pickle"))
                optimal_loss = loss
                del _state
                gc.collect()

        # save model
        _state = {'epoch': epoch,
                  'state_dict': self.model.state_dict(),
                  'optimizer': self.optimizer.state_dict()}
        if self.model_2nd is not None:
            _state = {'epoch': epoch,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'state_dict_2': self.model_2nd.state_dict(),
                      'optimizer_2': self.optimizer_2nd.state_dict()}

        # default (no epoch identifier) model are trained with 20 epochs
        torch.save(_state, os.path.join(self.output_folder, f"{self.model_name}_model_{epoch}_{self.iteration}.pickle"))

        del _state
        gc.collect()

        # inference and save
        if self.save_infer:
            if self.model_name == "annocluster" or self.model_name == "anno" or self.model_name == "anno_ZINB":
                _X, _clusters = self.infer_functions[self.model_name](**kwargs)
                np.save(os.path.join(self.output_folder, f"{self.model_name}_clusters_pre_{epoch}_{self.iteration}.npy"), _clusters)
            # elif self.model_name == "anno_ZINB":
            #     _X, _clusters, _X_2, _clusters_2 = self.infer_functions[self.model_name](**kwargs)
            #     np.save(os.path.join(self.output_folder, f"{self.model_name}_clusters_pre_{epoch}.npy"), _clusters)
            #     np.save(os.path.join(self.output_folder, f"{self.model_name}_clusters_pre_contrastive_{epoch}.npy"), _clusters_2)
            #     np.save(os.path.join(self.output_folder, f"{self.model_name}_contrastive_{epoch}.npy"), _X_2)

            else:
                _X = self.infer_functions[self.model_name](**kwargs)

            np.save(os.path.join(self.output_folder, f"{self.model_name}_{epoch}_{self.iteration}.npy"), _X)

        # save training stats
        with open(os.path.join(self.output_folder, f"stats_{epoch}.pickle"), 'wb') as handle:
            pickle.dump(self.train_stats_dicts[self.model_name], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_epoch_r(self, **kwargs):
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_r(X_batch=X_batch, **kwargs)

    def train_epoch_z(self,  **kwargs):
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_z(X_batch=X_batch, **kwargs)

    def train_epoch_z_ZINB(self,  **kwargs):
        l = 0.
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            l, _ = self.process_minibatch_z_ZINB(X_batch=X_batch, **kwargs)
        print("\nloss = " + str(l))
        return l

    def train_epoch_annotator(self,  **kwargs):
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(self.dataloader_train)):
            y_batch = torch.flatten(y_batch)
            self.process_minibatch_annotator(X_batch, y_batch, **kwargs)

    def train_epoch_cluster(self,  **kwargs):
        for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_cluster(X_batch, gene_set_batch, **kwargs)
    
    def train_epoch_anno(self,  **kwargs):
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            self.process_minibatch_anno(X_batch,  **kwargs)
    
    def train_epoch_anno_ZINB(self,  **kwargs):
        l = 0.
        for batch_idx, (X_batch, clusters_true, subjects) in enumerate(tqdm(self.dataloader_train)):
            loss, _, _ = self.process_minibatch_anno_ZINB(X_batch, subject=subjects, **kwargs)
            l = l + loss
        print("\nloss = " + str(l))
        return l


    def train_epoch_contrastive(self,  **kwargs):
        l = 0.
        for batch_idx, (X_batch, clusters_pre, subject) in enumerate(tqdm(self.dataloader_train)):
            loss, _, _ = self.process_minibatch_contrastive(X_batch, clusters_pre, subject, **kwargs)
            l = l + loss
        print("\nloss = " + str(l))
        return l

    def process_minibatch_cluster(self, X_batch, gene_set_batch, weight_decay: float = 0):
        """
        Process minibatch for the annocluster model.

        Parameters
        ----------
        X_batch: torch.Tensor
            gene expression
        gene_set_batch: torch.Tensor
            gene set activity scores
        weight_decay: float
            hyperparameter gamma regularizing exclusive lasso penalty

        """
        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
        gene_set_batch = gene_set_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training and self.model_2nd.training:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_2nd.zero_grad(set_to_none=True)

        x_e, x_q, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch)
        y_pre = self.model_2nd(gene_set_batch)
        pre_prob = torch.exp(y_pre)

        l = self.model.loss(X_batch, x_e, x_q, z_dist, prior_prob=pre_prob)
        l_prob = self.model_2nd.loss(y_pre, k)

        l_prob += weight_decay * torch.sum(torch.square(torch.sum(torch.abs(self.annotator_param_list[0]), dim=0)))

        if self.model.training and self.model_2nd.training:
            l.backward(retain_graph=True)
            self.optimizer.step()

            l_prob.backward(retain_graph=True)
            self.optimizer_2nd.step()

            self.scheduler.step()
            self.scheduler_2nd.step()

        return l.detach().cpu().item(), l_prob.detach().cpu().numpy(), k.detach().cpu().numpy(), \
               z_e.detach().cpu().numpy()


    def process_minibatch_anno(self, X_batch,  weight_decay: float = 0):
        """
        Process minibatch for the annocluster model.

        Parameters
        ----------
        X_batch: torch.Tensor
            gene expression
        gene_set_batch: torch.Tensor
            gene set activity scores
        weight_decay: float
            hyperparameter gamma regularizing exclusive lasso penalty

        """
        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
     

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)
            

        x_e, x_q, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch)
        

        l = self.model.loss(X_batch, x_e, x_q)



        if self.model.training:
            l.backward(retain_graph=True)
            self.optimizer.step()
            self.scheduler.step()


        return l.detach().cpu().item(), k.detach().cpu().numpy(), \
               z_e.detach().cpu().numpy()

    def process_minibatch_anno_ZINB(self, X_batch, subject, weight_decay: float = 0):
        """
        Process minibatch for the annocluster model.

        Parameters
        ----------
        X_batch: torch.Tensor
            gene expression
        gene_set_batch: torch.Tensor
            gene set activity scores
        weight_decay: float
            hyperparameter gamma regularizing exclusive lasso penalty

        """
        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
        ind_1 = np.where(subject == 0)
        ind_2 = np.where(subject == 1)

        X_batch_1 = X_batch[ind_1]
        X_batch_2 = X_batch[ind_2]
        subject_1 = subject[ind_1]
        subject_2 = subject[ind_2]
        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)
        
        if X_batch_1.shape[0]>0 and X_batch_2.shape[0]>0:
            e_mean_1, e_disp_1, e_pi_1, q_mean_1, q_disp_1, q_pi_1, z_e_1, z_q_1, k_1, z_dist_1, dist_prob_1 = self.model(X_batch_1, 0)
            e_mean_2, e_disp_2, e_pi_2, q_mean_2, q_disp_2, q_pi_2, z_e_2, z_q_2, k_2, z_dist_2, dist_prob_2 = self.model(X_batch_2, 1)

            X_batch = torch.cat([X_batch_1, X_batch_2],dim=0)
            e_mean = torch.cat([e_mean_1, e_mean_2],dim=0)
            e_disp = torch.cat([e_disp_1, e_disp_2],dim=0)
            e_pi = torch.cat([e_pi_1, e_pi_2],dim=0)
            q_mean = torch.cat([q_mean_1, q_mean_2],dim=0)
            q_disp = torch.cat([q_disp_1, q_disp_2],dim=0)
            q_pi = torch.cat([q_pi_1, q_pi_2],dim=0)
            z_e = torch.cat([z_e_1, z_e_2],dim=0)
            k = torch.cat([k_1, k_2],dim=0)
        elif X_batch_1.shape[0]>0 and X_batch_2.shape[0]==0:
            e_mean, e_disp, e_pi, q_mean, q_disp, q_pi, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch_1, 0)
        elif X_batch_1.shape[0]==0 and X_batch_2.shape[0]>0:
            e_mean, e_disp, e_pi, q_mean, q_disp, q_pi, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch_2, 0)






        l = self.model.loss(X_batch, e_mean, e_disp, e_pi, q_mean, q_disp, q_pi)
        if self.model.training:
            l.backward(retain_graph=True)
            self.optimizer.step()
            self.scheduler.step()


        return l.detach().cpu().item(), k.detach().cpu().numpy(), \
               z_e.detach().cpu().numpy()


    def evaluate_anno(self, **kwargs):
        """

        Evaluate the total loss and each loss term for the training data and the total loss for the validation set
        for the annotcluster model.

        """


        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        _l_e = []
        _l_q = []
        _mse_l = []


        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
            

            x_e, x_q, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch)


            l = self.model.loss(X_batch, x_e, x_q).detach().cpu().item()


            # calculate each loss
            l_e = self.model.mse_loss(X_batch, x_e).detach().cpu().item()
            l_q = self.model.mse_loss(X_batch, x_q).detach().cpu().item()
            mse_l = self.model._loss_reconstruct(X_batch, x_e, x_q).detach().cpu().item()


            _loss.append(l)
            _l_e.append(l_e)
            _l_q.append(l_q)
            _mse_l.append(mse_l)


        train_stats['train_loss'].append(np.mean(_loss))
        train_stats['train_mse'].append(np.mean(_mse_l))
        train_stats['train_mse_e'].append(np.mean(_l_e))
        train_stats['train_mse_q'].append(np.mean(_l_q))


        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _, _ = self.process_minibatch_anno(X_batch,  **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)
    def evaluate_anno_ZINB(self, **kwargs):
        """

        Evaluate the total loss and each loss term for the training data and the total loss for the validation set
        for the annotcluster model.

        """


        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        _l_e = []
        _l_q = []
        _mse_l = []


        for batch_idx, (X_batch, clusters_true, subject) in enumerate(tqdm(self.dataloader_train)):
            X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
            ind_1 = np.where(subject == 0)
            ind_2 = np.where(subject == 1)

            X_batch_1 = X_batch[ind_1]
            X_batch_2 = X_batch[ind_2]
            subject_1 = subject[ind_1]
            subject_2 = subject[ind_2]

            

            e_mean_1, e_disp_1, e_pi_1, q_mean_1, q_disp_1, q_pi_1, z_e_1, z_q_1, k_1, z_dist_1, dist_prob_1 = self.model(X_batch_1, subject_1[0])
            e_mean_2, e_disp_2, e_pi_2, q_mean_2, q_disp_2, q_pi_2, z_e_2, z_q_2, k_2, z_dist_2, dist_prob_2 = self.model(X_batch_2, subject_2[0])

            X_batch = torch.cat([X_batch_1, X_batch_2],dim=0)

            e_mean = torch.cat([e_mean_1, e_mean_2],dim=0)
            e_disp = torch.cat([e_disp_1, e_disp_2],dim=0)
            e_pi = torch.cat([e_pi_1, e_pi_2],dim=0)
            q_mean = torch.cat([q_mean_1, q_mean_2],dim=0)
            q_disp = torch.cat([q_disp_1, q_disp_2],dim=0)
            q_pi = torch.cat([q_pi_1, q_pi_2],dim=0)





            l = self.model.loss(X_batch, e_mean, e_disp, e_pi, q_mean, q_disp, q_pi).detach().cpu().item()


            # calculate each loss
            l_e = self.zinb_loss(x=X_batch, mean=e_mean, disp=e_disp, pi=e_pi).detach().cpu().item()
            l_q = self.zinb_loss(x=X_batch, mean=q_mean, disp=q_disp, pi=q_pi).detach().cpu().item()
            mse_l = l


            _loss.append(l)
            _l_e.append(l_e)
            _l_q.append(l_q)
            _mse_l.append(mse_l)




        # for batch_idx, (X_batch) in enumerate(tqdm(self.second_dataloader_train)):
        #     X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
            

        #     e_mean, e_disp, e_pi, q_mean, q_disp, q_pi, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch, subject = 2)


        #     l = self.model.loss(X_batch, e_mean, e_disp, e_pi, q_mean, q_disp, q_pi).detach().cpu().item()


        #     # calculate each loss
        #     l_e = self.zinb_loss(x=X_batch, mean=e_mean, disp=e_disp, pi=e_pi).detach().cpu().item()
        #     l_q = self.zinb_loss(x=X_batch, mean=q_mean, disp=q_disp, pi=q_pi).detach().cpu().item()
        #     mse_l = l


        #     _loss.append(l)
        #     _l_e.append(l_e)
        #     _l_q.append(l_q)
        #     _mse_l.append(mse_l)

        train_stats['train_loss'].append(np.mean(_loss))
        train_stats['train_mse'].append(np.mean(_mse_l))
        train_stats['train_mse_e'].append(np.mean(_l_e))
        train_stats['train_mse_q'].append(np.mean(_l_q))


        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _, _ = self.process_minibatch_anno(X_batch,  **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def evaluate_cluster(self, **kwargs):
        """

        Evaluate the total loss and each loss term for the training data and the total loss for the validation set
        for the annotcluster model.

        """


        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        _l_e = []
        _l_q = []
        _mse_l = []
        _prob_z_l = []

        for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_train)):
            X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
            gene_set_batch = gene_set_batch.to(self.device, non_blocking=self.non_blocking).float()

            x_e, x_q, z_e, z_q, k, z_dist, dist_prob = self.model(X_batch)
            y_pre = self.model_2nd(gene_set_batch)
            pre_prob = torch.exp(y_pre)

            l = self.model.loss(X_batch, x_e, x_q, z_dist, prior_prob=pre_prob).detach().cpu().item()
            l_prob = self.model_2nd.loss(y_pre, k).detach().cpu().item()

            # calculate each loss
            l_e = self.model.mse_loss(X_batch, x_e).detach().cpu().item()
            l_q = self.model.mse_loss(X_batch, x_q).detach().cpu().item()
            mse_l = self.model._loss_reconstruct(X_batch, x_e, x_q).detach().cpu().item()
            prob_z_l = self.model._loss_z_prob(z_dist, prior_prob=pre_prob).detach().cpu().item()

            _loss.append(l)
            _l_e.append(l_e)
            _l_q.append(l_q)
            _mse_l.append(mse_l)
            _prob_z_l.append(prob_z_l)

        train_stats['train_loss'].append(np.mean(_loss))
        train_stats['train_mse'].append(np.mean(_mse_l))
        train_stats['train_mse_e'].append(np.mean(_l_e))
        train_stats['train_mse_q'].append(np.mean(_l_q))
        train_stats['train_prob_z_l'].append(np.mean(_prob_z_l))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _, _, _ = self.process_minibatch_cluster(X_batch, gene_set_batch, **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def infer_anno(self, **kwargs):
        """

        Get z_e and clusters from the annocluster model. Also calculate ARI and NMI scores comparing with
        the ground truth (if available).

        Returns
        -------
        z_annocluster: numpy array
            z_e of cells
        clusters_pre: numpy array
            cluster assignments

        """
        train_stats = self.train_stats_dicts[self.model_name]

        with torch.no_grad():
            self.model.eval()


            # inference
            k_list = []
            z_e_list = []

            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_all)):
                _, k, z_e = self.process_minibatch_anno(X_batch, **kwargs)
                k_list.append(k)
                z_e_list.append(z_e)

            clusters_pre = np.concatenate(k_list)
            z_annocluster = np.concatenate(z_e_list)

            if self.dataset.clusters_true is not None:
                if self.dataset.N > 5e4:
                    idx_stratified, _ = train_test_split(range(self.dataset.N), test_size=0.5,
                                                         stratify=self.dataset.clusters_true)
                else:
                    idx_stratified = range(self.dataset.N)

                # metrics
                ari_smaller = adjusted_rand_score(clusters_pre[idx_stratified],
                                                  self.dataset.clusters_true[idx_stratified])
                nmi_smaller = adjusted_mutual_info_score(clusters_pre, self.dataset.clusters_true)
                print(f"annocluster: ARI for smaller cluster: {ari_smaller}")
                print(f"annocluster: NMI for smaller cluster: {nmi_smaller}")
            else:
                ari_smaller = np.nan
                nmi_smaller = np.nan

            train_stats["ARI"].append(ari_smaller)
            train_stats["NMI"].append(nmi_smaller)

        return z_annocluster, clusters_pre
    def infer_anno_ZINB(self, **kwargs):
        """

        Get z_e and clusters from the annocluster model. Also calculate ARI and NMI scores comparing with
        the ground truth (if available).

        Returns
        -------
        z_annocluster: numpy array
            z_e of cells
        clusters_pre: numpy array
            cluster assignments

        """
        train_stats = self.train_stats_dicts[self.model_name]

        with torch.no_grad():
            self.model.eval()


            # inference
            k_list = []
            z_e_list = []

            for batch_idx, (X_batch, clusters_true, subject) in enumerate(tqdm(self.dataloader_all)):


                    
                _, k, z_e = self.process_minibatch_anno_ZINB(X_batch, subject = subject, **kwargs)
                k_list.append(k)
                z_e_list.append(z_e)

            clusters_pre = np.concatenate(k_list)
            z_annocluster = np.concatenate(z_e_list)



            # for batch_idx, (X_batch) in enumerate(tqdm(self.second_dataloader_all)):
            #     _, k, z_e = self.process_minibatch_anno_ZINB(X_batch, subject = 2, **kwargs)
            #     k_list.append(k)
            #     z_e_list.append(z_e)

            # clusters_pre_2 = np.concatenate(k_list)
            # z_annocluster_2 = np.concatenate(z_e_list)

        return z_annocluster, clusters_pre

    def infer_cluster(self, **kwargs):
        """

        Get z_e and clusters from the annocluster model. Also calculate ARI and NMI scores comparing with
        the ground truth (if available).

        Returns
        -------
        z_annocluster: numpy array
            z_e of cells
        clusters_pre: numpy array
            cluster assignments

        """
        train_stats = self.train_stats_dicts[self.model_name]

        with torch.no_grad():
            self.model.eval()
            if self.model_2nd is not None:
                self.model_2nd.eval()

            # inference
            k_list = []
            z_e_list = []

            for batch_idx, (X_batch, gene_set_batch) in enumerate(tqdm(self.dataloader_all)):
                _, _, k, z_e = self.process_minibatch_cluster(X_batch, gene_set_batch, **kwargs)
                k_list.append(k)
                z_e_list.append(z_e)

            clusters_pre = np.concatenate(k_list)
            z_annocluster = np.concatenate(z_e_list)

            if self.dataset.clusters_true is not None:
                if self.dataset.N > 5e4:
                    idx_stratified, _ = train_test_split(range(self.dataset.N), test_size=0.5,
                                                         stratify=self.dataset.clusters_true)
                else:
                    idx_stratified = range(self.dataset.N)

                # metrics
                ari_smaller = adjusted_rand_score(clusters_pre[idx_stratified],
                                                  self.dataset.clusters_true[idx_stratified])
                nmi_smaller = adjusted_mutual_info_score(clusters_pre, self.dataset.clusters_true)
                print(f"annocluster: ARI for smaller cluster: {ari_smaller}")
                print(f"annocluster: NMI for smaller cluster: {nmi_smaller}")
            else:
                ari_smaller = np.nan
                nmi_smaller = np.nan

            train_stats["ARI"].append(ari_smaller)
            train_stats["NMI"].append(nmi_smaller)

        return z_annocluster, clusters_pre


    def process_minibatch_annotator(self, X_batch, y_batch, weight_decay: float = 0):

        """
        Process minibatch for the annotator model.

        Parameters
        ----------
        X_batch: torch.Tensor
            gene expression
        y_batch: torch.Tensor
            cluster assignment
        weight_decay: float
            hyperparameter gamma regularizing exclusive lasso penalty

        """

        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
        y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        y_pre = self.model(X_batch)
        l = self.model.loss(y_pre, y_batch)

        l += weight_decay * torch.sum(torch.square(torch.sum(torch.abs(self.annotator_param_list[0]), dim=0)))

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()

        return l.detach().cpu().item(), y_pre.detach().cpu().numpy()

    def evaluate_annotator(self,  **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]
        _loss = []
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(self.dataloader_train)):
            y_batch = torch.flatten(y_batch)
            l, _ = self.process_minibatch_annotator(X_batch, y_batch, **kwargs)
            _loss.append(l)

        train_stats['train_loss'].append(np.mean(_loss))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch, y_batch) in enumerate(tqdm(self.dataloader_val)):
                y_batch = torch.flatten(y_batch)
                l, _ = self.process_minibatch_annotator(X_batch, y_batch, **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def infer_annotator(self):
        """

        Get the prediction of labels on the all data.

        Return
        ------
        clusters_classifier: numpy array
            predicted labels from the trained annotator

        """
        with torch.no_grad():
            self.model.eval()
            tf_prob = self.model(
                torch.from_numpy(self.dataset.data).to(self.device, non_blocking=self.non_blocking).float())

        clusters_prob_pre = torch.exp(tf_prob)
        clusters_classifier = np.argmax(clusters_prob_pre.detach().cpu().numpy(), axis=1)

        return clusters_classifier

    def process_minibatch_r(self, X_batch, alpha: float = 0, beta: float = 0, beta_list: torch.Tensor = None,
                            gene_covered_matrix: torch.Tensor = None):
        """

        Process minibatch for gene set activity scores model (named as r).

        Parameters
        ----------
        X_batch: torch.Tensor
            gene expression
        alpha: float
            hyperparameter regularizing L1 term in the set cover loss
        beta: float
            hyperparameter regularizing set loss term in the set cover loss
        beta_list: torch.Tensor
            beta values for all genes
        gene_covered_matrix: torch.Tensor
            gene set membership matrix with genes that are at least covered by one of the available sets

        """


        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        x_e, z_e = self.model(X_batch)
        l = self.model.loss(X_batch.float(), x_e.float())

        # add sparsity regularization on the output
        if alpha != 0:
            sparsity_penalty = alpha * torch.mean(torch.abs(z_e))
            l += sparsity_penalty
        else:
            sparsity_penalty = torch.zeros(1)

        if beta != 0:
            cover_penality = - torch.mean(torch.matmul(torch.mm(z_e, gene_covered_matrix),
                                                       beta_list))
            l += cover_penality
        else:
            cover_penality = torch.zeros(1)

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()

        return l.detach().cpu().item(), z_e.detach().cpu().numpy(), \
               sparsity_penalty.detach().cpu().numpy(), cover_penality.detach().cpu().numpy()

    def evaluate_r(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]
        _loss = []
        _sp = []
        _cp = []

        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            l, _, sp, cp = self.process_minibatch_r(X_batch, **kwargs)
            _loss.append(l)
            _sp.append(sp)
            _cp.append(cp)

        train_stats['train_loss'].append(np.mean(_loss))
        train_stats['train_sparsity'].append(np.mean(_sp))
        train_stats['train_set_cover'].append(np.mean(_cp))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _, _, _ = self.process_minibatch_r(X_batch, **kwargs)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def infer_r(self, **kwargs):

        """

        Returns
        -------
        z_gene_set: numpy array
            gene set activity scores

        """
        with torch.no_grad():
            self.model.eval()

            z_e_list = []
            _loss = []
            print(f"Start inferring ...")
            for batch_idx, (X_batch) in enumerate(self.dataloader_all):
                l, z_e, _, _ = self.process_minibatch_r(X_batch, **kwargs)

                z_e_list.append(z_e)
                _loss.append(l)

            print(f"Finish inferring ...")
            z_gene_set = np.concatenate(z_e_list)

            del z_e_list
            del _loss
            del z_e

            gc.collect()

        return z_gene_set

    def process_minibatch_z(self, X_batch):

        """
        Process minibatch for pretraining the autocluster model (named as pretrain_z)

        """

        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        x_e, z_e = self.model(X_batch)

        l = self.model.loss(X_batch.float(), x_e.float())

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()
        return l.detach().cpu().item(), z_e.detach().cpu().numpy()

    def evaluate_z(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            l, _ = self.process_minibatch_z(X_batch)
            _loss.append(l)

        train_stats['train_loss'].append(np.mean(_loss))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _ = self.process_minibatch_z(X_batch)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def infer_z(self):

        """

        Returns
        -------
        z_init: numpy array
            z_e from the pretrain model

        """

        with torch.no_grad():
            self.model.eval()

            z_e_list = []

            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_all)):
                _, z_e = self.process_minibatch_z(X_batch)
                z_e_list.append(z_e)

            z_init = np.concatenate(z_e_list)

            del z_e_list
            gc.collect()

        return z_init

# Autoencoder

    def process_minibatch_z_ZINB(self, X_batch):

        """
        Process minibatch for pretraining the autocluster model (named as pretrain_z)

        """

        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)

        _mean, _disp, _pi, z_e = self.model(X_batch)

        l = self.model.loss(X_batch, _mean, _disp, _pi)

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()
        return l.detach().cpu().item(), z_e.detach().cpu().numpy()

    def evaluate_z_ZINB(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_train)):
            l, _ = self.process_minibatch_z_ZINB(X_batch)
            _loss.append(l)

        train_stats['train_loss'].append(np.mean(_loss))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_val)):
                l, _ = self.process_minibatch_z_ZINB(X_batch)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def infer_z_ZINB(self):

        """

        Returns
        -------
        z_init: numpy array
            z_e from the pretrain model

        """

        with torch.no_grad():
            self.model.eval()

            z_e_list = []

            for batch_idx, (X_batch) in enumerate(tqdm(self.dataloader_all)):
                _, z_e = self.process_minibatch_z_ZINB(X_batch)
                z_e_list.append(z_e)

            z_init = np.concatenate(z_e_list)

            del z_e_list
            gc.collect()

        return z_init

    def process_minibatch_contrastive(self, X_batch, clusters_pre, subject):

        """
        Process minibatch for pretraining the autocluster model (named as pretrain_z)

        """

        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking).float()
        clusters_pre = clusters_pre.to(self.device, non_blocking=self.non_blocking).float()
 

        if self.model.training:
            self.optimizer.zero_grad(set_to_none=True)


        z_e, z_c = self.model(X_batch)


        # clusters_pre = clusters_pre.numpy()
        l = self.model.loss(z_c, clusters_pre, self.contrastive_pair_1, self.contrastive_pair_2, subject, self.device, self.non_blocking)

        if self.model.training:
            l.backward()
            self.optimizer.step()
            self.scheduler.step()
        return l.detach().cpu().item(), z_e.detach().cpu().numpy(), z_c.detach().cpu().numpy()

    def evaluate_contrastive(self, **kwargs):
        train_stats = self.train_stats_dicts[self.model_name]

        _loss = []
        for batch_idx, (X_batch, clusters_pre, subject) in enumerate(tqdm(self.dataloader_train)):
            l, _, _ = self.process_minibatch_contrastive(X_batch, clusters_pre, subject)
            _loss.append(l)

        train_stats['train_loss'].append(np.mean(_loss))

        if self.percent_training != 1:
            _loss = []
            for batch_idx, (X_batch, clusters_pre, subject) in enumerate(tqdm(self.dataloader_val)):
                l, _, _= self.process_minibatch_contrastive(X_batch, clusters_pre, subject)
                _loss.append(l)

            train_stats['val_loss'].append(np.mean(_loss))
        else:
            train_stats['val_loss'].append(np.nan)

    def infer_contrastive(self):

        """

        Returns
        -------
        z_init: numpy array
            z_e from the pretrain model

        """

        with torch.no_grad():
            self.model.eval()

            z_e_list = []
            c_e_list = []

            for batch_idx, (X_batch, clusters_pre, subject) in enumerate(tqdm(self.dataloader_all)):
                _, z_e, c_e = self.process_minibatch_contrastive(X_batch, clusters_pre, subject)
                z_e_list.append(z_e)
                c_e_list.append(c_e)

            z_init = np.concatenate(z_e_list)
            c_init = np.concatenate(c_e_list)

            del z_e_list
            del c_e_list
            gc.collect()

        return z_init

    def infer_contrastive_c(self):

        """

        Returns
        -------
        z_init: numpy array
            z_e from the pretrain model

        """

        with torch.no_grad():
            self.model.eval()

            z_e_list = []
            c_e_list = []

            for batch_idx, (X_batch, clusters_pre, subject) in enumerate(tqdm(self.dataloader_all)):
                _, z_e, c_e = self.process_minibatch_contrastive(X_batch, clusters_pre, subject)
                z_e_list.append(z_e)
                c_e_list.append(c_e)

            z_init = np.concatenate(z_e_list)
            c_init = np.concatenate(c_e_list)

            del z_e_list
            del c_e_list
            gc.collect()

        return c_init
    @staticmethod
    def visualize_UMAP(X, epoch:int, output_folder: str, clusters_true=None, clusters_pre=None,
                       color_palette:str = "tab20"):
        """

        Visualize the low-dimensional representations using UMAP and save figures.

        Parameters
        ----------
        X: numpy array
            low-dimensional representations
        epoch: integer
            epoch of the model based on which the low-dimensional representations is inferred
        clusters_true: numpy array
            ground truth labels
        clusters_pre: numpy array
            cluster assignment

        """

        print(f"Start visualizing using UMAP...")
        umap_original = umap.UMAP().fit_transform(X)

        # color by cluster
        hues = {'label': clusters_true, 'cluster': clusters_pre}

        for k, v in hues.items():
            df_plot = pd.DataFrame(umap_original)
            if v is None:
                df_plot['label'] = np.repeat("Label not available", df_plot.shape[0])
            else:
                df_plot['label'] = v
            df_plot['label'].astype('str')
            df_plot.columns = ['dim_1', 'dim_2', 'label']

            plt.figure(figsize=(10, 10))
            sns.scatterplot(x='dim_1', y='dim_2', hue='label', data=df_plot, palette=color_palette,
                            legend=True)
            plt.title(f"Encoding (r) colored by {k}")
            plt.savefig(os.path.join(output_folder, f"r_{epoch}_{k}.png"), bbox_inches="tight", format="png")
            plt.close()

        del X
        del umap_original
        del df_plot

        gc.collect()

