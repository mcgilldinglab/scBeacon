import scanpy as sc
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csc_matrix

class AnnDataset(Dataset):
    def __init__(self, filepath: str):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = sc.read(filepath, dtype='float64', backed="r")
        self.N = self.data.shape[0]


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32).flatten()
        return x


class NumpyDataset(Dataset):
    def __init__(self, filepath: str, second_filepath: str = None):
        """

        Numpy array dataset.

        Parameters
        ----------
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """
        super().__init__()

        self.data = np.load(filepath)
        self.N = self.data.shape[0]
        self.G = self.data.shape[1]

        self.secondary_data = None
        if second_filepath is not None:
            self.secondary_data = np.load(second_filepath)
            assert len(self.secondary_data) == self.N, "The other file have same length as the main"

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        main = self.data[idx].flatten()

        if self.secondary_data is not None:
            secondary = self.secondary_data[idx].flatten()
            return main, secondary
        else:
            return main
            
class InteDataset(Dataset):
    def __init__(self, array, clusters_true):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = array


        self.clusters_true = clusters_true

        self.N = self.data.shape[0]


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        x = self.data[idx]
        x=x.astype(np.float32).flatten()

        return x

class ContrDataset(Dataset):
    def __init__(self, array, clusters_pre, subject):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = array


        self.clusters_pre = clusters_pre
        self.subject = subject
        self.N = self.data.shape[0]


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        x = self.data[idx]


        return x, self.clusters_pre[idx], self.subject[idx]


class NewAnnDataset(Dataset):
    def __init__(self, filepath: str):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = sc.read(filepath, dtype='float64', backed="r")


        self.clusters_true = self.data.obs['CellType_Category'].values

        self.N = self.data.shape[0]


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32).flatten()

        return x

class LabeledAnnDataset(Dataset):
    def __init__(self, filepath: str):
        """

        Anndata dataset.

        Parameters
        ----------
        label_name: string
            name of the cell type annotation, default 'label'
        second_filepath: string
            path to another input file other than the main one; e.g. path to predicted clusters or
            side information; only support numpy array

        """

        super().__init__()

        self.data = sc.read(filepath, dtype='float64', backed="r")

        self.N = self.data.shape[0]


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        x=csc_matrix(self.data.X[idx])[0].toarray()[0]
        x=x.astype(np.float32).flatten()

        return x