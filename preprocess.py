from .utils import *
import h5py
import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import*
import random
# define class sc-sequencing dataset
import torch
import torch.utils.data as data


def normalize(adata, copy=True, highly_genes=False, filter_min_counts=False, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    """
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error
    """
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=0)
        
    if logtrans_input:
        sc.pp.log1p(adata)
        
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
        
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata,min_counts=0)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if normalize_input:
        sc.pp.scale(adata)
    return adata

class multiviewDataset(data.Dataset):
    
    # highly_genes: [[a1,a2...],[b1,b2...]], ai the views needs to select, bi the number of highly var genes
    def __init__(self, views, labels, device, if_normalize=True, highly_genes=[[0],[4000]]):
        
        self.n_views = len(views)
        self.views = views
        self.labels = labels
        self.device = device
        # we only normalize select highly var genes for view[highly_genes[0]]
        highly_cnt = 0 # count the number of views needs hvg
        if if_normalize:
            for i in range(self.n_views):
                self.views[i] = sc.AnnData(self.views[i])
                if i in list(highly_genes[0]):
                    self.views[i] = normalize(self.views[i], highly_genes=highly_genes[1][highly_cnt])
                    highly_cnt += 1
                else:
                    self.views[i] = normalize(self.views[i], copy=True, highly_genes=False, filter_min_counts=False,
                                              size_factors=True, normalize_input=True, logtrans_input=True)

                print("The dimension of view %d is %d"%(i, np.shape(self.views[i].X)[1]))
            
    def __getitem__(self, index):

        batch_cells = [torch.Tensor(v.X[index]).to(self.device) for v in self.views]
        batch_raw = [torch.Tensor(v.raw.X[index]).to(self.device) for v in self.views]
        batch_sf = [torch.tensor(v.obs['size_factors'][index]).to(self.device) for v in self.views]
        batch_labels = torch.tensor(self.labels[index]).to(self.device)
       
        return [batch_cells, batch_raw, batch_sf], batch_labels
    
    def __len__(self):
        return len(self.labels)
    
class multiviewDataset_nolabel(data.Dataset):
    
    # highly_genes: [[a1,a2...],[b1,b2...]], ai the views needs to select, bi the number of highly var genes
    def __init__(self, views, device, if_normalize=True, highly_genes=[[0],[4000]]):
        
        self.n_views = len(views)
        self.views = views
        self.device = device
        # we only normalize select highly var genes for view[highly_genes[0]]
        highly_cnt = 0 # count the number of views needs hvg
        if if_normalize:
            for i in range(self.n_views):
                self.views[i] = sc.AnnData(self.views[i])
                if i in list(highly_genes[0]):
                    self.views[i] = normalize(self.views[i], highly_genes=highly_genes[1][highly_cnt])
                    highly_cnt += 1
                else:
                    self.views[i] = normalize(self.views[i], copy=True, highly_genes=False, filter_min_counts=False,
                                              size_factors=True, normalize_input=True, logtrans_input=True)

                print("The dimension of view %d is %d"%(i, np.shape(self.views[i].X)[1]))
            
    def __getitem__(self, index):

        batch_cells = [torch.Tensor(v.X[index]).to(self.device) for v in self.views]
        batch_raw = [torch.Tensor(v.raw.X[index]).to(self.device) for v in self.views]
        batch_sf = [torch.tensor(v.obs['size_factors'][index]).to(self.device) for v in self.views]
       
        return [batch_cells, batch_raw, batch_sf]
    
    def __len__(self):
        return len(self.views[1])


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, var, uns