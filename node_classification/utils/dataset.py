import torch
import torchvision
import torchvision.transforms as transforms

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, to_networkx, from_networkx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_sparse.tensor import from_scipy
from littleballoffur import RandomNodeSampler,DegreeBasedSampler, RandomEdgeSampler, RandomNodeEdgeSampler
import numpy as np
import dgl
import scipy.sparse as sp
import pickle as pkl
import sys
import networkx as nx

def get_dataset_dim(dataset):
    if dataset == 'ogbn-arxiv':
        num_classes =  40
    elif dataset == 'cora':
        num_classes = 7
    elif dataset == 'pubmed':
        num_classes = 3
    elif dataset == 'citeseer':
        num_classes = 6
    else:
        raise NotImplementedError
    return num_classes

def get_datasets(
        model_name, dataset_name, dataset_dir, 
        sampling=None, samplingtype=None):
    if dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root=dataset_dir)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data.edge_index = to_undirected(edge_index=data.edge_index, num_nodes=data.num_nodes)
        
        idx_train = split_idx['train']
        idx_valid = split_idx['valid']
        idx_test  = split_idx['test']

        node_num  = data.x.size(0)
        adj       = to_scipy_sparse_matrix(data.edge_index).tocoo()
        g         = dgl.DGLGraph()
        g.add_nodes(node_num)
        g.add_edges(adj.row,adj.col)
        features   = data.x
        edge_index = from_scipy(to_scipy_sparse_matrix(data.edge_index))
        labels     = data.y.squeeze()

        if sampling is not None:
            raise NotImplementedError
            graph=to_networkx(data,node_attrs=['x','y'],to_undirected=True)
            if samplingtype=='RandomNodeSampler':
                number_of_nodes = int(sampling*graph.number_of_nodes())
                sampler = RandomNodeSampler(number_of_nodes = number_of_nodes)
                new_graph = sampler.sample(graph)
            elif samplingtype=='DegreeBasedSampler':
                number_of_nodes = int(sampling*graph.number_of_nodes())
                sampler = DegreeBasedSampler(number_of_nodes = number_of_nodes)
                new_graph = sampler.sample(graph)
            elif samplingtype=='RandomEdgeSampler':            
                number_of_edges = int(sampling*graph.number_of_edges())
                sampler = RandomNodeEdgeSampler(number_of_edges = number_of_edges)
                new_graph = sampler.sample(graph)
                number_of_nodes = new_graph.number_of_nodes()
            else:
                raise NotImplementedError
            
            data1=from_networkx(new_graph)
            if samplingtype=="RandomEdgeSampler":
                idxes=list(new_graph.nodes.keys())
                data1.x=data.x[idxes].contiguous()
                data1.y=data.y[idxes].contiguous()
            data=data1
            train_num=int(0.55*number_of_nodes)
            test_num=int(0.3*number_of_nodes)
            all_index=np.arange(number_of_nodes)
            train_index=np.random.choice(all_index,size=train_num,replace=False)
            index_remain=set(all_index)-set(train_index)
            index_remain_array=np.array(list(index_remain))
            test_index=np.random.choice(index_remain_array,size=test_num,replace=False)
            val_index=list(index_remain-set(test_index))
            split_idx={"train":torch.tensor(train_index).long(),"test":torch.tensor(test_index).long(),"valid":torch.tensor(val_index).long()}
    elif dataset_name in ['cora', 'citeseer', 'pubmed']: 
        if model_name in ['gin_based_net']: 
            # based on https://github.com/TienjinHuang/UGTs-LoG/blob/main/UGTs_GNN/models/Dataloader.py
            adj, features, labels, idx_train, idx_val, idx_test = load_data_dgl(
                dataset_name=dataset_name, dataset_dir=dataset_dir)
            node_num = features.size()[0]
            g = dgl.DGLGraph()
            g.add_nodes(node_num)
            adj = adj.tocoo()
            g.add_edges(adj.row, adj.col)
            edge_index = None
            split_idx = dict(
                train = idx_train,
                valid = idx_val,
                test  = idx_test
            )
        elif model_name in ['gcn', 'gat']:
            data = Planetoid(
                root=dataset_dir, name=dataset_name.capitalize(), 
                split='public', transform=T.NormalizeFeatures())[0]
            g = None
            num_nodes = data.x.size(0)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
            if isinstance(edge_index, tuple):
            #     data.edge_index = edge_index[0]
            # else:
            #     data.edge_index = edge_index
                edge_index = edge_index[0]
            features  = data.x
            labels    = data.y
            idx_train = data.train_mask
            idx_valid = data.val_mask
            idx_test  = data.test_mask

            split_idx = dict(
                train = data.train_mask,
                valid = data.val_mask,
                test  = data.test_mask
            )
        else:
            raise ValueError
    else:
        raise NotImplementedError
    # g = torch.Generator()
    # g.manual_seed(seed)

    print(f'-----Dataset-----')
    print(f'train: {len(split_idx["train"])}')
    print(f'val:   {len(split_idx["valid"])}')
    print(f'test:  {len(split_idx["test"])}\n')

    # return data, split_idx
    return g, features, edge_index, labels, idx_train, idx_valid, idx_test

class Subset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
def load_data_dgl(dataset_name, dataset_dir):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    #import os
    #print("workspace",os.getcwd())
    objects = []
    for i in range(len(names)):
        with open(f"{dataset_dir}/{dataset_name.capitalize()}/raw/ind.{dataset_name}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{dataset_dir}/{dataset_name.capitalize()}/raw/ind.{dataset_name}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # preprocess adj
    # adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = torch.tensor((labels * range(l_num)).sum(axis=1), dtype=torch.int64)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))

    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    return features.todense()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)