import torch
import torch.nn.functional as F
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv
import pickle
import dgl
from tqdm import tqdm
from scipy import sparse as sp
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx

# The dataset pickle and index files are in ./data/molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, name, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        self.name = name
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        if self.name == 'AQSOL':
            self._prepare_AQSOL() 
        else:
            self._prepare_ZINC() 

    def _prepare_ZINC(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in tqdm(self.data):
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            if molecule['num_atom'] > len(edge_list):
                print("wtf")
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            reachable = set()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
                reachable.add(src.item())
                reachable.add(dst.item())
            g.edata['feat'] = edge_features

            deg = 1. / torch.sqrt(1. + g.in_degrees())
            g.ndata['deg'] = deg
            
            if len(reachable) != molecule['num_atom']:
                print(len(reachable), molecule['num_atom'])

            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        

    def _prepare_AQSOL(self):
        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
        
        for molecule in tqdm(self.data):
            x, edge_attr, edge_index, y = molecule

            # invalid molecule
            if len(x)==0 or len(edge_index[0])==0 or len(edge_attr)==0:
                print(x, edge_index, edge_attr)
                continue

            # cleans data to only include reachable nodes
            reachable = set()
            for i in range(len(edge_index[0])):
                reachable.add(edge_index[0][i])
                reachable.add(edge_index[1][i])
            all = set([i for i in range(len(x))])
            cant = list(all.difference(reachable))
            remap = {i:i for i in range(len(x))}
            for i in cant:
                for j in range(i,len(x)):
                    remap[j]-=1
            x = np.delete(x, cant)

            # create dgl graph object
            g = dgl.DGLGraph()
            g.add_nodes(len(x))
            g.ndata['feat'] = torch.tensor(x).long()
            for i in range(len(edge_index[0])):
                g.add_edges(remap[edge_index[0][i]], remap[edge_index[1][i]])
            g.edata['feat'] = torch.tensor(edge_attr).long()

            # compute degree of each node in graph
            deg = 1. / torch.sqrt(1. + g.in_degrees())
            g.ndata['deg'] = deg

            self.graph_lists.append(g)
            self.graph_labels.append(torch.tensor(y))
        self.n_samples = len(self.graph_lists)
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        # # compute complete edge index
        # n = len(self.graph_lists[idx].nodes())
        # s = torch.arange(n)
        # complete = torch.vstack((s.repeat_interleave(n), s.repeat(n)))

        # # compute edge index
        # edges = torch.stack(self.graph_lists[idx].edges())

        # # compute ptr
        # ptr = torch.tensor([0, n])

        # snorm_n = torch.FloatTensor(n,1).fill_(1./float(n)).sqrt()  
        # return self.graph_lists[idx], self.graph_labels[idx], complete, snorm_n, edges, ptr
        return self.graph_lists[idx], self.graph_labels[idx]
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='AQSOL'):
        t0 = time.time()
        self.name = name

        if name=='AQSOL':
            self.num_atom_type = 65 
            self.num_bond_type = 5
            data_dir='/edward-slow-vol/Graphs/datasets/AQSOL/raw'
            self.train = MoleculeDGL(name, data_dir, 'train')
            self.val = MoleculeDGL(name, data_dir, 'val')
            self.test = MoleculeDGL(name, data_dir, 'test')
        else:
            self.num_atom_type = 28 
            self.num_bond_type = 4
            data_dir='/edward-slow-vol/Graphs/datasets/ZINC/raw'
            self.train = MoleculeDGL(name, data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(name, data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(name, data_dir, 'test', num_graphs=1000)
                
        print("Time taken: {:.4f}s".format(time.time()-t0))
        print(len(self.train))
        print(len(self.val))
        print(len(self.test))


def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g

def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    
    return g


class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading AQSOL datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f.train
            self.val = f.val
            self.test = f.test
            self.num_atom_type = f.num_atom_type
            self.num_bond_type = f.num_bond_type
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        batched_graph = dgl.batch(graphs)

        # compute complete edge index
        n = len(batched_graph.nodes())
        s = torch.arange(n)
        complete = torch.vstack((s.repeat_interleave(n), s.repeat(n)))

        # compute edge index
        edges = torch.stack(batched_graph.edges())

        ptr = [0]
        batch = []
        for index, i in enumerate(tab_sizes_n):
            ptr.append(ptr[-1]+i)
            batch = batch + [index]*i 
        ptr = torch.tensor(ptr)
        batch = torch.tensor(batch)
        
        return batched_graph, labels, complete, snorm_n, edges, ptr, batch

    def _add_eig_vecs(self, pos_enc_dim):

        # This is used if we visualize the eigvecs
        self.train.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in tqdm(self.train.graph_lists)]
        self.val.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in tqdm(self.val.graph_lists)]
        self.test.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in tqdm(self.test.graph_lists)]
    
    def _init_positional_encodings(self, pos_enc_dim, type_init):
        
        # Initializing positional encoding randomly with l2-norm 1
        self.train.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in tqdm(self.train.graph_lists)]
        self.val.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in tqdm(self.val.graph_lists)]
        self.test.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in tqdm(self.test.graph_lists)]


if __name__ == "__main__":
    test = MoleculeDatasetDGL('ZINC')
    print(test.test[0][0].ndata['feat'])
    print(test.test[0][0].edata['feat'])
    with open('ZINC.pkl', 'wb') as f:
        pickle.dump(test, f)
    dataset = MoleculeDataset('AQSOL')
    _, _, testset = dataset.train, dataset.val, dataset.test
    test_loader = DataLoader(testset, num_workers=0, batch_size=4, shuffle=False, collate_fn=dataset.collate)
    for i in test_loader:
        print(i[0].ndata['feat'], i[0].edata['feat'], i[1])
        break