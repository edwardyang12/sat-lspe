import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

from layers.gatedgcn_layer import GatedGCNLayer
from layers.gatedgcn_lspe_layer import GatedGCNLSPELayer
from layers.mlp_readout_layer import MLPReadout
import torch_geometric.nn as gnn
from layers.transformer_layer import TransformerEncoderLayer
from einops import repeat
from nets.transformer import GraphTransformerEncoder

class SATLSPENet(nn.Module):
    def __init__(self, net_params, **kwargs):
        super().__init__()

        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout_transformer = net_params['drop_out_sat']
        dropout = net_params['dropout_lspe']
        num_heads = net_params['num_heads']
        num_layers = net_params['num_layers']
        gnn_type = net_params['gnn_type']
        k_hop = net_params['k_hop']
        se = net_params['se']
        deg = net_params['deg']

        batch_norm = True 

        kwargs['k_hop'] = k_hop
        kwargs['deg'] = deg
        kwargs['edge_dim'] = hidden_dim

        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        self.pos_enc_dim = net_params['pos_enc_dim']
        self.gnn_type = gnn_type
        self.se = se
        
        self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim) # self.embedding_abs_pe
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim) # self.embedding
        self.embedding_e = nn.Embedding(num_bond_type, hidden_dim) # self.embedding_edge
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # LSPE
        self.layers = nn.ModuleList([ GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, residual=self.residual) for _ in range(self.n_layers-1) ]) 
        self.layers.append(GatedGCNLSPELayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual))
       
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

        self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
        self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim*2, dropout_transformer, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

    def forward(self, graphs, x, pos_enc, e, snorm_n, edges, deg, complete, ptr, batch):   

        g = graphs 
        h = x 
        p = pos_enc

        subgraph_node_index = None
        subgraph_edge_index = None
        subgraph_indicator_index = None
        subgraph_edge_attr = None

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        p = self.embedding_p(p) 
        e = self.embedding_e(e)   
        
        
        # convnets
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)
        
        h = self.encoder(
            h, 
            edges, 
            complete,
            edge_attr=e, 
            degree=deg,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index, 
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            return_attn=False
        )

        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk':
            
            p = self.p_out(p)
            g.ndata['p'] = p

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)

            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            means = dgl.mean_nodes(g, 'p')

            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means
            
            g.ndata['p'] = p / batch_wise_p_l2_norms
        
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
            g.ndata['h'] = hp
        

        hg = dgl.mean_nodes(g, 'h')
        self.g = g # For util; To be accessed in loss() function
        
        return self.MLP_layer(hg), g
        
    def loss(self, scores, targets):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)
        
        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian 
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)

            loss_b = ( loss_b_1 + self.lambda_loss * loss_b_2 ) / ( self.pos_enc_dim * batch_size * n) 
            # print(loss_a, loss_b)
            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b
        else:
            loss = loss_a
        
        return loss