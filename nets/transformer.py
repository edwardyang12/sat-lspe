import torch
from torch import nn
import torch_geometric.nn as gnn
from layers.transformer_layer import TransformerEncoderLayer
from einops import repeat


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index, 
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, net_params, **kwargs):
        super().__init__()

        self.abs_pe = net_params['pe_init']
        self.abs_pe_dim = net_params['pos_enc_dim']
        in_size = net_params['num_atom_type']
        d_model = net_params['hidden_dim']
        num_class = 1
        dim_feedforward = d_model*2
        dropout = net_params['dropout']
        num_heads = net_params['num_heads']
        num_layers = net_params['num_layers']
        gnn_type = net_params['gnn_type']
        num_edge_features = net_params['num_bond_type']
        edge_dim = net_params['edge_dim']
        k_hop = net_params['k_hop']
        se = net_params['se']
        global_pool = net_params['readout']
        deg = net_params['deg']

        kwargs['deg'] = deg
        kwargs['edge_dim'] = edge_dim

        in_embed = True
        edge_embed=True
        batch_norm = True 
        use_global_pool = True
        use_edge_attr= True
        max_seq_len = None

        if self.abs_pe and self.abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(self.abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model) 
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, graphs, x, pos_enc, e, snorm_n, edges, deg, complete, ptr, \
        batch, return_attn=False):
        edge_index = edges 
        edge_attr = e
        
        if self.se == "khopgnn":
            # not yet implemented
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = complete
        abs_pe = pos_enc
        degree = deg
        output = self.embedding(x)
            
        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        output = self.encoder(
            output, 
            edge_index, 
            complete_edge_index,
            edge_attr=edge_attr, 
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index, 
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )
        # readout step
        if self.use_global_pool:
            output = self.pooling(output, batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        return self.classifier(output), None

    def loss(self, scores, targets):
        
        loss = nn.L1Loss()(scores, targets)
        
        return loss
