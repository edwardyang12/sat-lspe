from nets.transformer import GraphTransformer
import torch 

net_params = {
    "L": 16,
    "hidden_dim": 64,
    "out_dim": 64,
    "residual": True,
    "edge_feat": True,
    "readout": "mean",
    "in_feat_dropout": 0.0,
    "dropout_lspe": 0.0,
    "batch_norm": True,
    "pos_enc_dim": 20,
    "pe_init": "rand_walk",
    "use_lapeig_loss": True,
    "alpha_loss": 1,
    "lambda_loss": 1e-1,
    "num_atom_type": 65,
    "num_bond_type":5,


    "drop_out_sat": 0.3,
    "se": "gnn",
    "gnn_type": "pna2",
    "num_heads": 8,
    "num_layers": 6,
    "edge_dim": 32,
    "k_hop": 3,
    "deg": torch.ones((119828))
}
model = GraphTransformer(net_params)
print(model)
print(sum([p.numel() for p in model.parameters() if p.requires_grad]))