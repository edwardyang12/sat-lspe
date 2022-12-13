"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gatedgcn_net import GatedGCNNet
from nets.transformer import GraphTransformer
from nets.sat_lspe import SATLSPENet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def SAT(net_params):
    return GraphTransformer(net_params)

def SATLSPE(net_params):
    return SATLSPENet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'SAT': SAT,
        'SATLSPE': SATLSPE
    }
        
    return models[MODEL_NAME](net_params)