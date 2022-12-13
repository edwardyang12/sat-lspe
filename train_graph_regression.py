"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import dgl

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, batch_complete, batch_snorm_n, batch_edges, \
            batch_ptr) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        batch_complete = batch_complete.to(device)
        batch_edges = batch_edges.to(device)
        batch_deg = batch_graphs.ndata['deg'].to(device)
        batch_ptr = batch_ptr.to(device)
        optimizer.zero_grad()

        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            # print(batch_pos_enc)
        except KeyError:
            batch_pos_enc = None
        
        # print("================================data===============")
        # print(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        
        batch_scores, __ = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n, \
            batch_edges, batch_deg, batch_complete, batch_ptr) 
        del __
        # print(torch.max(batch_scores))

        loss = model.loss(batch_scores, batch_targets)
        # print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    out_graphs_for_lapeig_viz = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_complete, batch_snorm_n, batch_edges, \
            batch_ptr) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_complete = batch_complete.to(device)
            batch_edges = batch_edges.to(device)
            batch_deg = batch_graphs.ndata['deg'].to(device)
            batch_ptr = batch_ptr.to(device)
            optimizer.zero_grad()

            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                # print(batch_pos_enc)
            except KeyError:
                batch_pos_enc = None
            
            batch_scores, batch_g = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n, \
                batch_edges, batch_deg, batch_complete, batch_ptr) 

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            
            out_graphs_for_lapeig_viz += dgl.unbatch(batch_g)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, out_graphs_for_lapeig_viz

