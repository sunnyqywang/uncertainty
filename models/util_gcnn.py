import sys
sys.path.append("process_data/")

from class_dataset import CTA_Data
from class_gcn_lstm import GCN_LSTM
from class_gat_lstm import GAT_LSTM

from class_mve_loss import MVELoss
import util_eval

import glob
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
import torch
from torch.utils.data import DataLoader

def testset_output_gcn(testloader, meanonly, homo, net, criterion, adj, demo, device, n_time, return_components=False, std=None, time_size=1):
#     net.eval()
    loss = 0
    for i, data in enumerate(testloader, 0):

        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod = data
        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()
        batch_size=len(batch_x)
        batch_history = batch_history.float()
        batch_qod = batch_qod.view(-1,1)        
        batch_qod_onehot = torch.FloatTensor(batch_size, n_time)
        batch_qod_onehot.zero_()
        batch_qod_onehot.scatter_(1, batch_qod, 1)
        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod_onehot = batch_x.to(device),batch_y.to(device), batch_history.to(device),batch_weather.to(device), batch_los.to(device), batch_qod_onehot.to(device)

#         print(batch_x)
        # forward
        if return_components:
            outputs = net(batch_x, None, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot, 1, True)
            return outputs
        else:
            outputs = net(batch_x, None, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot, 1)

#         print(outputs.shape)
        # loss
        if (meanonly) & (homo==0):
            loss += criterion(outputs, target=batch_y).item()
        elif homo>0:
            loss += criterion(outputs, std, batch_y).item()
        else:
            loss += criterion(outputs[:batch_size,:], outputs[batch_size:,:], batch_y).item()

        if (meanonly) & (homo==0):
            if i == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)

        elif homo>0:
            if i == 0:
                test_out_mean = outputs.cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs.cpu().detach().numpy()), axis=0)

        else:
            if i == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
                test_out_var = outputs[batch_size:,:].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)
                test_out_var = np.concatenate((test_out_var, outputs[batch_size:,:].cpu().detach().numpy()), axis=0)

    if (meanonly) & (homo==0):
        return test_out_mean, None, loss
    elif homo>0:
        return test_out_mean, std.cpu().detach().numpy(), loss
    else:
        return test_out_mean, test_out_var, loss

