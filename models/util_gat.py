import sys
sys.path.append("process_data/")
sys.path.append("models/")

from class_dataset import CTA_Data
from class_gat_lstm import GAT_LSTM
from class_mve_loss import MVELoss
import util_gcnn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob 


def load_model(project_dir, save_dir, period, train_extent, adj_type, predict_hzn, time_size, lookback, ii, n_modes, n_stations, n_time, meanonly=False, homo=False):
    if type(adj_type) == str:
        adj_type=adj_type.replace('_', '-')
        nadj = adj_type.count('-')+1
    else:
        nadj = len(adj_type)
        adj_type='-'.join(adj_type)

    file = glob.glob(save_dir+"_"+str(ii)+"_*.pt")
    
    if len(file) == 0:
        print(save_dir+"_"+str(ii)+"_*.pt")
        print("Model %d not saved." % (ii))
        return None

    try:
        assert len(file)==1
    except:
        print("Multiple Files Found!")
        for f in file:
            print(f)
            
#     print(file[0])
    saved = torch.load(file[0])
    if len(saved['hyperparameters']) == 14:
        (_,_,_,_,_,_,_,_,dropout,n_hid_units,nlstm,ngc,weight_decay,_) = saved['hyperparameters']
    else:
        (_,_,_,_,_,_,_,_,dropout,n_hid_units,nlstm,ngc,weight_decay) = saved['hyperparameters']

#     print(saved['model_state_dict'].keys())
    # assuming that meanonly and homoskedastic models will not be loaded
    net = GAT_LSTM(meanonly=meanonly, nadj = 1, 
           nmode=n_modes, nstation=n_stations, ntime=n_time, ndemo=0,
           nhead=4*((ii%2)+1), nhid_g=n_hid_units, nga=ngc, nhid_l=n_hid_units, nlstm=nlstm, 
           nhid_fc=n_hid_units, dropout=dropout, homo=homo)
    
    net.load_state_dict(saved['model_state_dict'])

    return net


def ensemble(project_dir, save_dir, period, predict_hzn, time_size, lookback, ensemble_model_numbers, device, train_extent, adj_type, data, adj, spatial, downtown_filter, dist='norm'):
    # only normal ensemble available

    n_time = 96//time_size-7

    trainloader, trainloader_test, valloader, testloader, adj_torch, spatial_torch, y_train_eval, y_val_eval, y_test_eval = \
            util_gcnn.prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, val=True)
    (num_train, _, _, n_modes) = data['x'][0].shape
    (num_test, _, _, _) = data['x'][-1].shape
    n_stations = adj_torch.shape[0]

    y_train_eval = np.squeeze(y_train_eval)
    y_test_eval = np.squeeze(y_test_eval)

    mean_list = []
    std_list = []
    test_loss_list = []
    spatial_torch = None

    for ii in ensemble_model_numbers:

        net = load_model(project_dir, save_dir, period, train_extent, adj_type, predict_hzn, time_size, lookback, ii, n_modes, n_stations, n_time)
        
        if net is None:
            continue

        criterion = MVELoss(dist)
        net.eval()

        test_out_mean, test_out_std, test_loss = testset_output_gat(testloader, False, False, net, criterion, 
                adj_torch, spatial_torch, device, n_time)

        mean_list.append(test_out_mean)
        std_list.append(test_out_std)
        test_loss_list.append(test_loss)

    # calculate ensembled stats                     
    test_ens_mean = np.mean(np.array(mean_list), axis=0)
    test_ens_std = np.sqrt(np.mean(np.power(np.array(mean_list),2) \
            + np.power(np.array(std_list),2), axis=0)\
            - np.power(test_ens_mean,2))

    return data['ts'][-1], y_test_eval, test_ens_mean, test_ens_std, mean_list, std_list


def calc_attention(h, adj, W, a):
    (N,S,_) = h.shape
    _, out_features = W.shape
    
    # h.shape=(N,S,in_features) W.shape=(in_features,out_features) Wh.shape=(N,S,out_features)
    Wh = torch.matmul(h, W) 

    # a_input.shape=(N,S*S,2*out_features)
    a_input = torch.cat([Wh.repeat_interleave(S, dim=1), Wh.repeat(1,S,1)], dim=2)
    a_input = a_input.view(N,S,S,2*out_features)

    # a_input.shape=(N,S*S,2*out_features) a.shape=(2*out_features,1)
    e = nn.LeakyReLU(0.1)(torch.matmul(a_input, a).squeeze(3))

    attention = F.softmax(e, dim=2)

    return attention

def testset_output_gat(testloader, meanonly, homo, net, criterion, adj, demo, device, n_time, return_components=False, std=None):
    loss = 0
    net.eval()
    
    for i, data in enumerate(testloader, 0):

        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod = data
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_size=len(batch_x)
        batch_history = batch_history.float()
        batch_qod = batch_qod.view(-1,1)        
        batch_qod_onehot = torch.FloatTensor(batch_size, n_time)
        batch_qod_onehot.zero_()
        batch_qod_onehot.scatter_(1, batch_qod-6, 1)
        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod_onehot = batch_x.to(device),batch_y.to(device), batch_history.to(device),batch_weather.to(device), batch_los.to(device), batch_qod_onehot.to(device)

        # forward

        if return_components:
            outputs = net(batch_x, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot, True)
            return outputs
        else:
            outputs = net(batch_x, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot)

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
