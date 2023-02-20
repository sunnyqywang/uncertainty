import sys
sys.path.append("models/")
sys.path.append("process_data/")

import glob
import numpy as np
from scipy.stats import poisson, norm, laplace, lognorm
import torch

import util_data
import util_gcnn
import util_gat
from class_gcn_lstm import GCN_LSTM
from class_gat_lstm import GAT_LSTM
from class_mve_loss import MVELoss

from setup import *

def get_loss_curves(save_dir):
    
    file = glob.glob(save_dir)

    if len(file) == 0:
        print(save_dir)
        print("Model not saved!")
        return None

    try:
        assert len(file)==1
#         print(file, 'loaded')
    except:
        print("Multiple Files Found!")
        for f in file:
            print(f)

    saved = torch.load(file[0])
    
    return saved['train_loss'], saved['val_loss'], saved['test_loss']

def load_model(save_dir, n_modes, n_stations, n_time, meanonly=False, homo=False, model_type='GCN'):
    
    file = glob.glob(save_dir)

    if len(file) == 0:
        print(save_dir)
        print("Model not saved!")
        return None

    try:
        assert len(file)==1
#         print(file, 'loaded')
    except:
        print("Multiple Files Found!")
        for f in file:
            print(f)

    saved = torch.load(file[0])
    
    if model_type == 'GCN':
        (dropout, n_hid_units, nlstm, ngc, weight_decay) = saved['hyperparameters'][-6:-1]

        net = GCN_LSTM(meanonly=meanonly, homo=homo, nadj = 4, nmode=n_modes, nstation=n_stations, ntime=n_time, ndemo=0,
                nhid_g=n_hid_units, ngc=ngc, nhid_l=n_hid_units, nlstm=nlstm, 
                nhid_fc=n_hid_units, dropout=dropout)

    else:
        (n_head, dropout, n_hid_units, nlstm, ngc, weight_decay) = saved['hyperparameters'][-7:-1]
        net = GAT_LSTM(meanonly=meanonly, nadj = 1, 
           nmode=n_modes, nstation=n_stations, ntime=n_time, ndemo=0,
           nhead=n_head, nhid_g=n_hid_units, nga=ngc, nhid_l=n_hid_units, nlstm=nlstm, 
           nhid_fc=n_hid_units, dropout=dropout, homo=homo)
        
    net.load_state_dict(saved['model_state_dict'])
    
    return net
 

def load_and_run(dataset, model_type, dist, ii, save_dir, trainst, trained, testst, tested, 
                     time_size, predict_hzn, adj_type=["func","euc","con","net"], difference=True, max_lookback=6, lookback=6, 
                     std=None, include_spatial=False, device=torch.device("cpu"), train_extent='all'):
    
    data, adj, spatial, downtown_filter = \
        util_data.combine_datasources(project_dir, trainst, trained, testst, tested, 
        predict_hzn, time_size, difference, max_lookback, lookback, dataset)
            
    _, _, _, testloader, adj_torch, spatial_torch, y_train_eval, _, y_test_eval = \
        util_data.prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, 
                                    adj_type, bootstrap=False)
    
    if not include_spatial:
        spatial_torch=None
    
    n_modes = 3
    n_stations = adj_torch.shape[0]
    time_size = 1
    if dataset == 'rail_catchment':
        n_time = (96-28) // time_size
    else:
        n_time = (96-20) // time_size
        
    # Load Trained NN
    # Run Data Through Selected Network
    meanonly = False
    homo = False
    std = None
    
    if (dist == 'norm_homo') | (dist == 'mcdrop'):
        homo = 0.5
        meanonly = True
        std = torch.tensor([np.mean(y_train_eval)])*homo
        dist = 'norm_homo'
        
    if (dist == 'poisson') & (model_type == 'GAT') | (dist == 'poisson') & (dataset == 'census_tract'):
        # theoretically, the poisson distribution should be mean only
        # in GCN training this is set to False but the second half is not trained or used
        meanonly = True

    net = load_model(save_dir, n_modes, n_stations, n_time, meanonly=meanonly, homo=homo, model_type=model_type) 

    if dist != 'mcdrop':
        net.eval() 
        criterion = MVELoss(dist)
    
        if model_type == 'GCN':
            output_mean, output_var, _ = util_gcnn.testset_output_gcn(testloader, meanonly, homo, net, criterion, adj_torch, 
                                            spatial_torch, device, n_time, return_components=False, std=std)
        elif model_type == 'GAT':
            output_mean, output_var, _ = util_gat.testset_output_gat(testloader, meanonly, homo, net, criterion, adj_torch, 
                                            spatial_torch, device, n_time, return_components=False, std=std)
    else:
        net.train()
        criterion = MVELoss('norm_homo')
        output_mean = []
        
        for i in range(100):
            if model_type == 'GCN':
                om, ov, _ = util_gcnn.testset_output_gcn(testloader, meanonly, homo, net, criterion, adj_torch, 
                                                spatial_torch, device, n_time, return_components=False, std=std)
            elif model_type == 'GAT':
                om, ov, _ = util_gat.testset_output_gat(testloader, meanonly, homo, net, criterion, adj_torch, 
                                                spatial_torch, device, n_time, return_components=False, std=std)
            output_mean.append(om)
        
        output_var = None
        
    target = np.squeeze(data['y'][-1])

    return output_mean, output_var, target

def post_process_dist(dist, loc, scale):
    
    if dist == "lognorm":
        out_predict = np.exp(loc - np.power(scale,2))
#         out_std = np.mean(np.sqrt((np.exp(scale * scale)-1)*(np.exp(2*loc+scale * scale))))
    elif dist == 'tnorm':
        out_predict = loc
#         out_std = np.mean(scale)
    elif dist == 'laplace':
        out_predict = loc
#         out_std = np.mean(scale) * np.sqrt(2)
    elif dist == 'poisson':
        out_predict = loc
#         out_std = np.sqrt(loc)
    elif dist == 'norm':
        out_predict = loc
        
        
    return out_predict, None

def post_process_pi(dist, loc, scale, z):
    
    if dist == "lognorm":
#         predict = np.exp(loc - np.power(scale,2))
#         lb = np.exp(predict - z*scale)
#         ub = np.exp(predict + z*scale)
        lb, ub = lognorm.interval(z, loc, scale)
    elif dist == 'tnorm':
#         lb = np.max([0, loc - z*scale])
#         ub = loc + z*scale
        lb, ub = norm.interval(z, loc, scale)
        lb = lb * (lb>0)
    elif dist == 'laplace':
#         predict = loc
#         lb = predict + scale * np.log(2*z)
#         ub = predict - scale * np.log(2-2*(1-z))
        lb, ub = laplace.interval(z, loc, scale)
    elif dist == 'poisson':
        predict = loc
        lb,ub = poisson.interval(z, loc)   
    elif dist == 'norm':
        lb, ub = norm.interval(z, loc, scale)
        
    return lb, ub

def eval_theils(modelled, target, stdout=False):
    modelled = modelled.flatten()
    target = target.flatten()

    mse = np.mean(np.power(modelled - target, 2))
    denom = np.sqrt(np.mean(np.power(modelled, 2))) + np.sqrt(np.mean(np.power(target, 2)))

    u = np.sqrt(mse) / denom
    u_bias = np.power(np.mean(modelled) - np.mean(target), 2) / mse
    u_var = np.power(np.std(modelled) - np.std(target),2) / mse
    u_cov = 2*(1-np.corrcoef(modelled, target)[0,1])*np.std(modelled)*np.std(target) / mse
 
    assert np.round(u_bias+u_var+u_cov,2) == 1.0
    
    if stdout:
        print("Theil\'s U: %.6f" % (u))
        print('Bias: %.6f'% (u_bias))
        print('Variance: %.6f' % (u_var))
        print('Covariance: %.6f' % (u_cov))
  
    return u, u_bias, u_var, u_cov

def eval_pi(output_lower, output_upper, target, stdout=False):
    lower = output_lower.flatten()
    lower[lower<0] = 0
    upper = output_upper.flatten()
    t = target.flatten()
    kh = (np.sign(upper-t) >= 0) * (np.sign(t-lower) >= 0)
    picp = np.mean(kh)
    mpiw = np.sum((upper-lower) * kh) / np.sum(kh)
    
    if stdout:
        print("MPIW: %.6f" %(mpiw))
        print("PICP: %.6f"%(picp))

    return mpiw, picp

def eval_mean(modelled, target, dataset, stdout=False):
    mae = np.mean(np.abs(modelled.flatten() - target.flatten()))
    mape = mae / np.mean(target.flatten())
#     mse = np.mean(np.power(modelled.flatten() - target.flatten(), 2))
    if stdout:
        print("(" + dataset + ") Mean Absolute Error: %.3f (%.3f)" % (mae,mape))
#         print("(" + dataset + ") Mean Squared Error: %.3f" % mse)
    
    nonzero_mask = target.flatten() > 0
    pct_nonzeros = np.sum(nonzero_mask) / np.size(target) * 100
    if stdout:
        print("Percent Nonzeros: %d%%" % pct_nonzeros)

#     nz_mae = np.mean(np.abs(modelled.flatten()[nonzero_mask] - target.flatten()[nonzero_mask]))
#     nz_mse = np.mean(np.power(modelled.flatten()[nonzero_mask] - target.flatten()[nonzero_mask], 2))
#     if stdout:
#         print("Nonzero Entries:")
#         print("(" + dataset + ") Mean Absolute Error: %.3f" % nz_mae)
#         print("(" + dataset + ") Mean Squared Error: %.3f" % nz_mse)

    return mae, mape, pct_nonzeros


def eval_nll(criterion, modelled, modelled_std, target, stdout=False):
    modelled = torch.tensor(modelled)
    modelled_std = torch.tensor(modelled_std)
    target = torch.tensor(target)

    loss = criterion(modelled, modelled_std, target).item()/len(modelled)

    if stdout:
        print("NLL: %.6f" %(loss))

    return loss
