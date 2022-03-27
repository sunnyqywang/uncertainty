import sys
sys.path.append("../")

from class_dataset import CTA_Data
from class_gcn_lstm import GCN_LSTM
from class_mve_loss import T_MVELoss
import util_eval

import glob
import libpysal
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
import torch
from torch.utils.data import DataLoader

def combine_datasources(project_dir, period, predict_hzn, time_size, difference, max_lookback, lookback):
    if difference:
        differenced='diff'
    else:
        differenced='raw'

    n_time = 96 // time_size

    ## I. Ridership
    with open(project_dir+"data/data_processed/rail_catchment/"+period+"/"+
            period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_train.pkl","rb") as f:
        x_train = pkl.load(f)[:,max_lookback-lookback:,:,:]
        ref_train = pkl.load(f)[:,-1:,:]
        los_train = pkl.load(f)[:,-1:,:]
        weather_train = pkl.load(f)
        y_train = pkl.load(f)[:,-1:,:]
        ts_train = pkl.load(f)
        station_id = pkl.load(f)

    with open(project_dir+"data/data_processed/rail_catchment/"+period+"/"+
            period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_test.pkl","rb") as f:
        x_test = pkl.load(f)[:,max_lookback-lookback:,:,:]
        ref_test = pkl.load(f)[:,-1:,:]
        los_test = pkl.load(f)[:,-1:,:]
        weather_test = pkl.load(f)
        y_test = pkl.load(f)[:,-1:,:]
        ts_test = pkl.load(f)

    fl= glob.glob(project_dir+"data/data_processed/rail_catchment/"+period+"/"+
            period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_val.pkl")

    if len(fl) == 1 :
        with open(project_dir+"data/data_processed/rail_catchment/"+period+"/"+
                period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_val.pkl","rb") as f:
            x_val = pkl.load(f)[:,max_lookback-lookback:,:,:]
            ref_val = pkl.load(f)[:,-1:,:]
            los_val = pkl.load(f)[:,-1:,:]
            weather_val = pkl.load(f)
            y_val = pkl.load(f)[:,-1:,:]
            ts_val = pkl.load(f)
        qod_val = ts_val % n_time

    (_, _, n_stations, n_modes) = x_train.shape
    qod_train = ts_train % n_time
    qod_test = ts_test % n_time
    common_stations = np.array(station_id)

    ## II. Demo and POI
    spatial = pd.read_csv(project_dir+"data/data_processed/rail_catchment/spatial.csv")

    #print("All Available Columns:\n", spatial.columns.tolist())

    spatial['pct_adults'] = spatial['pct25_34yrs']+spatial['pct35_50yrs']
    spatial = spatial[['STATION_ID','tot_population','pct_adults','pctover65yrs',
        'pctPTcommute','avg_tt_to_work','inc_per_capita',
        'entertainment', 'restaurant', 'school', 'shop']]

    #print("\n\nColumns included:\n", spatial.columns.tolist())
    spatial = spatial.to_numpy()

    # update station selections
    common_stations = np.intersect1d(common_stations, spatial[:,0])

    stations_mask = np.isin(spatial[:,0], common_stations)
    spatial = spatial[stations_mask,1:]
    # normalize the values that are not percentages
    for i in [0,4,5,6,7,8,9]:
        spatial[:,i] = (spatial[:,i] - np.mean(spatial[:,i]))/np.std(spatial[:,i])

    stations_mask = np.isin(np.array(station_id), common_stations)
    x_train = x_train[:, :, stations_mask,:]
    y_train = y_train[:, :, stations_mask]
    x_test = x_test[:, :, stations_mask,:]
    y_test = y_test[:, :, stations_mask]
    ref_train = ref_train[:,:,stations_mask]
    ref_test = ref_test[:,:,stations_mask]
    los_train = los_train[:,:,stations_mask]
    los_test = los_test[:,:,stations_mask]
    if len(fl) == 1:
        x_val = x_val[:, :, stations_mask,:]
        y_val = y_val[:, :, stations_mask]
        ref_val = ref_val[:,:,stations_mask]
        los_val = los_val[:,:,stations_mask]

    '''
    x_mean = np.mean(x_train)
    x_std = np.std(x_train.flatten())
    x_train = (x_train - x_mean)/x_std
    x_test = (x_test - x_mean)/x_std
    '''

    #print("Number of Stations Included:", len(common_stations))
    with open(project_dir+"data/data_processed/common_stations.pkl", "wb") as f:
        pkl.dump(common_stations, f)

    # III. Downtown Stations
    downtown_stations = pd.read_csv(project_dir+"data/data_processed/downtown_stations.csv")
    downtown_filter = np.isin(np.array(station_id)[stations_mask], downtown_stations['STATION_ID'])

    # IV. Adjacency Matrix
    adj = pd.read_csv(project_dir+"data/data_processed/rail_catchment/rail_adjlist.csv")

    w = {"con":libpysal.weights.W.from_adjlist(adj,focal_col='start_id',neighbor_col='end_id',weight_col='conn').full()[0], 
            "net":libpysal.weights.W.from_adjlist(adj,focal_col='start_id',neighbor_col='end_id',weight_col='network_dist').full()[0],
            "euc":libpysal.weights.W.from_adjlist(adj,focal_col='start_id',neighbor_col='end_id',weight_col='euc_dist').full()[0],
            "func":libpysal.weights.W.from_adjlist(adj,focal_col='start_id',neighbor_col='end_id',weight_col='func_sim').full()[0]}

    # Filter to common stations only and calculate degree matrix and adj matrix for models
    deg = {}
    adj = {}
    stations_mask = np.isin(np.array(station_id), common_stations)
    for temp in w.keys():
        w[temp] = w[temp][stations_mask,:][:,stations_mask]
        w[temp] = w[temp] / np.max(w[temp])
        deg[temp] = np.sum(w[temp]+np.identity(len(common_stations)), axis=0)
        adj[temp] = np.matmul(np.matmul(scipy.linalg.fractional_matrix_power(np.diag(deg[temp]), -0.5), 
            w[temp]+np.identity(len(deg[temp]))),
            scipy.linalg.fractional_matrix_power(np.diag(deg[temp]), 0.5))

    if len(fl) == 1:
        data = {'x': [x_train,x_val,x_test],
                'y': [y_train,y_val,y_test], 
                'ref': [ref_train,ref_val,ref_test], 
                'los': [los_train,los_val,los_test], 
                'weather': [weather_train, weather_val,weather_test], 
                'qod': [qod_train,qod_val,qod_test],
                'ts': [ts_train,ts_val,ts_test],
                'stations': np.array(station_id)[stations_mask][downtown_filter]}
    else:
        data = {'x': [x_train,x_test],
                'y': [y_train,y_test], 
                'ref': [ref_train, ref_test], 
                'los': [los_train, los_test], 
                'weather': [weather_train, weather_test], 
                'qod': [qod_train, qod_test],
                'ts': [ts_train, ts_test],
                'stations': np.array(station_id)[stations_mask][downtown_filter]}

    return data, adj, spatial, downtown_filter


def prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, val=True, bootstrap=False):

    if len(data['x']) == 2:
        # there is no validation set
        [x_train,x_test] = data['x']
        [y_train,y_test] = data['y'] 
        [ref_train, ref_test] = data['ref'] 
        [los_train, los_test] = data['los']
        [weather_train, weather_test] = data['weather'] 
        [qod_train, qod_test] = data['qod']
    elif val == False:
        # there is a validation set but we are not using it
        [x_train,_,x_test] = data['x']
        [y_train,_,y_test] = data['y'] 
        [ref_train,_, ref_test] = data['ref'] 
        [los_train,_, los_test] = data['los']
        [weather_train,_, weather_test] = data['weather'] 
        [qod_train,_, qod_test] = data['qod']
    elif len(data['x'])==3:
        [x_train,x_val,x_test] = data['x']
        [y_train,y_val,y_test] = data['y'] 
        [ref_train,ref_val,ref_test] = data['ref'] 
        [los_train,los_val,los_test] = data['los']
        [weather_train,weather_val,weather_test] = data['weather'] 
        [qod_train,qod_val,qod_test] = data['qod']

   
    if train_extent == 'all':
        # validation set not implemented
        n_stations = len(common_stations)
        trainset = CTA_Data(torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(ref_train), 
                torch.Tensor(weather_train), torch.Tensor(los_train), torch.LongTensor(qod_train))
        if bootstrap:
            sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
            trainloader = DataLoader(trainset, batch_size = 256, num_workers=10, sampler=sampler)
        else:
            trainloader = DataLoader(trainset, batch_size = 256, shuffle=True, num_workers=10)
        trainloader_test = DataLoader(trainset, batch_size = 256, shuffle=False, num_workers=10)

        testset = CTA_Data(torch.Tensor(x_test), torch.Tensor(y_test), torch.Tensor(ref_test), 
                torch.Tensor(weather_test), torch.Tensor(los_test), torch.LongTensor(qod_test))
        testloader = DataLoader(testset, batch_size = len(y_test), shuffle=False, num_workers=10)

        adj_torch = torch.tensor([])
        for t in adj_type:
            adj_torch = torch.cat((adj_torch, torch.Tensor(adj[t][:,:,np.newaxis])),dim=2)
        adj_torch = adj_torch.to(device)
        spatial_torch = torch.Tensor(spatial).to(device)

        y_train_eval = y_train
        y_test_eval = y_test
        y_val_eval = y_val

    elif train_extent == 'downtown':
        n_stations = np.sum(downtown_filter)
        trainset = CTA_Data(torch.Tensor(x_train[:,:,downtown_filter,:]), torch.Tensor(y_train[:,:,downtown_filter]), 
                torch.Tensor(ref_train[:,:,downtown_filter]), 
                torch.Tensor(weather_train), torch.Tensor(los_train[:,:,downtown_filter]), 
                torch.LongTensor(qod_train))
        if bootstrap:
            sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
            trainloader = DataLoader(trainset, batch_size = 64, num_workers=10, sampler=sampler)
        else:
            trainloader = DataLoader(trainset, batch_size = 64, shuffle=True, num_workers=10)
        trainloader_test = DataLoader(trainset, batch_size = 64, shuffle=False, num_workers=10)

        if len(data['x']) == 3 and val == True:
            valset = CTA_Data(torch.Tensor(x_val[:,:,downtown_filter,:]), torch.Tensor(y_val[:,:,downtown_filter]), 
                    torch.Tensor(ref_val[:,:,downtown_filter]), 
                    torch.Tensor(weather_val), torch.Tensor(los_val[:,:,downtown_filter]), 
                    torch.LongTensor(qod_val))
            valloader = DataLoader(valset, batch_size = len(y_val), shuffle=False, num_workers=10)
            y_val_eval = y_val[:,:,downtown_filter]

        testset = CTA_Data(torch.Tensor(x_test[:,:,downtown_filter,:]), torch.Tensor(y_test[:,:,downtown_filter]), 
                torch.Tensor(ref_test[:,:,downtown_filter]), 
                torch.Tensor(weather_test), torch.Tensor(los_test[:,:,downtown_filter]), 
                torch.LongTensor(qod_test))
        testloader = DataLoader(testset, batch_size = len(y_test), shuffle=False, num_workers=10)

        adj_torch = torch.tensor([])
        for t in adj_type:
            adj_torch = torch.cat((adj_torch, torch.Tensor(adj[t][downtown_filter,:][:,downtown_filter][:,:,np.newaxis])),dim=2)
        adj_torch = adj_torch.to(device)
        spatial_torch = torch.Tensor(spatial[downtown_filter]).to(device)

        y_train_eval = y_train[:,:,downtown_filter]
        y_test_eval = y_test[:,:,downtown_filter]

    else:
        print("Error")
        return

    if len(data['x']) == 2 or val == False:
        return  trainloader, trainloader_test, testloader, adj_torch, spatial_torch, y_train_eval, y_test_eval
    else:
        return  trainloader, trainloader_test, valloader, testloader, adj_torch, spatial_torch, y_train_eval, y_val_eval, y_test_eval

def testset_output_gat(testloader, meanonly, net, criterion, adj, demo, device, n_time):
    loss = 0
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
        outputs = net(batch_x, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot)

        # loss
        if meanonly:
            loss += criterion(outputs, batch_y).item()
        else:
            loss += criterion(outputs[:batch_size,:], outputs[batch_size:,:], batch_y).item()

        if meanonly:
            if i == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)

        else:
            if i == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
                test_out_var = outputs[batch_size:,:].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)
                test_out_var = np.concatenate((test_out_var, outputs[batch_size:,:].cpu().detach().numpy()), axis=0)

    if meanonly:
        return test_out_mean,loss
    else:
        return test_out_mean, test_out_var, loss

def testset_output_gcn(testloader, meanonly, homo, net, criterion, adj, demo, device, n_time, return_components=False):
    net.eval()
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
        batch_qod_onehot.scatter_(1, batch_qod-6, 1)
        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod_onehot = batch_x.to(device),batch_y.to(device), batch_history.to(device),batch_weather.to(device), batch_los.to(device), batch_qod_onehot.to(device)

        # forward
        if return_components:
            outputs = net(batch_x, None, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot, 1, True)
            return outputs
        else:
            outputs = net(batch_x, None, adj, batch_history, demo, batch_weather, batch_los, batch_qod_onehot, 1)

        # loss
        if meanonly:
            loss += criterion(outputs, batch_y).item()
        elif homo:
            loss += criterion(outputs[0], outputs[1], batch_y).item()
        else:
            loss += criterion(outputs[:batch_size,:], outputs[batch_size:,:], batch_y).item()

        if meanonly:
            if i == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)

        elif homo:
            if i == 0:
                test_out_mean = outputs[0][:batch_size,:].cpu().detach().numpy()
                test_out_var = outputs[1].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[0][:batch_size,:].cpu().detach().numpy()), axis=0)
        else:
            if i == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
                test_out_var = outputs[batch_size:,:].cpu().detach().numpy()
            else:
                test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)
                test_out_var = np.concatenate((test_out_var, outputs[batch_size:,:].cpu().detach().numpy()), axis=0)

    if meanonly:
        return test_out_mean, None, loss
    else:
        return test_out_mean, test_out_var, loss

def load_model(project_dir, out_folder, period, train_extent, adj_type, predict_hzn, time_size, lookback, ii, n_modes, n_stations, n_time):
    if type(adj_type) == str:
         adj_type=adj_type.replace('_', '-')
         nadj = adj_type.count('-')+1
    else:
         nadj = len(adj_type)
         adj_type='-'.join(adj_type)

    file = glob.glob(project_dir+"models/"+out_folder+"/"+period+"_"+train_extent+"_"+adj_type+"_"+
            str(predict_hzn)+"_"+str(time_size)+"_"+str(lookback)+"_"+str(ii)+"_*.pt")

    if len(file) == 0:
        print("Model %d not saved." % (ii))
        return None

    try:
        assert len(file)==1
    except:
        print("Multiple Files Found!")
        for f in file:
            print(f)

    saved = torch.load(file[0])
    if len(saved['hyperparameters']) == 14:
        (_,_,_,_,_,_,_,_,dropout,n_hid_units,nlstm,ngc,weight_decay,_) = saved['hyperparameters']
    else:
        (_,_,_,_,_,_,_,_,dropout,n_hid_units,nlstm,ngc,weight_decay) = saved['hyperparameters']

    # assuming that meanonly and homoskedastic modelswill not be loaded
    net = GCN_LSTM(meanonly=False, homo=False, nadj = nadj, nmode=n_modes, nstation=n_stations, ntime=n_time, ndemo=0,
            nhid_g=n_hid_units, ngc=ngc, nhid_l=n_hid_units, nlstm=nlstm, 
            nhid_fc=n_hid_units, dropout=dropout)
    net.load_state_dict(saved['model_state_dict'])

    return net
 

def ensemble(project_dir, out_folder, period, predict_hzn, time_size, lookback, ensemble_model_numbers, device, train_extent, adj_type, z, data, adj, spatial, downtown_filter):

    n_time = 96//time_size-7

    trainloader, trainloader_test, testloader, adj_torch, spatial_torch, y_train_eval, y_test_eval = \
            prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, val=False)

    (num_train, _, _, n_modes) = data['x'][0].shape
    (num_test, _, _, _) = data['x'][-1].shape
    n_stations = adj_torch.shape[0]

    y_train_eval = np.squeeze(y_train_eval)
    y_test_eval = np.squeeze(y_test_eval)

    mean_list = []
    std_list = []
    mae_list = []
    mse_list = []
    picp_list = []
    mpiw_list = []
    test_loss_list = []
    u_list = []
    ub_list = []
    uv_list = []
    uc_list = []
    spatial_torch = None

    for ii in ensemble_model_numbers:

        net = load_model(project_dir, out_folder, period, train_extent, adj_type, predict_hzn, time_size, lookback, ii, n_modes, n_stations, n_time)
        
        if net is None:
            continue

        criterion = T_MVELoss()
        net.eval()

        '''
        train_out_mean, train_out_std, train_loss = util_gcnn.testset_output_gcn(trainloader_test, False, net, criterion, 
                                                                                 adj_torch, spatial_torch, device, n_time)
        tr_mae, tr_mse, _, _, _ = util_eval.eval_mean(train_out_mean, y_train_eval, 'Train')
        tr_u, tr_ub, tr_uv, tr_uc = util_eval.eval_theils(np.squeeze(train_out_mean), y_train, stdout = False)
        tr_mpiw, tr_picp = util_eval.eval_pi(train_out_mean - z*train_out_std, train_out_mean + z*train_out_std, y_train_eval)
        '''

        test_out_mean, test_out_std, test_loss = testset_output_gcn(testloader, False, False, net, criterion, 
                adj_torch, spatial_torch, device, n_time)
        mae, mse, _, _, _ = util_eval.eval_mean(test_out_mean, y_test_eval, 'Test')
        u, ub, uv, uc = util_eval.eval_theils(np.squeeze(test_out_mean), y_test_eval, stdout = False)
        mpiw, picp = util_eval.eval_pi(test_out_mean - z*test_out_std, test_out_mean + z*test_out_std, y_test_eval)

        mean_list.append(test_out_mean)
        std_list.append(test_out_std)
        test_loss_list.append(test_loss)

        mae_list.append(mae)
        mse_list.append(mse)

        u_list.append(u)
        ub_list.append(ub)
        uv_list.append(uv)
        uc_list.append(uc)

        mpiw_list.append(mpiw)
        picp_list.append(picp)

    ## Fill in Missing U values (only for the first few runs where U not implemented)
    # df.loc[(df['Period'] == period) & (df['Lookback'] == lookback) & (df['Adjacency']=='_'.join(adj_type)) & (np.isnan(df['u'])),'u'] = u_list
    # df.loc[(df['Period'] == period) & (df['Lookback'] == lookback) & (df['Adjacency']=='_'.join(adj_type)) & (np.isnan(df['um'])),'um'] = ub_list
    # df.loc[(df['Period'] == period) & (df['Lookback'] == lookback) & (df['Adjacency']=='_'.join(adj_type)) & (np.isnan(df['uc'])),'uc'] = uc_list
    # df.loc[(df['Period'] == period) & (df['Lookback'] == lookback) & (df['Adjacency']=='_'.join(adj_type)) & (np.isnan(df['us'])),'us'] = uv_list

    # df.to_csv(project_dir+"results/rail_catchment_mve_results.csv", index=False)

    # calculate ensembled stats                     
    test_ens_mean = np.mean(np.array(mean_list), axis=0)
    test_ens_std = np.sqrt(np.mean(np.power(np.array(mean_list),2) \
            + np.power(np.array(std_list),2), axis=0)\
            - np.power(test_ens_mean,2))

    return data['ts'][1], y_test_eval, test_ens_mean, test_ens_std, mean_list, std_list

