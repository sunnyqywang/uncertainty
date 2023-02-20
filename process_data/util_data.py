import libpysal
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
import torch
from torch.utils.data import DataLoader

from class_dataset import CTA_Data

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def expand_input(x, exp_ref, columns):
    if columns is None:
        columns = np.arange(x.shape[1])
    x_exp = exp_ref * x[:,columns[0]][:, np.newaxis]
    for i in columns[1:]:
        x_exp = np.concatenate((x_exp, exp_ref * x[:,i][:,np.newaxis]), axis=1)
    return x_exp

def get_neighbour_gridcells(cell, dim, spatial_filter, directions):
    r,c = dim
    spatial_filter = spatial_filter.flatten()
    lookup = pd.DataFrame(np.array([spatial_filter,np.arange(r*c)]).T, columns=['busstop','index'])
    lookup['filtered_index'] = -1
    lookup.loc[spatial_filter,'filtered_index'] = np.arange(np.sum(spatial_filter))

    target = lookup[lookup['filtered_index'] == cell].iloc[0]['index']
    t_r = target // r
    t_c = target % r

    up = -1
    down = -1
    left = -1
    right = -1
    upleft = -1
    upright = -1
    downleft = -1
    downleft = -1

    if target-c >= 0:
        if spatial_filter[target-c]:
            up = np.sum(spatial_filter[:target-c])

    if target+c < r*c:
        if spatial_filter[target+c]:
            down = np.sum(spatial_filter[:target+c])

    if target-1 >= 0:
        if spatial_filter[target-1]:
            left = np.sum(spatial_filter[:target-1])

    if target+1 < r*c:
        if spatial_filter[target+1]:
            right = np.sum(spatial_filter[:target+1])

    if target-c-1 >= 0:
        if spatial_filter[target-c-1]:
            upleft = np.sum(spatial_filter[:target-c-1])
            
    if target-c+1 >= 0:
        if spatial_filter[target-c+1]:
            upright = np.sum(spatial_filter[:target-c+1])

    if target+c-1 < r*c:
        if spatial_filter[target+c-1]:
            downleft = np.sum(spatial_filter[:target+c-1])

    if target+c+1 < r*c:
        if spatial_filter[target+c+1]:
            downright = np.sum(spatial_filter[:target+c+1])
                
    if directions == 4:
        return up,down,left,right 
    elif directions == 8:
        return up,down,left,right,upleft,upright,downleft,downright



def get_reference(dfs, offset=672, diff=1):
    ''' 
    returns the difference (if diff==1) between the values and the reference value
    returns the reference value if diff==0
    default offset 672 (a week) i.e. between now and a week before
    takes dfs a list of dataframes to be differenced
    requires dataframes to have timestamps as index
    to be implemented: reference the average during this period
    '''

    dfs_new = []
    for df in dfs:
        dfs_new.append(df.copy())

    drop_list = []
    for t in dfs[0].index:
        try:
            for df,df_new in zip(dfs, dfs_new):
                df_new.loc[t] = diff*(df.loc[t] - 2*df.loc[t-offset]) + df.loc[t-offset]
        except:
            drop_list.append(t)

    for df_new in dfs_new:
        df_new.drop(index=drop_list, inplace=True)

    return dfs_new, drop_list

def generate_time_series(data, targets, others, ts, offset, lookback, ref_ts, difference, remove_list, remove_cascade=None):
    '''
    data, targets, others are [lists] of dataframes/ndarrays to be processed
    dataframes/arrays are of the shape (timestamps, spatial units (could be 1D-graph, or 2D-grid)
    ts is an array of time stamps that denote the timestamps of each row in data, target
    offset is the prediction horizon (the number of time stamps to skip between x and y, offset >=1)
    lookback is the number of time stamps (history) to include in x
    ref: the period to take difference/return
    difference: whether to take difference
    remove_list is a list/array of time stamps to remove from y (but can occur in x)
    remove_cascade is  a list/array of time stamps to be removed from x and y

    output x,y,others,ts
    x.shape=(ts, lookback, spatial units, #dfs in data)
    y.shape=(ts, spatial units, #dfs in targets)
    ts: the list of valid time stamps in x,y, and other
    '''

    for i in range(len(targets)):
        assert len(ts) == len(targets[i])

    x = [[] for i in range(len(data))]
    y = [[] for i in range(len(targets))]
    o = [[] for i in range(len(others))]
    ref = [[] for i in range(len(targets))]
    tt = []

    for t in ts:
        
        if t < ref_ts+offset:
            continue

        x_times = np.arange(t-offset-lookback+1, t-offset+1)
        x_times_ref = np.arange(t-ref_ts-offset-lookback+1, t-ref_ts-offset+1)
        y_times = np.arange(t-offset+1, t+1)
        y_times_ref = np.arange(t-ref_ts-offset+1, t-ref_ts+1)

        # if target timestamp is in the remove list, skip
        if t in remove_list:
            continue

        # if data overlaps with remove cascade list, skip
        if any(np.isin(x_times,remove_cascade)):
            continue

        # if times needed do not have values
        if not all(np.isin(np.concatenate((x_times,x_times_ref,y_times,y_times_ref)), ts)):
            continue

        if difference == 0:
            # return the raw values, not differenced
            for j in range(len(data)):
                x[j].append(np.array(data[j][np.isin(ts, x_times)]))
            for j in range(len(others)):
                o[j].append(np.array(others[j][np.isin(ts, t)]))
        else:
            # take the difference compared to reference
            # variables in x and others. not target (y).
            for j in range(len(data)):
                x[j].append(np.array(data[j][np.isin(ts, x_times)])-np.array(data[j][np.isin(ts, x_times_ref)]))
            for j in range(len(others)):
                o[j].append(np.array(others[j][np.isin(ts, t)]) - np.array(others[j][np.isin(ts, t-ref_ts)]))

        for j in range(len(targets)):
            y[j].append(np.array(targets[j][np.isin(ts, y_times)]))
            ref[j].append(np.array(targets[j][np.isin(ts, y_times_ref)]))
           
        tt.append(t)

    x = np.moveaxis(np.squeeze(np.array(x)),0,-1)                   

    
    return x, np.array(y), np.array(o), np.array(tt), np.array(ref)

def combine_datasources(project_dir, train_start_date, train_end_date, test_start_date, test_end_date, predict_hzn, time_size, difference, max_lookback, lookback, dataset):
    if difference:
        differenced='diff'
    else:
        differenced='raw'

    n_time = 96 // time_size

    ## I. Ridership
    with open(project_dir+"data/data_processed/"+dataset+"/"+\
              train_start_date+'_'+train_end_date+"_"+\
              str(predict_hzn)+"_"+str(time_size)+"_"+differenced+".pkl","rb") as f:
        x_train = pkl.load(f)[:,max_lookback-lookback:,:,:]
        ref_train = pkl.load(f)[:,-1:,:]
        los_train = pkl.load(f)[:,-1:,:]
        weather_train = pkl.load(f)
        y_train = pkl.load(f)[:,-1:,:]
        ts_train = pkl.load(f)
        station_id1 = pkl.load(f)

    with open(project_dir+"data/data_processed/"+dataset+"/"+\
              test_start_date+'_'+test_end_date+"_"+\
              str(predict_hzn)+"_"+str(time_size)+"_"+differenced+".pkl","rb") as f:
        x_test = pkl.load(f)[:,max_lookback-lookback:,:,:]
        ref_test = pkl.load(f)[:,-1:,:]
        los_test = pkl.load(f)[:,-1:,:]
        weather_test = pkl.load(f)
        y_test = pkl.load(f)[:,-1:,:]
        ts_test = pkl.load(f)
        station_id2 = pkl.load(f)

    (_, _, n_stations, n_modes) = x_train.shape

    ## II. Demo and POI
    spatial = pd.read_csv(project_dir+"data/data_processed/"+dataset+"/other/spatial.csv")

    if dataset == 'rail_catchment':
        spatial_id = 'STATION_ID'
        common_stations = np.intersect1d(np.array(station_id1), np.array(station_id2))
    else:
        spatial_id = 'GEOID10'
        with open(project_dir+"data/data_processed/select_tracts.pkl", "rb") as f:
            common_stations = pkl.load(f)
    
#     print("Number of Spatial Units Included:", len(common_stations))
    
    spatial = spatial[spatial[spatial_id].isin(common_stations)]
    spatial['pct_adults'] = spatial['pct25_34yrs']+spatial['pct35_50yrs']
    spatial = spatial[['tot_population','pct_adults','pctover65yrs',
        'pctPTcommute','avg_tt_to_work','inc_per_capita',
        'entertainment', 'restaurant', 'school', 'shop']]

    spatial = spatial.to_numpy(dtype=np.float64)
    # normalize the values that are not percentages
    for i in [0,4,5,6,7,8,9]:
        spatial[:,i] = (spatial[:,i] - np.mean(spatial[:,i]))/np.std(spatial[:,i])

    stations_mask1 = np.isin(np.array(station_id1), common_stations)
    stations_mask2 = np.isin(np.array(station_id2), common_stations)
    
    # Set validation set
    index_one_week = 7*n_time
    val_filter = ts_train >= (ts_train[-1]-index_one_week)

    x_val = x_train[val_filter][:,:, stations_mask1,:].copy()
    y_val = y_train[val_filter][:,:, stations_mask1].copy()
    ref_val = ref_train[val_filter][:,:,stations_mask1].copy()
    los_val = los_train[val_filter][:,:,stations_mask1].copy()
    qod_val = ts_train[val_filter].copy() % n_time
    weather_val = weather_train[val_filter].copy()
    ts_val = ts_train[val_filter].copy()
    
    x_train = x_train[~val_filter][:,:, stations_mask1,:]
    y_train = y_train[~val_filter][:,:, stations_mask1]
    ref_train = ref_train[~val_filter][:,:,stations_mask1]
    los_train = los_train[~val_filter][:,:,stations_mask1]
    qod_train = ts_train[~val_filter] % n_time
    weather_train = weather_train[~val_filter]
    ts_train = ts_train[~val_filter]
    
    x_test = x_test[:, :, stations_mask2,:]
    y_test = y_test[:, :, stations_mask2]
    ref_test = ref_test[:,:,stations_mask2]
    los_test = los_test[:,:,stations_mask2]
    qod_test = ts_test % n_time
    
    if dataset == 'rail_catchment':
        qod_train = qod_train - 6*(4//time_size)
        qod_val = qod_val - 6*(4//time_size)
        qod_test = qod_test - 6*(4//time_size)
    else:
        mask = qod_train < 3*(4//time_size)
        qod_train[~mask] = qod_train[~mask] - 5*(4//time_size)
        mask = qod_val < 3*(4//time_size)
        qod_val[~mask] = qod_val[~mask] - 5*(4//time_size)
        mask = qod_test < 3*(4//time_size)
        qod_test[~mask] = qod_test[~mask] - 5*(4//time_size)

    # III. Downtown Stations
    if dataset == 'rail_catchment':
        downtown_stations = pd.read_csv(project_dir+"data/data_processed/downtown_stations.csv")
        downtown_filter = np.isin(np.array(station_id1)[stations_mask1], downtown_stations['STATION_ID'])
        with open(project_dir+"data/data_processed/common_stations.pkl", "wb") as f:
            pkl.dump(common_stations, f)
    else:
        downtown_filter = None

    # IV. Adjacency Matrix
    adj = pd.read_csv(project_dir+"data/data_processed/"+dataset+"/other/adjlist.csv")
    
    if dataset == 'rail_catchment':
        adjcol = 'id'
    elif dataset == 'census_tract':
        adjcol = 'tract'
    stations_adj = adj['start_'+adjcol].drop_duplicates().to_numpy()
    
    w = {}
    for i in ['con', 'net', 'euc','func']:
        if i in adj.columns:
            w[i] = libpysal.weights.W.from_adjlist(adj,focal_col='start_'+adjcol,neighbor_col='end_'+adjcol,weight_col=i).full()[0]
        else:
#             print(i, "not available. Skipped...")
            pass

    # Filter to common stations only and calculate degree matrix and adj matrix for models
    deg = {}
    adj = {}
    stations_mask = np.isin(np.array(stations_adj), common_stations)
    
    for temp in w.keys():
        w[temp] = w[temp][stations_mask,:][:,stations_mask]
        w[temp] = w[temp] / np.max(w[temp])
        deg[temp] = np.sum(w[temp]+np.identity(len(common_stations)), axis=0)
        adj[temp] = np.matmul(np.matmul(scipy.linalg.fractional_matrix_power(np.diag(deg[temp]), -0.5), 
            w[temp]+np.identity(len(deg[temp]))),
            scipy.linalg.fractional_matrix_power(np.diag(deg[temp]), 0.5))
    
    data = {'x': [x_train,x_val,x_test],
            'y': [y_train,y_val,y_test], 
            'ref': [ref_train,ref_val,ref_test], 
            'los': [los_train,los_val,los_test], 
            'weather': [weather_train,weather_val,weather_test], 
            'qod': [qod_train,qod_val,qod_test],
            'ts': [ts_train,ts_val,ts_test],
            'stations': common_stations}

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
        
        trainset = CTA_Data(torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(ref_train), 
                torch.Tensor(weather_train), torch.Tensor(los_train), torch.LongTensor(qod_train))
        if bootstrap:
            sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
            trainloader = DataLoader(trainset, batch_size = 16, num_workers=10, sampler=sampler)
        else:
            trainloader = DataLoader(trainset, batch_size = 16, shuffle=True, num_workers=10)
        trainloader_test = DataLoader(trainset, batch_size = 8, shuffle=False, num_workers=10)
        
        if len(data['x']) == 3 and val == True:
            valset = CTA_Data(torch.Tensor(x_val), torch.Tensor(y_val), torch.Tensor(ref_val), 
                    torch.Tensor(weather_val), torch.Tensor(los_val), torch.LongTensor(qod_val))
            valloader = DataLoader(valset, batch_size = 8, shuffle=False, num_workers=10)
            y_val_eval = y_val

        testset = CTA_Data(torch.Tensor(x_test), torch.Tensor(y_test), torch.Tensor(ref_test), 
                torch.Tensor(weather_test), torch.Tensor(los_test), torch.LongTensor(qod_test))
        testloader = DataLoader(testset, batch_size = 8, shuffle=False, num_workers=10)

        adj_torch = torch.tensor([])
        for t in adj_type:
            if t in adj.keys():
                adj_torch = torch.cat((adj_torch, torch.Tensor(adj[t][:,:,np.newaxis])),dim=2)
            else:
#                 print(t, "not available. Skipped...")
                pass
        adj_torch = adj_torch.to(device)
        spatial_torch = torch.Tensor(spatial).to(device)

        y_train_eval = y_train
        y_test_eval = y_test

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
