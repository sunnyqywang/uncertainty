import numpy as np
import pandas as pd

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
        if not all(np.isin(np.concatenate((x_times,x_times_ref)), ts)):
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

