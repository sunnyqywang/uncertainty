import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import pickle as pkl

from class_mve_loss import MVELoss
from default_parameters import *
import util_eval
import util_gcnn

def calc_nll_benchmark(project_dir, out_folder, period, **kwargs):

    if 'val' in kwargs:
        val=kwargs['val']
    else:
        val=True

    data, adj, spatial, downtown_filter = util_gcnn.combine_datasources(project_dir, period, predict_hzn, time_size, difference, max_lookback, lookback)
    nstations = np.sum(downtown_filter)

    _, _, _, _, _, y_train_eval, y_test_eval = util_gcnn.prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, val=val)
    
    criterion = T_MVELoss()
    with open(project_dir+'results/'+out_folder+'/'+period+'_dt_benchmark.pkl', 'rb') as f:
        ref_test = pkl.load(f)
        emp_std = pkl.load(f)
        wls_pred = pkl.load(f)
        wls_pred_std = pkl.load(f)
        
    valid1 = emp_std.flatten() > 0
    valid2 = wls_pred.flatten() > -400

    nll_ref = util_eval.eval_nll(criterion, ref_test.flatten()[valid1], emp_std.flatten()[valid1], y_test_eval.flatten()[valid1], stdout=False)
    nll_wls = util_eval.eval_nll(criterion, wls_pred.flatten()[valid2], wls_pred_std.flatten()[valid2], y_test_eval.flatten()[valid2], stdout=False)

    return nll_ref*nstations, nll_wls*nstations
    
def get_bmk_results(project_dir, out_folder, metrics, filt):

    bmk = pd.read_csv(project_dir+"results/"+out_folder+"rail_catchment_benchmark_results.csv")
    bmk['train_rmse'] = np.sqrt(bmk['train_mse'])
    bmk['test_rmse'] = np.sqrt(bmk['test_mse'])
    bmk = bmk[bmk['extent']=='downtown']

    for k,v in filt.items():
        bmk = bmk[bmk[k].isin([np.nan, v])]
    
    bmk_before = {}
    bmk_after = {}
    for k in metrics:
        if k == 'nll_loss':
            bmk_before['az_'+k] = np.nan
            bmk_after['az_'+k] = np.nan
            bmk_before['lw_'+k],bmk_before['wls_'+k] = calc_nll_benchmark(project_dir, out_folder, period='before', val=False)
            bmk_after['lw_'+k],bmk_after['wls_'+k] = calc_nll_benchmark(project_dir, out_folder, period='after', val=False)
            
            continue
            
        if len(bmk[(bmk['period']=='before')&(bmk['model']=='all zeros')]['test_'+k])>1:
            print('More than 1 model left. Selecting the most recent one.')
        
        bmk_before['az_'+k] = bmk[(bmk['period']=='before')&(bmk['model']=='all zeros')]['test_'+k].iloc[-1]
        bmk_before['lw_'+k] = bmk[(bmk['period']=='before')&(bmk['model']=='last week')]['test_'+k].iloc[-1]
        bmk_before['wls_'+k] = bmk[(bmk['period']=='before')&(bmk['model']=='WLS5')]['test_'+k].iloc[-1]

        bmk_after['az_'+k] = bmk[(bmk['period']=='after')&(bmk['model']=='all zeros')]['test_'+k].iloc[-1]
        bmk_after['lw_'+k] = bmk[(bmk['period']=='after')&(bmk['model']=='last week')]['test_'+k].iloc[-1]
        bmk_after['wls_'+k] = bmk[(bmk['period']=='after')&(bmk['model']=='WLS5')]['test_'+k].iloc[-1]


    return bmk_before, bmk_after
