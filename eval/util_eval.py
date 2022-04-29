import numpy as np
from scipy.stats import poisson, norm, laplace, lognorm
import torch

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
    mse = np.mean(np.power(modelled.flatten() - target.flatten(), 2))
    if stdout:
        print("(" + dataset + ") Mean Absolute Error: %.3f" % mae)
        print("(" + dataset + ") Mean Squared Error: %.3f" % mse)
    
    nonzero_mask = target.flatten() > 0
    pct_nonzeros = np.sum(nonzero_mask) / np.size(target) * 100
    if stdout:
        print("Percent Nonzeros: %d%%" % pct_nonzeros)

    nz_mae = np.mean(np.abs(modelled.flatten()[nonzero_mask] - target.flatten()[nonzero_mask]))
    nz_mse = np.mean(np.power(modelled.flatten()[nonzero_mask] - target.flatten()[nonzero_mask], 2))
    if stdout:
        print("Nonzero Entries:")
        print("(" + dataset + ") Mean Absolute Error: %.3f" % nz_mae)
        print("(" + dataset + ") Mean Squared Error: %.3f" % nz_mse)

    return mae, mse, nz_mae, nz_mse, pct_nonzeros


def eval_nll(criterion, modelled, modelled_std, target, stdout=False):
    modelled = torch.tensor(modelled)
    modelled_std = torch.tensor(modelled_std)
    target = torch.tensor(target)

    loss = criterion(modelled, modelled_std, target).item()/len(modelled)

    if stdout:
        print("NLL: %.6f" %(loss))

    return loss
