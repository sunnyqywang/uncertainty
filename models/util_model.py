import numpy as np
import statsmodels.api as sm
import torch

def get_one_hot_tensor(t,ncat):
    t = t.view(-1,1)
    t_onehot = torch.FloatTensor(len(t), ncat)
    t_onehot = t_onehot.zero_()
    t_onehot.scatter_(1, t, 1)

    return t_onehot


def fit_wls(y,x):
    # fit OLS on mean (consistent but inefficient)
    ols_mean = sm.OLS(y,x)
    ols_mean_results = ols_mean.fit()
    ols_pred = ols_mean.predict(ols_mean_results.params, x)

    # fit OLS on transformed residuals (log transform -> std always positive)
    ols_res = np.log(np.power(y - ols_pred,2))
    ols_std = sm.OLS(ols_res, x)
    ols_std_results = ols_std.fit()

    return ols_mean, ols_mean_results.params, ols_std, ols_std_results.params

