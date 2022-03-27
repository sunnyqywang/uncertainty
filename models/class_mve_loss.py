import torch
import torch.nn as nn
import torch.nn.functional as F
from TruncatedNormal import TruncatedNormal

class LN_MVELoss(nn.Module):
    # mean-variance loss with lognormal distribution

    def __init__(self,):
        super().__init__()

    def forward(self, output_mean, output_var, target):
        mean = torch.flatten(output_mean)
        std = torch.flatten(output_var)
        t = torch.flatten(target)
        try:
            dist = torch.distributions.log_normal.LogNormal(mean, std)
        except:
            print(mean)
        loss = dist.log_prob(t+0.000001)

        if torch.sum(torch.isnan(loss)) != 0:
                print(mean[torch.isnan(loss)][0])
                print(std[torch.isnan(loss)][0])
                print(t[torch.isnan(loss)][0])
        
        return -torch.sum(loss)


class T_MVELoss(nn.Module):
    # mean-variance loss with truncated normal (left at 0) distribution

    def __init__(self,):
        super().__init__()

    def forward(self, output_mean, output_var, target):
        mean = torch.flatten(output_mean)
        std = torch.flatten(output_var)
        t = torch.flatten(target)

        try:
            dist = torch.distributions.normal.Normal(mean, std)
        except:
            print('loss', mean)
        prob0 = dist.cdf(torch.Tensor([0]).to(target.device))
        loss = dist.log_prob(t) - torch.log(1-prob0)
        '''
        dist = TruncatedNormal(a=0, b=float("Inf"), loc=mean, scale=std)
        loss = dist.log_prob(t)
        '''
        if torch.sum(torch.isnan(loss)) != 0:
                print(mean[torch.isnan(loss)])
                print(std[torch.isnan(loss)])
                print(t[torch.isnan(loss)])

        return -torch.sum(loss)
