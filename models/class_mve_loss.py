import torch
import torch.nn as nn
import torch.nn.functional as F
from TruncatedNormal import TruncatedNormal


class MVELoss(nn.Module):
    def __init__(self,dist):
        super().__init__()
        self.dist = dist
        
    def forward(self, output_loc, output_scale=None, target=None):
        
        loc = torch.flatten(output_loc)
        if output_scale is not None:
            scale = torch.flatten(output_scale)
        t = torch.flatten(target)
        # try:
        if self.dist == 'laplace':
            d = torch.distributions.laplace.Laplace(loc, scale)
            loss = d.log_prob(t)

        elif self.dist == 'tnorm':
            d = torch.distributions.normal.Normal(loc, scale)
            prob0 = d.cdf(torch.Tensor([0]).to(target.device))
            loss = d.log_prob(t) - torch.log(1-prob0)

        elif self.dist == 'lognorm':
            d = torch.distributions.log_normal.LogNormal(loc, scale)
            loss = d.log_prob(t+0.000001)
            
        elif self.dist == 'poisson':
            d = torch.distributions.poisson.Poisson(loc)
            loss = d.log_prob(t)
        
        elif (self.dist == 'norm') | (self.dist == 'norm_homo'):
            d = torch.distributions.normal.Normal(loc, scale)
            loss = d.log_prob(t)
            
        elif self.dist == 'nb':
            
            def nb_nll_loss(y,n,p):
                """
                y: true values
                y_mask: whether missing mask is given
                """
                nll = torch.lgamma(n) + torch.lgamma(y+1) - torch.lgamma(n+y) - n*torch.log(p) - y*torch.log(1-p)
                return torch.sum(nll)
            
            loss = nb_nll_loss(t, loc, scale) # check scale constraints
        
        elif self.dist == 'zinb':
            def nb_zeroinflated_nll_loss(y,n,p,pi):
                """
                y: true values
                https://stats.idre.ucla.edu/r/dae/zinb/
                """
                idx_yeq0 = y==0
                idx_yg0  = y>0

                n_yeq0 = n[idx_yeq0]
                p_yeq0 = p[idx_yeq0]
                pi_yeq0 = pi[idx_yeq0]
                yeq0 = y[idx_yeq0]

                n_yg0 = n[idx_yg0]
                p_yg0 = p[idx_yg0]
                pi_yg0 = pi[idx_yg0]
                yg0 = y[idx_yg0]

                L_yeq0 = torch.log(pi_yeq0) + torch.log((1-pi_yeq0)*torch.pow(p_yeq0,n_yeq0))
                L_yg0  = torch.log(1-pi_yg0) + torch.lgamma(n_yg0+yg0) - torch.lgamma(yg0+1) - torch.lgamma(n_yg0) + n_yg0*torch.log(p_yg0) + yg0*torch.log(1-p_yg0)

                return -torch.sum(L_yeq0)-torch.sum(L_yg0)
            
            loss = nb_zeroinflated_nll_loss(t, loc, scale, pi) ## get another pi estimate
            
        else:
            print("Dist error")
            return 0
            
        # except:
            # print(loc)
            # print(scale)

       
        return -torch.sum(loss)
        
