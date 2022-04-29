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
        try:
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
                
            else:
                print("Dist error")
                return 0
                
        except:
            print(loc)

        if torch.sum(torch.isnan(loss)) != 0:
            print(loc[torch.isnan(loss)][0])
            print(scale[torch.isnan(loss)][0])
            print(t[torch.isnan(loss)][0])
            
#         if -loss.reshape(len(output_loc),-1) > 100:
#         i = torch.argmax(-loss)
#         print(loc[i].detach().cpu().numpy(),scale[i].detach().cpu().numpy(), t[i].detach().cpu().numpy(), loss[i].detach().cpu().numpy())
#         print(torch.median(-loss))
        
        return -torch.sum(loss)
        
