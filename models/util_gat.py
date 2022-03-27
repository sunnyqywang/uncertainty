import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_attention(h, adj, W, a):
    (N,S,_) = h.shape
    _, out_features = W.shape
    
    # h.shape=(N,S,in_features) W.shape=(in_features,out_features) Wh.shape=(N,S,out_features)
    Wh = torch.matmul(h, W) 

    # a_input.shape=(N,S*S,2*out_features)
    a_input = torch.cat([Wh.repeat_interleave(S, dim=1), Wh.repeat(1,S,1)], dim=2)
    a_input = a_input.view(N,S,S,2*out_features)

    # a_input.shape=(N,S*S,2*out_features) a.shape=(2*out_features,1)
    e = nn.LeakyReLU(0.1)(torch.matmul(a_input, a).squeeze(3))

    attention = F.softmax(e, dim=2)

    return attention


