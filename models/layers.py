import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphAttention(Module):

    def __init__(self, in_features, out_features, head=1, alpha=0.1):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.head = head

        # feature transform matrix
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features*head)))
        # attention vector
        self.a = Parameter(torch.empty(size=(2*out_features*head, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        (N,S,_) = h.shape
        # h.shape=(N,S,in_features) W.shape=(in_features,out_features*head) Wh.shape=(N,S,out_features*head)
        Wh = torch.matmul(h, self.W) 

        # a_input.shape=(N,S*S,2*out_features*head)
        a_input = torch.cat([Wh.repeat_interleave(S, dim=1), Wh.repeat(1,S,1)], dim=2)
        a_input = a_input.view(N,S,S,2*self.out_features*self.head)

        # a_input.shape=(N,S,S,2*out_features*head) a.shape=(2*out_features*head,1) e.shape=(N,S,S)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # attention.shape=(N,S,S)
        attention = F.softmax(e, dim=2)
        # output.shape=(N,S,out_features*head)
        output = torch.matmul(attention, Wh)
        output = torch.stack(torch.split(output, torch.tensor(self.out_features), dim=2))
        # output.shape=(N,S,out_features)
        output = F.elu(torch.mean(output, axis=0))

        return output

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
