import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttention
from layers import GraphConvolution

class GAT_LSTM(nn.Module):
    def __init__(self, meanonly, nadj, nmode, nstation, ntime, ndemo, nhead, nhid_g, nga, nhid_l, nlstm, nhid_fc, dropout):
        # meanonly: bool, whether output is mean only or mean and standard deviation
        # nmode: number of modes (features) in x (ridership immediately before)
        # nstation: number of spatial units
        # ntime: number of time periods in a day
        # ndemo: number of time-independent quantities (all collapsed into demo)
        # nhid_[g,l,fc]: number of hidden (output) units of the graph/lstm/fc layers
        # nga: number of graph attention layers
        # nlstm: number of lstm layers
        # dropout
        # needs implementation: regularization

        super().__init__()

        self.meanonly = meanonly
        self.nhid_g = nhid_g
        self.nhead = nhead
        self.nadj = nadj
        self.ntime = ntime
        self.nga = nga
        
        # batchnorm1d: over 3D input of size (N,C,L); num_features = C
        self.batchnorm = nn.BatchNorm1d(num_features=nstation)
        self.dropout = dropout

        # Ridership immediately before
        self.gat = []
        for i in range(nadj):
            innerlist = []
            for j in range(nga):
                if j == 0:
                    innerlist.append(GraphAttention(nmode, nhid_g, nhead))
                else:
                    innerlist.append(GraphAttention(nhid_g, nhid_g))
            self.gat.append(nn.ModuleList(innerlist))
        self.gat = nn.ModuleList(self.gat)

        self.layernorm = []
        for j in range(nga):
            self.layernorm.append(nn.LayerNorm(nhid_g))
        self.layernorm = nn.ModuleList(self.layernorm)

        self.fc1 = nn.Linear(nhid_g*nstation, nhid_fc)
        self.fc2 = nn.Linear(nhid_fc, nhid_g)
        self.lstm = nn.LSTM(input_size=int(nhid_g), hidden_size=int(nhid_l), num_layers=nlstm, batch_first=True)

        # History
        self.fc3 = nn.Linear(nhid_l, nhid_fc)
        self.history_weights_mean = nn.Parameter(torch.rand((ntime, nstation))/2+0.5)

        # Weather
        self.weather_weights_mean = nn.Parameter(torch.rand((ntime, 2*nstation)))

        # Level of Service
        self.los_weights_mean = nn.Parameter(torch.rand(ntime, nstation))

        if meanonly:
           self.gcs = GraphConvolution(ndemo, ntime)
           self.final = nn.Linear(nhid_fc, nstation)
        else:
            self.weather_weights_var = nn.Parameter(torch.rand((ntime, 2*nstation)))
            self.history_weights_var = nn.Parameter(torch.rand((ntime, nstation))/2+0.5)
            # layers bringing everything together
            self.final = nn.Linear(nhid_fc, nstation*2)
            # layers processing time-independent quantities
            self.gcs = GraphConvolution(ndemo, ntime*2)


    def forward(self, x, adj, history, xs, weather, los, qod):

        batch_size, timesteps, stations, features = x.size()
        device = adj.device

        # concatenate each time period and look back period
        x = x.view(batch_size*timesteps, stations, features)
        x = self.batchnorm(x)

        # graph attention
        if len(adj.size()) == 2:
            adj = adj.unsqueeze(2)

        gc_out = torch.zeros(batch_size*timesteps, stations, self.nhid_g).to(device)
        for i in range(self.nadj):
            temp = x
            for j in range(self.nga):
                temp = F.dropout(self.gat[i][j](temp, adj[:,:,i]), self.dropout, training=self.training)
                #temp = self.layernorm[j](temp)
        gc_out = gc_out/self.nadj

        # LSTM
        r_in = gc_out.contiguous().view(batch_size, timesteps, -1)
        r_in = F.relu(self.fc2(F.dropout(F.relu(self.fc1(r_in)), self.dropout, training=self.training)))
        r_out,_ = self.lstm(r_in)
        r_out = torch.squeeze(r_out[:,-1,:]) # only take the last timestep

        # FC
        out = F.dropout(F.relu(self.fc3(r_out)), self.dropout, training=self.training)
        out = self.final(out)

        # Demographics, points of interest, etc (time-independent quantities)
        if xs is not None:
            gcs_out = torch.zeros(stations, self.ntime*(2-self.meanonly)).to(device)
            for j in range(self.nadj):
                gcs_out += F.dropout(F.relu(self.gcs(xs,adj[:,:,j])), self.dropout, training=self.training)
            gcs_out = gcs_out/self.nadj
            gcs_mean = torch.matmul(qod, torch.transpose(gcs_out[:,:self.ntime],0,1))
        else:
            gcs_mean = torch.zeros(batch_size, stations).to(device)

        # History
        history = torch.squeeze(history)
        history_mean = history * torch.mm(qod, self.history_weights_mean)

        # Weather
        weather = weather.view(batch_size, 1, 2)
        weather_mean = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_mean).view(batch_size, 2, stations)))

        # Level of Service
        los_mean = los * torch.mm(qod, self.los_weights_mean)

        if self.meanonly:
            out = out.view(batch_size, -1, 1)
            out = F.relu(out[:,:,0]+history_mean+gcs_mean+weather_mean+los_mean)
        else:
            history_var = history * torch.mm(qod, self.history_weights_var)
            weather_var = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_var).view(batch_size, 2, stations)))
            if xs is not None:
                gcs_var = torch.matmul(qod, torch.transpose(gcs_out[:,self.ntime:],0,1))
            else:
                gcs_var = torch.zeros(batch_size, stations).to(device)
            out = out.view(batch_size, -1, 2)
            out = torch.cat((F.relu(out[:,:,0]+gcs_mean+history_mean+los_mean+weather_mean),
                F.softplus(out[:,:,1]+gcs_var+history_var+weather_var)), 0)

        return out
