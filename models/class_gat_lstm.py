import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttention
from layers import GraphConvolution

class GAT_LSTM(nn.Module):
    def __init__(self, meanonly, nadj, nmode, nstation, ntime, ndemo, nhead, nhid_g, nga, nhid_l, nlstm, nhid_fc, dropout, homo=False, std_starter=None):
        
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
        self.nhid_fc = nhid_fc

        if homo > 0 : # if homoskedastic, then only mean is produced by the convolutions
            self.meanonly = True
            # self.std = nn.Parameter(torch.tensor([std_starter]), requires_grad=True)
#         self.homo = homo
        
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
                    innerlist.append(GraphAttention(nhid_g, nhid_g, nhead))
            self.gat.append(nn.ModuleList(innerlist))
        self.gat = nn.ModuleList(self.gat)

#         self.layernorm = []
#         for j in range(nga):
#             self.layernorm.append(nn.LayerNorm(nhid_g))
#         self.layernorm = nn.ModuleList(self.layernorm)

        self.fc1 = nn.Linear(nhid_g*nstation, nhid_fc)
        self.fc2 = nn.Linear(nhid_fc, nhid_g)
        self.lstm = nn.LSTM(input_size=int(nhid_g), hidden_size=int(nhid_l), num_layers=nlstm, batch_first=True)

        # History
        self.fc3 = nn.Linear(nhid_l, nhid_fc)
        self.fc4 = nn.Linear(nhid_fc, nhid_fc)
        self.recent_on_history_mean = nn.Linear(nhid_fc, nstation)

        # Weather
        self.weather_weights_mean = nn.Parameter(torch.rand((ntime, 2*nstation)))

        # Level of Service
        self.los_weights_mean = nn.Parameter(torch.rand(ntime, nstation))

        if self.meanonly:
            self.gcs = GraphConvolution(ndemo, ntime)
            self.final = nn.Linear(nhid_fc, nstation)
        else:
            self.recent_on_history_var = nn.Linear(nhid_fc, nstation)
            self.weather_weights_var = nn.Parameter(torch.rand((ntime, 2*nstation)))
            # layers bringing everything together
            self.final = nn.Linear(nhid_fc, nstation*2)
            # layers processing time-independent quantities
            self.gcs = GraphConvolution(ndemo, ntime*2)


    def forward(self, x, adj, history, xs, weather, los, qod, return_components=False):

        batch_size, timesteps, stations, features = x.size()
        device = adj.device

        # concatenate each time period and look back period
        x = x.view(batch_size*timesteps, stations, features)
        x = self.batchnorm(x)

        # graph attention
        if len(adj.size()) == 2:
            adj = adj.unsqueeze(2)

        gc_out = torch.zeros(batch_size*timesteps, stations, self.nhid_g).to(device)
#         for i in range(self.nadj):
        for i in range(1):
            temp = x
            for j in range(self.nga):
                ## 220414, not using adj
                temp = F.dropout(self.gat[i][j](temp, None), self.dropout, training=self.training)
                ## 220415, using adj
#                 temp = F.dropout(self.gat[i][j](temp, adj[:,:,i]), self.dropout, training=self.training)
                gc_out += temp
                #temp = self.layernorm[j](temp)
#         gc_out = gc_out/self.nadj

#         print(gc_out.view(batch_size, timesteps, -1)[0:5,:,:])

        # LSTM
        r_in = gc_out.contiguous().view(batch_size, timesteps, -1)
        r_in = F.relu(self.fc2(F.dropout(F.relu(self.fc1(r_in)), self.dropout, training=self.training)))
#         print(r_in.shape) 
        r_out,_ = self.lstm(r_in)
        r_out = torch.squeeze(r_out[:,-1,:]) # only take the last timestep
        
        # FC
        out = self.fc3(r_out)
        out = F.dropout(F.relu(self.fc4(r_out)), self.dropout, training=self.training)
        # 1
        recent_on_history_weights_mean = torch.sigmoid(self.recent_on_history_mean(out)).view(batch_size, stations)
#         recent_on_history_weights_mean = F.softplus(self.recent_on_history_mean(out)).view(batch_size, stations)

        # Demographics, points of interest, etc (time-independent quantities)
        ntime = qod.shape[1]
        if xs is not None:
            if self.meanonly:
                gcs_out = torch.zeros(stations, self.ntime).to(device)
            else:
                gcs_out = torch.zeros(stations, 2*self.ntime).to(device)
            for j in range(self.nadj):
                gcs_out += F.dropout(F.relu(self.gcs(xs,adj[:,:,j])), self.dropout, training=self.training)
            gcs_out = gcs_out/self.nadj
            gcs_mean = torch.matmul(qod, torch.transpose(gcs_out[:,:ntime],0,1))
        else:
            gcs_mean = torch.zeros(batch_size, stations).to(device)

        # History and Weather
        history = torch.squeeze(history)
        history_mean = history * recent_on_history_weights_mean

        weather = weather.view(batch_size, 1, 2)
        weather_mean = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_mean).view(batch_size, 2, stations)))
#         print(out)
        
#         print(self.recent_on_history_mean(out)[0,:])
#         print(self.recent_on_history_var(out)[0,:]) 
        
        if not self.meanonly:
            recent_on_history_weights_var = torch.sigmoid(self.recent_on_history_var(out)).view(batch_size, stations)
#             recent_on_history_weights_var = F.softplus(self.recent_on_history_var(out)).view(batch_size, stations)
            history_var = history * recent_on_history_weights_var
            weather_var = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_var).view(batch_size, 2, stations)))
            if xs is not None:
                gcs_var = torch.matmul(qod, torch.transpose(gcs_out[:,ntime:],0,1))
            else:
                gcs_var = torch.zeros(batch_size, stations).to(device)

        # Level of Service
        los = torch.squeeze(los)
        los_mean = los * torch.mm(qod, self.los_weights_mean)

        out = self.final(out)

        if self.meanonly:
            out = out.view(batch_size, -1, 1)
#                 out_mean = F.softplus(out[:,:,0]+history_mean+gcs_mean+weather_mean+los_mean)
            out_mean = F.softplus(out[:,:,0]+gcs_mean+history_mean+weather_mean)
            final_out = out_mean
        else:
            out = out.view(batch_size, -1, 2)
#                 out_mean = F.softplus(out[:,:,0]+gcs_mean+history_mean+los_mean+weather_mean)
            out_mean = F.softplus(out[:,:,0]+gcs_mean+history_mean+weather_mean)
            out_var = F.softplus(out[:,:,1]+gcs_var+history_var+weather_var)
            current_out = torch.cat((out_mean, out_var), 0)
            final_out = current_out
             
#             print(recent_on_history_weights_mean)
#             print(recent_on_history_weights_var)
        if not return_components:
            return final_out
        else:
            # return-components for homoskedastic variance is not implemented
            if xs is not None:
                return [[out[:,:,0],gcs_mean,history_mean,weather_mean],[out[:,:,1],gcs_var,history_var,weather_var]]
            else:
                return [[out[:,:,0],history_mean,weather_mean],[out[:,:,1],history_var,weather_var]]


         # FC
#         out = F.dropout(F.relu(self.fc3(r_out)), self.dropout, training=self.training)
#         out = self.final(out)

#         # Demographics, points of interest, etc (time-independent quantities)
#         if xs is not None:
#             gcs_out = torch.zeros(stations, self.ntime*(2-self.meanonly)).to(device)
#             for j in range(self.nadj):
#                 gcs_out += F.dropout(F.relu(self.gcs(xs,adj[:,:,j])), self.dropout, training=self.training)
#             gcs_out = gcs_out/self.nadj
#             gcs_mean = torch.matmul(qod, torch.transpose(gcs_out[:,:self.ntime],0,1))
#         else:
#             gcs_mean = torch.zeros(batch_size, stations).to(device)

#         # History
#         history = torch.squeeze(history)
#         history_mean = history * torch.mm(qod, self.history_weights_mean)

#         # Weather
#         weather = weather.view(batch_size, 1, 2)
#         weather_mean = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_mean).view(batch_size, 2, stations)))

#         # Level of Service
#         los_mean = los * torch.mm(qod, self.los_weights_mean)

#         if self.meanonly:
#             out = out.view(batch_size, -1, 1)
#             out = F.relu(out[:,:,0]+history_mean+gcs_mean+weather_mean+los_mean)
#         else:
#             history_var = history * torch.mm(qod, self.history_weights_var)
#             weather_var = torch.squeeze(torch.bmm(weather, torch.mm(qod, self.weather_weights_var).view(batch_size, 2, stations)))
#             if xs is not None:
#                 gcs_var = torch.matmul(qod, torch.transpose(gcs_out[:,self.ntime:],0,1))
#             else:
#                 gcs_var = torch.zeros(batch_size, stations).to(device)
#             out = out.view(batch_size, -1, 2)
#             out = torch.cat((F.relu(out[:,:,0]+gcs_mean+history_mean+los_mean+weather_mean),
#                 F.softplus(out[:,:,1]+gcs_var+history_var+weather_var)), 0)

#         return out
