import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN_LSTM(nn.Module):
    def __init__(self, meanonly, homo, nadj, nmode, nstation, ntime, ndemo, nhid_g, ngc, nhid_l, nlstm, nhid_fc, dropout, std_starter=1.0):
        # meanonly: bool, whether output is mean only or mean and standard deviation
        # nmode: number of modes (features) in x (ridership immediately before)
        # nstation: number of spatial units
        # ntime: number of time periods in a day
        # ndemo: number of time-independent quantities (all collapsed into demo)
        # nhid_[g,l,fc]: number of hidden (output) units of the graph/lstm/fc layers
        # ngc: number of graph convolution layers
        # nlstm: number of lstm layers
        # dropout

        super().__init__()

        self.meanonly = meanonly
        if homo: # if homoskedastic, then only mean is produced by the convolutions
            self.meanonly = True
#             self.std = nn.Parameter(torch.tensor([std_starter]), requires_grad=True)
        self.homo = homo
        self.nhid_g = nhid_g
        self.ntime = ntime
        self.ngc = ngc
        self.nadj = nadj
        
        # batchnorm1d: over 3D input of size (N,C,L); num_features = C
        self.batchnorm = nn.BatchNorm1d(num_features=nstation)
        self.dropout = dropout

        # Ridership immediately before
        self.gc = []
        for i in range(nadj):
            innerlist = []
            for j in range(ngc):
                if j == 0:
                    innerlist.append(GraphConvolution(nmode, nhid_g))
                else:
                    innerlist.append(GraphConvolution(nhid_g, nhid_g))
            self.gc.append(nn.ModuleList(innerlist))
        self.gc = nn.ModuleList(self.gc)

        self.fc1 = nn.Linear(nhid_g*nstation, nhid_fc)
        self.fc2 = nn.Linear(nhid_fc, nhid_g)
       
        self.lstm = nn.LSTM(input_size=int(nhid_g), hidden_size=int(nhid_l), num_layers=nlstm, batch_first=True)
        self.fc3 = nn.Linear(nhid_l, nhid_fc)
        self.fc4 = nn.Linear(nhid_fc, nhid_fc)

        # History
        # 1
        self.recent_on_history_mean = nn.Linear(nhid_fc, nstation)

        # Weather
        self.weather_weights_mean = nn.Parameter(torch.rand((ntime, 2*nstation)))

        # Level of Service
        self.los_weights_mean = nn.Parameter(torch.rand(ntime, nstation))

        if meanonly | (homo>0):
            mult = 1
        else:
            mult = 2

        # layers processing time-independent quantities
        if ndemo != 0:
            self.gcs = GraphConvolution(ndemo, ntime*mult)
        # layers bringing everything together
        self.final = nn.Linear(nhid_fc, nstation*mult)

#         if (not self.meanonly) | (homo): # 220405 glitch
        if (not self.meanonly):
            self.recent_on_history_var = nn.Linear(nhid_fc, nstation)
            self.weather_weights_var = nn.Parameter(torch.rand((ntime, 2*nstation)))

    def _embed_x(self, x, adj, device):
        # graph convolution on x

        batch_size, timesteps, stations, features = x.size()
        if len(adj.size()) == 2:
            nadj = 1
            adj = adj.unsqueeze(2)
        else:
            _,_,nadj = adj.size()

        # concatenate each time period and look back period
        x = x.view(batch_size * timesteps, stations, features)
        x = self.batchnorm(x)

        # graph convolution
        gc_out = torch.zeros(batch_size*timesteps, stations, self.nhid_g).to(device)
        for i in range(nadj):
            temp = x
            for j in range(self.ngc):
                temp = self.gc[i][j](temp, adj[:,:,i])
                if torch.sum(torch.isnan(temp))!=0:
                    print('gc')
                    print(self.gc[i][j].weight)
                temp = F.dropout(F.relu(temp), self.dropout, training=self.training)
            gc_out += temp
            if torch.sum(torch.isnan(temp)) != 0:
                print(i)
        gc_out = gc_out / nadj

        r_in = gc_out.contiguous().view(batch_size, timesteps, -1)
        r_in = F.relu(self.fc2(F.dropout(F.relu(self.fc1(r_in)), self.dropout, training=self.training)))

        return r_in


    def forward(self, x, x_future, adj, history, xs, weather, los, qod, futures, return_components=False):

        batch_size, timesteps, stations, features = x.size()
        device = adj.device

        r_in = self._embed_x(x, adj, device)
#         print(r_in.view(batch_size, timesteps, -1)[0:5,:,:])

        # LSTM step by step
        '''
        h_t = torch.zeros(batch_size, self.nhid_g, dtype=torch.float32).to(device)
        c_t = torch.zeros(batch_size, self.nhid_g, dtype=torch.float32).to(device)
        for i in range(timesteps):
            h_t,c_t = self.lstmcell(r_in[:,i,:], (h_t,c_t))
        r_out = h_t
        '''
        # Standard one-step LSTM
        r_out,_ = self.lstm(r_in)
        r_out = torch.squeeze(r_out[:,-1,:]) # only take the last timestep
        
        for i in range(futures):

            if i > 0:
                print("Feature not implemented")
                break

            '''
            # if predicting more than one step, then running lstm unit again (feature not tested)
            # tnc and bus ridership yet to be added.
            if i > 0:
                if x_future is not None:
                    # teacher forcing
                    r_in = self._embed_x(x_future[:,i-1,:,:], adj, device)
                else:
                    # use prediction from previous step
                    r_in = self._embed_x(out_mean[:,-1,:], adj, device)

                h_t,c_t = self.lstmcell(r_in, (h_t, c_t))
                r_out = self.lstm_out(h_t)
            '''

            # FC
            out = self.fc3(r_out)
            out = F.dropout(F.relu(self.fc4(r_out)), self.dropout, training=self.training)
            # 1
            recent_on_history_weights_mean = torch.sigmoid(self.recent_on_history_mean(out)).view(batch_size, stations)

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
#             print(self.recent_on_history_mean(out)[0,:])
#             print(self.recent_on_history_var(out)[0,:])
#             print(out)

            if not self.meanonly:
                recent_on_history_weights_var = torch.sigmoid(self.recent_on_history_var(out)).view(batch_size, stations)
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

            if (self.meanonly)|(self.homo>0):
                out = out.view(batch_size, -1, 1)
#                 out_mean = F.softplus(out[:,:,0]+history_mean+gcs_mean+weather_mean+los_mean)
                out_mean = F.softplus(out[:,:,0]+gcs_mean+history_mean+weather_mean)
                if i == 0:
                    final_out = out_mean
                else:
                    final_out = torch.cat((final_out, out_mean), -1)
            else:
                out = out.view(batch_size, -1, 2)
#                 out_mean = F.softplus(out[:,:,0]+gcs_mean+history_mean+los_mean+weather_mean)
                out_mean = F.softplus(out[:,:,0]+gcs_mean+history_mean+weather_mean)
                out_var = F.softplus(out[:,:,1]+gcs_var+history_var+weather_var)
                current_out = torch.cat((out_mean, out_var), 0)
                if i == 0:
                    final_out = current_out
                else:
                    final_out = torch.cat((final_out, current_out), -1)
#                 print(recent_on_history_weights_mean[0,:])
#                 print(recent_on_history_weights_var[0,:])
                
        if not return_components:
            return final_out
        else:
            # return-components for homoskedastic variance is not implemented
            if xs is not None:
                return [[out[:,:,0],gcs_mean,history_mean,weather_mean,los_mean],[out[:,:,1],gcs_var,history_var,weather_var]]
            else:
                return [[out[:,:,0],history_mean,weather_mean,los_mean],[out[:,:,1],history_var,weather_var]]
    
