import argparse
import glob
import itertools
import os
import sys
sys.path.append('../rail_catchment/')
sys.path.append('../')

import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import scipy

# custom GCNN classes
from class_dataset import CTA_Data
from class_gcn_lstm import GCN_LSTM
from class_mve_loss import LN_MVELoss
from class_mve_loss import T_MVELoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import util_gcnn
import util_eval
import util_plot

plt.rcParams.update({'font.size': 16})

if __name__ == "__main__":

    project_dir = "/home/jtl/Dropbox (MIT)/project_uncertainty_quantification/"

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--Period", help="Study period: before/after", default=['before'])    
    parser.add_argument("-ph", "--PredictHorizon", help="Predict horizon", default=[1])
    parser.add_argument("-ts", "--TimeSize", help="Time aggregation", default=[4])
    parser.add_argument("-df", "--Difference", help="Whether to difference", default=[True])
    parser.add_argument("-l", "--Lookback", help="Lookback period", default=[2])
    parser.add_argument("-a", "--Adj", help="Adjacency matrix type", default=[["func","euc","con","net"]])
    parser.add_argument("-d", "--Dist", help="Lognorm/Tnorm distribution", default=['tnorm'])
    parser.add_argument("-ls", "--LossFunc", help="Loss function mve/mse", default='mve')
    parser.add_argument("-m", "--MeanOnly", help="Whether to output mean only or mean+variance", default=False)
    parser.add_argument("-hm", "--Homoskedastic", help="Whether homo or hetero-skedastic variance is estimated", default=False)
    parser.add_argument("-te", "--TrainExtent", help="Training Extent: downtown/all", default=['downtown'])
    parser.add_argument("-e", "--Epoch", help="Number of training epochs", default=500)
    parser.add_argument("-lr", "--LearningRate", help="Learning rate", default=0.0002)
    parser.add_argument("-ms", "--ModelNumberStart", help="Start number of models for each combination of the parameters", default=0)
    parser.add_argument("-me", "--ModelNumberEnd", help="End number of models for each combination of the parameters", default=10)
    parser.add_argument("-s", "--IncludeSpatial", help="Whether to include spatial (demo+poi) information", default=False)
    parser.add_argument("-b", "--Bootstrap", help="Whether to bootstrap training data", default=False)
    parser.add_argument("-of", "--OutFolder", help="Output Folder", default="")
    parser.add_argument("-sv", "--Save", help="Whether to save models", default=False)

    args = parser.parse_args()

    dropout_rate_list=[0.25, 0.5]
    n_hid_units_list=[32,64,128,256]
    weight_decay_list = [0.005, 0.01, 0.05]
    ngc_list = [1,2,3]
    nlstm_list = [1,2,3]

    hps = list(itertools.product(dropout_rate_list, n_hid_units_list, weight_decay_list, nlstm_list, ngc_list))
    max_lookback = 6
    z = 1.96

    if args.ModelNumberStart == -1:
        run_all = True
    else:
        run_all = False

    for (period, predict_hzn, time_size, difference, lookback, adj_type, dist, train_extent) in itertools.product(args.Period, args.PredictHorizon, args.TimeSize, args.Difference, args.Lookback, args.Adj, args.Dist, args.TrainExtent):

        print("Period:", period, "Predict Horizon:", predict_hzn, "Time Size:", time_size, "Lookback:", lookback, "\nadj:", '_'.join(adj_type), "dist:", dist, "trainextent:", train_extent)
        n_time = 96 // time_size - 7

        data, adj, spatial, downtown_filter = \
                util_gcnn.combine_datasources(project_dir, period, predict_hzn, time_size, difference, max_lookback, lookback)

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        trainloader, trainloader_test, valloader, testloader, adj_torch, spatial_torch, y_train_eval, y_val_eval, y_test_eval = \
                util_gcnn.prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, bootstrap=args.Bootstrap)
        (num_train, _, _, n_modes) = data['x'][0].shape
        num_test = len(y_test_eval)
        num_val = len(y_val_eval)
        n_stations = adj_torch.shape[0]
        y_train_eval = np.squeeze(y_train_eval)
        y_val_eval = np.squeeze(y_val_eval)
        y_test_eval = np.squeeze(y_test_eval)

        same_hp = False
        if not args.IncludeSpatial:
            spatial_torch = None
            ndemo = 0
        else:
            ndemo = spatial.shape[1]

        if run_all:
            results = pd.read_csv(project_dir+"results/"+args.OutFolder+"rail_catchment_mve_results.csv")
            hp_run = results[(results['Period']==period)&(results['Predict Horizon']==predict_hzn)&(results['Time Size']==time_size)&(results['Adjacency']=='_'.join(adj_type))& \
                    (results['Lookback']==lookback)&(results['Train Extent']=='downtown')&(results['spatial']==int(args.IncludeSpatial))] 
            if len(hp_run) != 0:
                model_offset = hp_run['Model'].max()+1
            else:
                model_offset = 0
            hp_run = hp_run[['dropout','n_hid_units','weight_decay','nlstm','ngc']].to_numpy()
            hp_run = [tuple(l) for l in hp_run]
            args.ModelNumberStart = 0
            args.ModelNumberEnd = len(hps)
            model_offset_2 = 0
        else:
            model_offset = 0
            model_offset_2 = 0

        for ii in range(args.ModelNumberStart, args.ModelNumberEnd):

            if run_all: 
                hp_index = ii
                if hps[hp_index] in hp_run:
                    model_offset_2 += 1
                    continue
            else:
                hp_index = np.random.choice(np.arange(0, len(hps)))

            save_dir = project_dir+"models/"+args.OutFolder+period+"_"+train_extent+"_"+'-'.join(adj_type)+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+str(lookback)+"_"+str(ii+model_offset-model_offset_2)
            best = 0
            best_epoch = 0

            dropout = hps[hp_index][0]
            n_hid_units = hps[hp_index][1]
            weight_decay = hps[hp_index][2]
            nlstm = hps[hp_index][3]
            ngc = hps[hp_index][4]

            print('model ', ii+model_offset-model_offset_2)

            if args.LossFunc == 'mve':
                if dist == 'lognorm':
                    criterion = LN_MVELoss()
                else:
                    criterion = T_MVELoss()
            elif args.LossFunc == 'mse':
                criterion = nn.MSELoss()
            else:
                print("loss function not valid.")
                sys.exit()

            net = GCN_LSTM(meanonly=args.MeanOnly, homo=args.Homoskedastic, nadj = len(adj_type), nmode=n_modes, nstation=n_stations, ntime=n_time, 
                    ndemo=ndemo,
                    nhid_g=n_hid_units, ngc=ngc, nhid_l=n_hid_units, nlstm=nlstm, 
                    nhid_fc=n_hid_units, dropout=dropout)
            net.to(device)

            optimizer = optim.Adam(net.parameters(), lr=args.LearningRate, weight_decay=weight_decay)
            ref1 = 0
            ref2 = 0
            success=True

            for epoch in range(args.Epoch):

                running_loss = 0.0

                for i, batch_data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels, history, quarter of the day]
                    batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod = batch_data
                    batch_size=len(batch_x)

                    batch_x = batch_x.float()
                    batch_y = torch.squeeze(batch_y).float()
                    batch_history = batch_history.float()
                    batch_qod = batch_qod.view(-1,1)
                    batch_qod_onehot = torch.FloatTensor(batch_size, n_time)
                    batch_qod_onehot.zero_()
                    batch_qod_onehot.scatter_(1, batch_qod-6, 1)
                    batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod_onehot = \
                            batch_x.to(device), batch_y.to(device), batch_history.to(device), \
                            batch_weather.to(device), batch_los.to(device), batch_qod_onehot.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward 
                    outputs = net(batch_x, None, adj_torch, batch_history, spatial_torch, 
                            batch_weather, batch_los, batch_qod_onehot, 1)
                    if args.LossFunc == 'mve':
                        if not args.Homoskedastic:
                            output_mean = outputs[:batch_size,:]
                            output_var = outputs[batch_size:,:]
                        else:
                            output_mean, output_var = outputs

                        loss = criterion(output_mean, output_var, batch_y)
                    elif args.LossFunc == 'mse':
                        output_mean = outputs
                        loss = criterion(output_mean, batch_y)

                    # backward
                    loss.backward()

                    # optimize
                    optimizer.step()

                    running_loss += loss.item()

                    if torch.sum(torch.isnan(output_mean)) > 0:
                        success=False
                        break

                if not success:
                    break

                if epoch % 50 == 0:
                    print('[%d] training loss: %.3f' %
                            (epoch + 1, running_loss/num_train), end = '\t')

                    net.eval()
                    test_out_loc, test_out_std, test_loss = util_gcnn.testset_output_gcn(testloader, args.MeanOnly, args.Homoskedastic, net, criterion, 
                            adj_torch, spatial_torch, device, n_time)
                    if dist == "lognorm":
                        test_out_loc = np.exp(test_out_loc - np.power(test_out_std,2))

                    mae = np.mean(np.abs(test_out_loc - y_test_eval))
                    if not args.MeanOnly:
                        #print(test_out_std)
                        print("test loss %.2f, mae %.2f, std %.2f" % (test_loss/len(y_test_eval), mae, np.mean(test_out_std)))
                    else:
                        print("test loss %.2f, mae %.2f" % (test_loss/len(y_test_eval), mae))

                    net.train()

                if epoch % 10 == 0:
                    if epoch > 40:
                        if (np.abs(running_loss - ref1)/ref1<0.005) & (np.abs(running_loss - ref2)/ref2<0.005):
                            print("early stopping at epoch", epoch)
                            break
                        if (ref1 < running_loss) & (ref1 < ref2):
                            print("diverging. stop.")
                            break
                        if running_loss < best:
                            best = running_loss
                            best_epoch = epoch
                    else:
                        best = running_loss
                        best_epoch = epoch

                    ref2 = ref1
                    ref1 = running_loss

                    if (args.Save) & (best_epoch==epoch):
                        torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'hyperparameters': (period, train_extent, train_extent, '_'.join(adj_type), ii, 
                                predict_hzn, time_size, lookback, 
                                dropout, n_hid_units, nlstm, ngc, weight_decay, args.Bootstrap)},
                            save_dir+"_"+str(epoch)+".pt")

                        files = glob.glob(save_dir+"_*.pt")

            if args.Save:
                for f in files:
                    e = int(f.split("_")[-1].split(".")[0])
                    if e != best_epoch:
                        os.remove(f)
                    else:
                        net.load_state_dict(torch.load(f)['model_state_dict'])

            if not success:
                print("trainig failed ", str(ii+model_offset-model_offset_2))
            else:
                print('finished training ', str(ii+model_offset-model_offset_2))

            net.eval()

            # validation set performance
            val_out_loc, val_out_std, val_loss = util_gcnn.testset_output_gcn(valloader, args.MeanOnly, args.Homoskedastic, net, criterion, 
                    adj_torch, spatial_torch, device, n_time)
            val_mae, val_mse, _, _, _ = util_eval.eval_mean(val_out_loc, y_val_eval, 'Val')
            val_u, val_ub, val_uv, val_uc = util_eval.eval_theils(val_out_loc, y_val_eval)
            if args.MeanOnly == False:
                if dist == "lognorm":
                    val_out_loc = np.exp(val_out_loc - np.power(val_out_std,2))
                val_mpiw, val_picp = util_eval.eval_pi(val_out_loc - z*val_out_std, val_out_loc + z*val_out_std, y_val_eval)
            else:
                val_mpiw = 0
                val_picp = 0
    
            # Test Set Performance 
            test_out_loc, test_out_std, test_loss = util_gcnn.testset_output_gcn(testloader, args.MeanOnly, args.Homoskedastic, net, criterion, 
                    adj_torch, spatial_torch, device, n_time)
            mae, mse, _, _, _ = util_eval.eval_mean(test_out_loc, y_test_eval, 'Test')
            u, ub, uv, uc = util_eval.eval_theils(test_out_loc, y_test_eval)
            if args.MeanOnly == False:
                if dist == "lognorm":
                    test_out_loc = np.exp(test_out_loc - np.power(test_out_std,2))
                mpiw, picp = util_eval.eval_pi(test_out_loc - z*test_out_std, test_out_loc + z*test_out_std, y_test_eval)
            else:
                mpiw = 0
                picp = 0

            # Training Set Performance
            train_out_loc, train_out_std, train_loss = util_gcnn.testset_output_gcn(trainloader_test, args.MeanOnly, args.Homoskedastic, net, criterion, 
                    adj_torch, spatial_torch, device, n_time)
            tr_mae, tr_mse, _, _, pct_nonzeros = util_eval.eval_mean(train_out_loc, y_train_eval, 'Train')
            tr_u, tr_ub, tr_uv, tr_uc = util_eval.eval_theils(train_out_loc, y_train_eval)
            if args.MeanOnly == False:
                if dist == "lognorm":
                    train_out_loc = np.exp(train_out_loc - np.power(train_out_std,2))
                tr_mpiw, tr_picp = util_eval.eval_pi(train_out_loc - z*train_out_std, train_out_loc + z*train_out_std, y_train_eval)
            else:
                tr_mpiw = 0
                tr_picp = 0

            # Write to File
            with open(project_dir+"results/"+args.OutFolder+"rail_catchment_"+args.LossFunc+"_results.csv","a") as f:
                f.write("%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%.2f,%d,%d,%d,%.2E,%d,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % 
                        ("GCN", period, train_extent, train_extent, '_'.join(adj_type), ii+model_offset-model_offset_2, 
                            predict_hzn, time_size, lookback, args.Bootstrap, 
                            dropout, n_hid_units, nlstm, ngc, weight_decay, spatial_torch is not None, 
                            train_loss/num_train, tr_mae, tr_mse, tr_mpiw, tr_picp,
                            val_loss/num_val, val_mae, val_mse, val_mpiw, val_picp,
                            test_loss/num_test, mae, mse, mpiw, picp,
                            tr_u, tr_ub, tr_uv, tr_uc, 
                            val_u, val_ub, val_uv, val_uc, 
                            u, ub, uv, uc))

            # Not updated. If training on the whole network
            if train_extent == 'all':
                mae, mse, _, _, _ = util_eval.eval_mean(test_out_loc[:,downtown_filter], y_test_eval[:,downtown_filter], 'Test')
                u, ub, uv, uc = util_eval.eval_theils(test_out_loc[:,downtown_filter], y_test_eval[:,downtown_filter])
                mpiw, picp = util_eval.eval_pi(test_out_loc[:,downtown_filter] - z*test_out_std[:,downtown_filter], 
                        test_out_loc[:,downtown_filter] + z*test_out_std[:,downtown_filter], y_test_eval[:,downtown_filter])
                tr_mae, tr_mse, _, _, pct_nonzeros = util_eval.eval_mean(train_out_loc[:,downtown_filter], y_train_eval[:,downtown_filter], 'Train')
                tr_u, tr_ub, tr_uv, tr_uc = util_eval.eval_theils(train_out_loc[:,downtown_filter], y_train_eval[:,downtown_filter])
                tr_mpiw, tr_picp = util_eval.eval_pi(train_out_loc[:,downtown_filter] - z*train_out_std[:,downtown_filter], 
                        train_out_loc[:,downtown_filter] + z*train_out_std[:,downtown_filter], y_train_eval[:,downtown_filter])

                with open(project_dir+"results/"+args.OutFolder+"rail_catchment"+args.LossFunc+"_results.csv","a") as f:
                    f.write("%s,%s,%s,%s,%s,%d,%d,%d,%d,%.2f,%d,%d,%d,%.2E,%d,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % 
                            ("GCN", period, train_extent, 'downtown', '_'.join(adj_type), ii+model_offset-model_offset_2, 
                                predict_hzn, time_size, lookback, 
                                dropout, n_hid_units, nlstm, ngc, weight_decay, spatial_torch is not None,
                                train_loss/num_train, tr_mae, tr_mse, tr_mpiw, tr_picp,
                                test_loss/num_test, mae, mse, mpiw, picp,
                                tr_u, tr_ub, tr_uv, tr_uc, u, ub, uv, uc))


