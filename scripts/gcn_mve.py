import argparse
from datetime import datetime
import glob
import itertools
import os
import sys
sys.path.append('../models/')
sys.path.append('../eval/')
sys.path.append('../process_data/')
sys.path.append('../')

# import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
from scipy.stats import poisson, norm, laplace, lognorm

# custom GCNN classes
from class_dataset import CTA_Data
from class_gcn_lstm import GCN_LSTM
from class_mve_loss import MVELoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import util_gcnn
import util_eval
import util_data
# import util_plot

from setup import *

# plt.rcParams.update({'font.size': 11})

def post_process_dist(dist, loc, scale):
    
    if dist == "lognorm":
        out_predict = np.exp(loc - np.power(scale,2))
#         out_std = np.mean(np.sqrt((np.exp(scale * scale)-1)*(np.exp(2*loc+scale * scale))))
    elif dist == 'tnorm':
        out_predict = loc
#         out_std = np.mean(scale)
    elif dist == 'laplace':
        out_predict = loc
#         out_std = np.mean(scale) * np.sqrt(2)
    elif dist == 'poisson':
        out_predict = loc
#         out_std = np.sqrt(loc)
    elif dist == 'norm':
        out_predict = loc
        
        
    return out_predict, None
 
    
def post_process_pi(dist, loc, scale, z):
    
    if dist == "lognorm":
#         predict = np.exp(loc - np.power(scale,2))
#         lb = np.exp(predict - z*scale)
#         ub = np.exp(predict + z*scale)
        lb, ub = lognorm.interval(z, loc, scale)
    elif dist == 'tnorm':
#         lb = np.max([0, loc - z*scale])
#         ub = loc + z*scale
        lb, ub = norm.interval(z, loc, scale)
        lb = lb * (lb>0)
    elif dist == 'laplace':
#         predict = loc
#         lb = predict + scale * np.log(2*z)
#         ub = predict - scale * np.log(2-2*(1-z))
        lb, ub = laplace.interval(z, loc, scale)
    elif dist == 'poisson':
        predict = loc
        lb,ub = poisson.interval(z, loc) 
    elif dist == 'norm':
        lb, ub = norm.interval(z, loc, scale)

    return lb, ub

def evaluate(args, dist, dataloader, z, y_eval, std=None, time_size=1):
    
    val_out_loc, val_out_scale, val_loss = util_gcnn.testset_output_gcn(dataloader, args.MeanOnly, args.Homoskedastic, net, criterion, 
                adj_torch, spatial_torch, device, n_time, std=std, time_size=time_size)
        
    val_out_predict, val_out_std = post_process_dist(dist, val_out_loc, val_out_scale)
   
    val_mae, val_mape, _ = util_eval.eval_mean(val_out_predict, y_eval, "")

    if args.LossFunc == 'mve':
        lb, ub = post_process_pi(dist, val_out_loc, val_out_scale, z)
        val_mpiw, val_picp = util_eval.eval_pi(lb, ub, y_eval)
    else:
        val_mpiw = 0
        val_picp = 0
        
    return val_loss, val_mae, val_mape, val_mpiw, val_picp

if __name__ == "__main__":

    project_dir = "/home/jtl/Dropbox (MIT)/project_uncertainty_quantification/"

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    
    parser.add_argument("-dataset", "--Dataset", help="rail_catchment / census_tract", type=str, default="rail_catchment")
    parser.add_argument("-trainst", "--TrainStartDate", help="Train start date", type=str, default='20190801')
    parser.add_argument("-trained", "--TrainEndDate", help="Train end date", type=str, default='20200301')
    parser.add_argument("-testst", "--TestStartDate", help="Test start date", type=str, default='20200302')
    parser.add_argument("-tested", "--TestEndDate", help="Test end date", type=str, default='20200315')
    parser.add_argument("-ph", "--PredictHorizon", help="Predict horizon", default=1)
    parser.add_argument("-ts", "--TimeSize", help="Time aggregation", default=4)
    parser.add_argument("-df", "--Difference", help="Whether to difference", default=1)
    parser.add_argument("-l", "--Lookback", help="Lookback period", default=2)
    parser.add_argument("-a", "--Adj", help="Adjacency matrix type", default=["func","euc","con","net"])
    parser.add_argument("-d", "--Dist", help="Lognorm/Tnorm distribution", default='tnorm')
    parser.add_argument("-ls", "--LossFunc", help="Loss function mve/mse", default='mve')
    parser.add_argument("-m", "--MeanOnly", help="Whether to output mean only or mean+variance", default=0)
    parser.add_argument("-hm", "--Homoskedastic", help="Whether homo or hetero-skedastic variance is estimated", default=0)
    parser.add_argument("-te", "--TrainExtent", help="Training Extent: downtown/all", default='downtown')
    parser.add_argument("-e", "--Epoch", help="Number of training epochs", default=500)
    parser.add_argument("-lr", "--LearningRate", help="Learning rate", default=5e-4)
    parser.add_argument("-ms", "--ModelNumberStart", help="Start number of models for each combination of the parameters", type=int, default=-1)
    parser.add_argument("-me", "--ModelNumberEnd", help="End number of models for each combination of the parameters", type=int, default=10)
    parser.add_argument("-s", "--IncludeSpatial", help="Whether to include spatial (demo+poi) information", default=0)
    parser.add_argument("-b", "--Bootstrap", help="Whether to bootstrap training data", default=0)
    parser.add_argument("-of", "--OutFolder", help="Output Folder", default="")
    parser.add_argument("-sv", "--Save", help="Whether to save models", default=0)
    parser.add_argument("-dev", "--Device", help="Cuda device to run on", default="cuda:0")
    parser.add_argument("-tb", "--Tensorboard", help="Whether to write to tensorboard", default=0)
    
    args = parser.parse_args(['@arguments_c.txt'])
    
    args.TrainStartDate = args.TrainStartDate.split(',')
    args.TrainEndDate = args.TrainEndDate.split(',')
    args.TestStartDate = args.TestStartDate.split(',')
    args.TestEndDate = args.TestEndDate.split(',')
    args.TrainExtent = args.TrainExtent.split(',')
    args.Lookback = [int(x) for x in args.Lookback.split(',')]
    args.TimeSize = [int(x) for x in args.TimeSize.split(',')]
    args.Dist = args.Dist.split(',')
    args.Difference = bool(int(args.Difference))
    args.MeanOnly = bool(int(args.MeanOnly))
    args.Homoskedastic = float(args.Homoskedastic)
    args.IncludeSpatial = bool(int(args.IncludeSpatial))
    args.Save = bool(int(args.Save))
    args.Bootstrap = bool(int(args.Bootstrap))
    args.Tensorboard = bool(int(args.Tensorboard))
    
    dropout_rate_list=[0.1,0.3]
    n_hid_units_list=[64]
    weight_decay_list = [0.01,0.05,0.1,0.5]
    ngc_list = [1]
    nlstm_list = [1]
    
    hps = list(itertools.product(dropout_rate_list, n_hid_units_list, weight_decay_list, nlstm_list, ngc_list))
    max_lookback = 6

    if args.ModelNumberStart == -1:
        run_all = True
    else:
        run_all = False

    predict_hzn = int(args.PredictHorizon)
    difference = args.Difference
    adj_type = args.Adj
    dataset = args.Dataset
    
    device = torch.device(args.Device if torch.cuda.is_available() else "cpu")
    
    for (trainst, trained, testst, tested, dist, train_extent, time_size) in itertools.product(args.TrainStartDate, args.TrainEndDate, args.TestStartDate, args.TestEndDate, args.Dist, args.TrainExtent, args.TimeSize): 
        
        if (dist=='norm')&(args.Homoskedastic>0):
            dist_save = 'norm_homo'+str(args.Homoskedastic)
        else:
            dist_save = dist
        
        if dist == 'poisson':
            meanonly=True
        else:
            meanonly=args.MeanOnly

        if dataset == 'rail_catchment':
            n_time = (96-28) // time_size
        elif dataset == 'census_tract':
            n_time = (96-20) // time_size


        if args.Tensorboard:
            writer_hp = SummaryWriter(project_dir+"tensorboard/runs/"+args.OutFolder+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/")

        for lookback in args.Lookback:

            print("Train Period:", trainst, trained, "Test Period:", testst, tested, "\nPredict Horizon:", predict_hzn, "Time Size:", time_size, "Lookback:", lookback, "\nadj:", '_'.join(adj_type), "Dist:", dist_save, "Train Extent:", train_extent)
            
            z = 0.95
                
            data, adj, spatial, downtown_filter = \
                    util_data.combine_datasources(project_dir, trainst, trained, testst, tested, predict_hzn, time_size, difference, max_lookback, lookback, args.Dataset)
            
            trainloader, trainloader_test, valloader, testloader, adj_torch, spatial_torch, y_train_eval, y_val_eval, y_test_eval = \
                    util_data.prepare_for_torch(device, train_extent, data, adj, spatial, downtown_filter, adj_type, bootstrap=args.Bootstrap)
            (num_train, _, _, n_modes) = data['x'][0].shape
            (num_val, _, _, n_modes) = data['x'][1].shape
            num_test = len(y_test_eval)
            n_stations = adj_torch.shape[0]
            y_train_eval = np.squeeze(y_train_eval)
            y_val_eval = np.squeeze(y_val_eval)
            y_test_eval = np.squeeze(y_test_eval)
    
#             print(np.mean(y_train_eval)*4)
#             print(np.mean(y_test_eval)*4)
            
            if args.Homoskedastic > 0:
                std = torch.tensor([args.Homoskedastic*np.mean(y_train_eval)])
                std = std.to(device)
            else:
                std = None
 
            same_hp = False
            if not args.IncludeSpatial:
                spatial_torch = None
                ndemo = 0
            else:
                ndemo = spatial.shape[1]

            if not os.path.isdir(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/"):
                os.mkdir(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/")
                os.mkdir(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/training_plots/")

                
            if run_all:
                if os.path.isfile(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/results.csv"):
                    results = pd.read_csv(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/results.csv")
                    hp_run = results[(results['Model Type']=="GCN")&\
                                     (results['Distribution']==dist_save)&(results['Predict Horizon']==predict_hzn)&(results['Time Size']==time_size)&(results['Adjacency']=='_'.join(adj_type))& \
                            (results['Lookback']==lookback)&(results['Train Extent']==train_extent)&(results['spatial']==int(args.IncludeSpatial))] 
                else:
                    hp_run = []
                    
                if len(hp_run) != 0:
                    model_offset = hp_run['Model'].max()+1
                    hp_run = hp_run[['dropout','n_hid_units','weight_decay','nlstm','ngc']].to_numpy()
                    hp_run = [tuple(l) for l in hp_run]
                else:
                    model_offset = 0

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

                save_dir = model_dir+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_GCN_"+dist_save+"/"
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                    
                save_dir += train_extent+"_"+'-'.join(adj_type)+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+str(lookback)+"_"+str(ii+model_offset-model_offset_2)
                if args.Tensorboard:
                    # need to update
                    writer = SummaryWriter(project_dir+"tensorboard/runs/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/"+str(lookback)+"_"+str(int(ii+model_offset-model_offset_2))+"/")

                best = 0
                best_epoch = 0

                dropout = hps[hp_index][0]
                n_hid_units = hps[hp_index][1]
                weight_decay = hps[hp_index][2]
                nlstm = hps[hp_index][3]
                ngc = hps[hp_index][4]

                print('Model ', int(ii+model_offset-model_offset_2))

                if args.LossFunc == 'mve':
                    criterion = MVELoss(dist)
                elif args.LossFunc == 'mse':
                    criterion = nn.MSELoss()
                else:
                    print("loss function not valid.")
                    sys.exit()

                net = GCN_LSTM(meanonly=meanonly, homo=args.Homoskedastic, nadj = len(adj_type), nmode=n_modes, 
                        nstation=n_stations, ntime=n_time, ndemo=ndemo,
                        nhid_g=n_hid_units, ngc=ngc, nhid_l=n_hid_units, nlstm=nlstm, 
                        nhid_fc=n_hid_units, dropout=dropout, std_starter=std)
                net = net.to(device)

                optimizer = optim.Adam(net.parameters(), lr=args.LearningRate, weight_decay=weight_decay)
                ref1 = 0
                ref2 = 0
                success=True
                train_loss_list = []
                val_loss_list = []
                test_loss_list = []

                start_runtime = datetime.now()
                
                for epoch in range(args.Epoch):

                    running_loss = 0.0

                    for i, batch_data in enumerate(trainloader, 0):

                        net.train()
                        
                        # get the inputs; data is a list of [inputs, labels, history, quarter of the day]
                        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod = batch_data
                        batch_size=len(batch_x)

                        batch_x = batch_x.float()
                        batch_y = torch.squeeze(batch_y).float()
                        batch_history = batch_history.float()
                        batch_qod = batch_qod.view(-1,1)
                        batch_qod_onehot = torch.FloatTensor(batch_size, n_time)
                        batch_qod_onehot.zero_()
                        batch_qod_onehot.scatter_(1, batch_qod, 1)
                            
                        batch_x, batch_y, batch_history, batch_weather, batch_los, batch_qod_onehot = \
                                batch_x.to(device), batch_y.to(device), batch_history.to(device), \
                                batch_weather.to(device), batch_los.to(device), batch_qod_onehot.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward 
                        outputs = net(batch_x, None, adj_torch, batch_history, spatial_torch, 
                                batch_weather, batch_los, batch_qod_onehot, 1)
                        
                        if meanonly:
                            output_loc = outputs
                            loss = criterion(output_loc, target=batch_y)
                        else:
                            if args.Homoskedastic == 0:
                                output_loc = outputs[:batch_size,:]
                                output_scale = outputs[batch_size:,:]
                            else:
                                output_loc = outputs
                                output_scale = std

                            loss = criterion(output_loc, output_scale, batch_y)

                        # backward
                        loss.backward()

                        # optimize
                        optimizer.step()

                        running_loss += loss.item()

                        if torch.sum(torch.isnan(output_loc)) > 0:
                            success=False
                            break

                    if not success:
                        break
                    
                    if epoch % 5 == 0:
                        
                        net.eval()

                        tr_loss, tr_mae, tr_mape, tr_mpiw, tr_picp = evaluate(args, dist, trainloader_test, z, y_train_eval, std=std, time_size=time_size)
                        val_loss, val_mae, val_mape, val_mpiw, val_picp = evaluate(args, dist, valloader, z, y_val_eval, std=std, time_size=time_size)
                        test_loss, test_mae, test_mape, test_mpiw, test_picp = evaluate(args, dist, testloader, z, y_test_eval, std=std, time_size=time_size)
                        print('[%d] training loss: %.3f, %.1f, %.3f' % (epoch + 1, tr_loss/num_train, tr_mae, tr_mape))
#                         print(output_loc.mean(), output_loc.min(), output_scale.max())
#                         print(output_scale.mean(), output_scale.min(), output_scale.max())
                        
                        if args.Tensorboard:
                            writer.add_scalar("Loss/Train", tr_loss/num_train, epoch+1)
                            writer.add_scalar("MAE/Train", tr_mae, epoch+1)
                            writer.add_scalar("MPIW/Train", tr_mpiw, epoch+1)
                            writer.add_scalar("PICP/Train", tr_picp, epoch+1)

                            writer.add_scalar("Loss/Test", test_loss/num_test, epoch+1)
                            writer.add_scalar("MAE/Test", test_mae, epoch+1)
                            writer.add_scalar("MPIW/Test", test_mpiw, epoch+1)
                            writer.add_scalar("PICP/Test", test_picp, epoch+1)

                        if epoch > 40:
                            if running_loss < best:
                                best = running_loss
                                best_epoch = epoch
                                output_string = (tr_loss/num_train, tr_mae, tr_mape, tr_mpiw, tr_picp,
                                    val_loss/num_val, val_mae, val_mape, val_mpiw, val_picp,
                                    test_loss/num_test, test_mae, test_mape, test_mpiw, test_picp)
                            if (np.abs(running_loss - ref1)/ref1<0.001) & (np.abs(running_loss - ref2)/ref2<0.001):
                                print("early stopping at epoch", epoch)
                                print("best epoch:", best_epoch)
                                break
                            if (ref1 < running_loss) & (ref2 < ref1):
                                print("diverging. stop.")
                                break
                                
                        else:
                            best = running_loss
                            best_epoch = epoch
                            output_string = (tr_loss/num_train, tr_mae, tr_mape, tr_mpiw, tr_picp,
                                    val_loss/num_val, val_mae, val_mape, val_mpiw, val_picp,
                                    test_loss/num_test, test_mae, test_mape, test_mpiw, test_picp)

                        ref2 = ref1
                        ref1 = running_loss

                        if (args.Save) & (best_epoch==epoch):
                            torch.save({'epoch': epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'hyperparameters': (trainst, trained, testst, tested, 
                                                    train_extent, '_'.join(adj_type), ii, 
                                                    predict_hzn, time_size, lookback, 
                                                    dropout, n_hid_units, nlstm, ngc, weight_decay, args.Bootstrap),
                                'train_loss': train_loss_list,
                                'val_loss': val_loss_list,
                                'test_loss': test_loss_list},
                                save_dir+"_"+str(epoch)+".pt")

                            files = glob.glob(save_dir+"_*.pt")
                    
                        train_loss_list.append(tr_loss/num_train)
                        val_loss_list.append(val_loss/num_val)
                        test_loss_list.append(test_loss/num_test)

                        
                if args.Save:
                    if best_epoch==epoch:
                        torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'hyperparameters': (trainst, trained, testst, tested, 
                                                train_extent, '_'.join(adj_type), ii, 
                                                predict_hzn, time_size, lookback, 
                                                dropout, n_hid_units, nlstm, ngc, weight_decay, args.Bootstrap),
                            'train_loss': train_loss_list,
                            'val_loss': val_loss_list,
                            'test_loss': test_loss_list},
                            save_dir+"_"+str(epoch)+".pt")

                        files = glob.glob(save_dir+"_*.pt")
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
                    
                if args.Tensorboard:
                    hparams_dict = {'train_extent':train_extent, 'adjacency': '_'.join(adj_type), 
                                        'prediction_horizon': predict_hzn, 'time_size': time_size, 'lookback': lookback, 
                                        'dropout': dropout, 'n_hid_units': n_hid_units, 'nlstm': nlstm, 'ngc': ngc, 'weight_decay': weight_decay, 'bootstrap': args.Bootstrap}
                    metrics_dict = {"loss_train": tr_loss/num_train, "mae_train": tr_mae, "mape_train": tr_mape, "mpiw_train": tr_mpiw, "picp_train": tr_picp,
                                    "loss_val": val_loss/num_train, "mae_val": val_mae, "mape_val": val_mape, "mpiw_val": val_mpiw, "picp_val": val_picp,
                                    "loss_test": test_loss/num_test, "mae_test": test_mae, "mape_test":test_mape, "mpiw_test": test_mpiw, "picp_test": test_picp}
                    writer_hp.add_hparams(hparams_dict, metrics_dict, run_name = str(lookback)+"_"+str(ii+model_offset-model_offset_2))

                runtime = datetime.now() - start_runtime
                runtime = runtime.total_seconds()/60
                
                # Write to File
                if not os.path.isdir(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/"):
                    os.mkdir(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/")
                
                fig, ax = plt.subplots(figsize=(4,3))
                ax.plot(np.arange(0,epoch,5), train_loss_list, color='cornflowerblue', label='Train')
                ax.plot(np.arange(0,epoch,5), val_loss_list, color='forestgreen', label='Validation')
                ax.plot(np.arange(0,epoch,5), test_loss_list, color='sandybrown', label='Test')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_ylim([0.9*np.min(train_loss_list+val_loss_list+test_loss_list), 1.1*np.max(train_loss_list+val_loss_list+test_loss_list)])
                ax.legend()
                fig.savefig(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/training_plots/"+dist_save+"_"+train_extent+"_"+str(ii+model_offset-model_offset_2)+"_"+str(time_size)+".png", bbox_inches='tight')
                
                with open(project_dir+"results/"+args.OutFolder+"_"+dataset.split("_")[0]+"_"+trainst+"-"+trained+"_"+testst+"-"+tested+"/results.csv","a") as f:
                    f.write("%s,%s,%s,%s,%d,%d,%d,%d,%d,%.2f,%d,%d,%d,%.2E,%d,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f\n" % 
                           (("GCN", dist_save, train_extent, '_'.join(adj_type), int(ii+model_offset-model_offset_2), 
                                predict_hzn, time_size, lookback, args.Bootstrap, 
                                dropout, n_hid_units, nlstm, ngc, weight_decay, spatial_torch is not None) + output_string + (runtime,)))

