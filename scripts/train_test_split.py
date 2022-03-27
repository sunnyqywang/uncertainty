import argparse
import itertools
import sys
sys.path.append('../')

import libpysal
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split

from util_data import generate_time_series
from util_data import get_reference

if __name__ == "__main__":
    
    data_dir = "/home/jtl/Dropbox (MIT)/project_uncertainty_quantification/data/"
    max_lookback = 6
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--Period", help="Study period: before/after")    
    parser.add_argument("-ph", "--PredictHorizon", help="Predict horizon")
    parser.add_argument("-ts", "--TimeSize", help="Time aggregation")
    parser.add_argument("-d", "--Difference", help="Whether to difference")
    
    data_dir = "/home/jtl/Dropbox (MIT)/project_uncertainty_quantification/data/"

    args = parser.parse_args()
    
    if args.Period is None:
        period_list = ['before']
    else:
        period_list = args.Period.split(",")
    if args.PredictHorizon is None:
        predict_hzn_list = [2]
    else:
        predict_hzn_list = list(map(int,args.PredictHorizon.split(",")))
    if args.TimeSize is None:
        time_size_list = [4]
    else:
        time_size_list = list(map(int,args.TimeSize.split(",")))
    if args.Difference is None:
        difference = True
    else:
        difference = list(map(bool,args.Difference.split(",")))
        

    for (period, predict_hzn, time_size) in itertools.product(period_list, predict_hzn_list, time_size_list):
        dates = pd.read_csv(data_dir+period+"_dates.csv")
        ref_period = 7 * (96//time_size)
        
        # read in ridership data
        rail = pd.read_csv(data_dir+"data_processed/rail_catchment/"+period+"/rail_df.csv")
        bus = pd.read_csv(data_dir+"data_processed/rail_catchment/"+period+"/bus_rail_df.csv")
        tnc = pd.read_csv(data_dir+"data_processed/rail_catchment/"+period+"/tnc_rail_df.csv")
        rail_los = pd.read_csv(data_dir+"data_processed/rail_catchment/"+period+"/rail_count_df.csv")
        
        # merge into one df
        df = pd.merge(rail, bus, how='left', left_on=['ts','station_id'], right_on=['ts','STATION_ID'])
        df = pd.merge(df, tnc, how='left', left_on=['ts','station_id'], right_on=['ts','STATION_ID'])
        df = pd.merge(df, rail_los, how='inner', left_on=['ts','station_id'], right_on=['ts','station_id'])
        df.fillna(0, inplace=True)
        
        # temporal aggregation
        df['ts'] = df['ts'] // time_size
        df = df.groupby(['ts','station_id'], as_index=False).sum()

        # libpysal will automatically order stations in ascending order when reading adj matrix
        # sort to match
        df.sort_values(by=['ts','station_id'], inplace=True)

        # form vectors for train test split
        rail_pivot = pd.pivot_table(df, index='ts', columns='station_id', values='count', aggfunc=sum, fill_value=0)
        bus_pivot = pd.pivot_table(df, index='ts', columns='station_id', values='bus_count', aggfunc=sum, fill_value=0)
        tnc_pivot = pd.pivot_table(df, index='ts', columns='station_id', values='tnc_count', aggfunc=sum, fill_value=0)
        rail_los_pivot = pd.pivot_table(df, index='ts', columns='station_id', values='num_schd_trp_15min', aggfunc=sum, fill_value=0)

        station_id_list = rail_pivot.columns.tolist()

        timestamps = rail_pivot.index.to_numpy()
        ts_hours = (timestamps % (96//time_size)) / time_size * 4
        mask = (ts_hours >=6) & (ts_hours <= 22)

        # generate time series data w.r.t. lookback and prediction horizon
        x, y, los, ts, ref = generate_time_series(data=[rail_pivot,bus_pivot,tnc_pivot], 
                                           targets=[rail_pivot], 
                                           others=[rail_los_pivot], 
                                           ts=timestamps,
                                           offset=predict_hzn, 
                                           lookback=max_lookback,
                                           ref_ts=ref_period,
                                           difference=difference,
                                           remove_list=timestamps[~mask].tolist())

        # there is only one target, others and reference
        y = y[0]
        los = los[0]
        ref = ref[0]

        # process weather information
        ts_day = pd.DataFrame(ts // (96//time_size), columns=['DAY_INDEX'])

        weather = pd.read_csv(data_dir+"data_processed/weather_"+period+".csv")
        weather['TEMP_PCT_DIFF'] = np.abs((weather['TAVG'] - weather['TAVGAVG']) / weather['TAVGAVG'])
        weather['PRCP'] = weather['PRCP'] / weather['PRCP'].max()

        weather = pd.merge(ts_day, weather, on='DAY_INDEX') [['PRCP','TEMP_PCT_DIFF']].to_numpy()
        assert len(weather) == len(ts_day)

        # train test split
        test_size = 0.10
        x_train, x_test, y_train, y_test, ts_train, ts_test, ref_train, ref_test, los_train, los_test, w_train, w_test = \
                train_test_split(x, y, ts, ref, los, weather, test_size=test_size, shuffle=False)

        n_stations = x_train.shape[-2]

        # val test split
        val_test_split = 0.5
        x_val, x_test, y_val, y_test, ts_val, ts_test, ref_val, ref_test, los_val, los_test, w_val, w_test = \
                train_test_split(x_test, y_test, ts_test, ref_test, los_test, w_test, 
                                 test_size=val_test_split, shuffle=False)
        
        if difference:
            differenced = "diff"
        else:
            differenced = "raw"

        with open(data_dir+"data_processed/rail_catchment/"+period+"/"+
                  period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_train.pkl","wb") as f:
            pkl.dump(x_train, f)
            pkl.dump(ref_train,f)
            pkl.dump(los_train, f)
            pkl.dump(w_train, f)
            pkl.dump(y_train, f)
            pkl.dump(ts_train, f)
            pkl.dump(station_id_list, f)

        with open(data_dir+"data_processed/rail_catchment/"+period+"/"+
                  period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_val.pkl","wb") as f:
            pkl.dump(x_val, f)
            pkl.dump(ref_val,f)
            pkl.dump(los_val, f)
            pkl.dump(w_val, f)
            pkl.dump(y_val, f)
            pkl.dump(ts_val, f)
            pkl.dump(station_id_list, f)
            
        with open(data_dir+"data_processed/rail_catchment/"+period+"/"+
                  period+"_"+str(predict_hzn)+"_"+str(time_size)+"_"+differenced+"_data_test.pkl","wb") as f:
            pkl.dump(x_test, f)
            pkl.dump(ref_test,f)
            pkl.dump(los_test, f)
            pkl.dump(w_test, f)
            pkl.dump(y_test, f)
            pkl.dump(ts_test, f)
            pkl.dump(station_id_list, f)



