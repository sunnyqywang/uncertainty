import numpy as np
import util_eval
import statsmodels.api as sm

def WLS(x_train, x_test, ref_train, ref_test, y_train, y_test, 
        model, extent,
        ts_train_onehot=None, ts_test_onehot=None, time_specific_coefs=[], lookback=2, z=1.96, 
        project_dir='', out_folder='', out_file=None, stdout=True):
    
    # x_train: time, lookback, station, mode
    # ols_x_train: time*station, lookback*mode

    _,_,n_station,n_mode = x_train.shape
    ols_x_train = x_train
    ols_x_train = ols_x_train.reshape((-1, lookback*n_mode))
    for i in ref_train:
        ols_x_train = np.concatenate((ols_x_train,np.reshape(i, (len(ols_x_train), -1))), axis=1)
    temp = np.ones(len(ols_x_train)).reshape((-1,1))
    for i in range(ols_x_train.shape[1]):
        if i in time_specific_coefs:
            temp = np.concatenate((temp, ts_train_onehot*ols_x_train[:,i:i+1]), axis=1)
        else:
            temp = np.concatenate((temp, ols_x_train[:,i:i+1]), axis=1)
    ols_x_train = temp

    ols_x_test = x_test
    ols_x_test = ols_x_test.reshape((-1, lookback*n_mode))
    for i in ref_test:
        ols_x_test = np.concatenate((ols_x_test,np.reshape(i, (len(ols_x_test), -1))), axis=1)
    temp = np.ones(len(ols_x_test)).reshape((-1,1))
    for i in range(ols_x_test.shape[1]):
        if i in time_specific_coefs:
            temp = np.concatenate((temp, ts_test_onehot*ols_x_test[:,i:i+1]), axis=1)
        else:
            temp = np.concatenate((temp, ols_x_test[:,i:i+1]), axis=1)    
    ols_x_test = temp

    ols_y_train = y_train.flatten()
    ols_y_test = y_test.flatten()
    
    print("Sample Size (Train):", ols_x_train.shape[0])
    print("Number of Variables:", ols_x_train.shape[1])
    print()
    # fit OLS on mean (consistent but inefficient)
    ols_mean = sm.OLS(ols_y_train, ols_x_train)
    ols_mean_results = ols_mean.fit()
    ols_pred_train = ols_mean.predict(ols_mean_results.params, ols_x_train)

    # fit OLS on transformed residuals
    ols_res = np.log(np.power(ols_y_train - ols_pred_train,2))
    ols_std = sm.OLS(ols_res, ols_x_train)
    ols_std_results = ols_std.fit()

    ols_pred_test = ols_mean.predict(ols_mean_results.params, ols_x_test)
    
    ols_pred_std_train = np.sqrt(np.exp(ols_std.predict(ols_std_results.params, ols_x_train)))
    ols_pred_std_test = np.sqrt(np.exp(ols_std.predict(ols_std_results.params, ols_x_test)))

    tr_mae, tr_mse, tr_nz_mae, tr_nz_mse, tr_pct_nonzeros = \
        util_eval.eval_mean(ols_pred_train, ols_y_train, 'Train', stdout=stdout)
    mae, mse, nz_mae, nz_mse, pct_nonzeros = \
        util_eval.eval_mean(ols_pred_test, ols_y_test, 'Test', stdout=stdout)

    if stdout:
        print("(Train)")
    tr_u, tr_ub, tr_uv, tr_uc = util_eval.eval_theils(ols_pred_train, ols_y_train, stdout = stdout)
    if stdout:
        print("(Test)")
    u, ub, uv, uc = util_eval.eval_theils(ols_pred_test, ols_y_test, stdout = stdout)
    
    tr_mpiw, tr_picp = util_eval.eval_pi(ols_pred_train-z*ols_pred_std_train,
                                         ols_pred_train+z*ols_pred_std_train,ols_y_train)
    mpiw, picp = util_eval.eval_pi(ols_pred_test-z*ols_pred_std_test,ols_pred_test+z*ols_pred_std_test,ols_y_test)
    
    if stdout:
        print("\n(Train)")
        print("Mean Prediction Interval: %.2f"%(tr_mpiw))
        print("Coverage Probability: %.4f"%(tr_picp))
        print("(Test)")
        print("Mean Prediction Interval: %.2f"%(mpiw))
        print("Coverage Probability: %.4f"%(picp))
    
    if out_file is not None:
        with open(project_dir+"results/"+out_folder+out_file,"a") as f:
            f.write("%s,%s,%s,%d,%d,%d,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n" % \
                    (period, 'WLS'+model, extent, 
                    predict_hzn, time_size, lookback, 
                    tr_mae, tr_mse, tr_u, tr_ub, tr_uv, tr_uc, tr_mpiw, tr_picp, 
                    mae, mse, u, ub, uv, uc, mpiw, picp, ols_x_train.shape[1]))

    return ols_pred_test, ols_pred_std_test, (mae, mse, u, ub, uv, uc, mpiw, picp)