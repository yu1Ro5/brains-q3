import csv
import datetime
import os
import pickle
import time
import warnings

import joblib
import lightgbm as lgb
import pandas as pd

import util

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    start = time.time()
    df=pd.read_csv("datasets/dataset.csv")
    X,y=util.prepare_data(df)
    prepared_time = time.time() - start
    print("prepared_time:{0}".format(prepared_time) + "[sec]")
    print(X.shape,y.shape)
    dall=lgb.Dataset(data=X,label=y)
    best_params={
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'feature_pre_filter': False,
        'lambda_l1': 8.546942828085179e-06,
        'lambda_l2': 1.0468971074369936e-07,
        'num_leaves': 31,
        'feature_fraction': 0.4,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'min_child_samples': 20,
        'force_col_wise':'true'
    }
    lgb_cv = lgb.cv(
        params=best_params,
        train_set=dall,
        num_boost_round=1000,
        # folds=KFold(n_splits=5),
        nfold=5,
        stratified=False,
        shuffle=True,
        metrics=["mean_squared_error"],
        early_stopping_rounds=100,
        seed=42,
        return_cvbooster=True
    )
    print("l2-mean: ",lgb_cv['l2-mean'][-1],"±",lgb_cv['l2-stdv'][-1])
    joblib.dump(value=lgb_cv['cvbooster'].boosters,filename=open(file=os.path.dirname(__file__)+"/lgb_cv_boosters.joblib",mode="wb"),compress=3)
    joblib.dump(value=lgb_cv['cvbooster'].best_iteration,filename=open(file=os.path.dirname(__file__)+"/lgb_cv_best_iter.joblib",mode="wb"),compress=3)
    now_time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    with open('log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now_time_str,X.shape[1],prepared_time,lgb_cv['l2-mean'][-1],lgb_cv['l2-stdv'][-1]])
