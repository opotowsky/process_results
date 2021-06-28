#! /usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error

def MAPE(y_true,y_pred):
    ape = np.abs((y_true - y_pred)/y_true)*100
    std = ape.std()
    mape = np.mean(ape)
    return mape, std

def conf_int(metric, n):
    # for 1 std dev, 68% conf
    # can change to 95%, which is 1.96 std dev
    z = 1 
    ci = z * np.sqrt(metric * (1 - metric) / n)
    return ci

def rxtr_metrics(df, idx, mll, knn, dtr, pred, metric):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Score'
    errname = 'AbsError'    
    for en_list in ['_short', '_auto', '_long']:
        if metric = 'BalAcc':
            dfmetric = 'Balanced Accuracy'
            dfstd = 'BalAcc CI'
            ### MLL ###
            bal_acc = balanced_accuracy_score(mll[en_list][predmll[pred]],
                                              mll[en_list]['pred_' + predmll[pred]],
                                              adjusted=True)
            df.loc[idx, ('mll'+en_list, dfmetric)] = bal_acc
            df.loc[idx, ('mll'+en_list, dfstd)] = conf_int(bal_acc, len(mll[en_list]))
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                bal_acc = balanced_accuracy_score(alg[en_list]['TrueY'], 
                                                  alg[en_list][A], 
                                                  adjusted=True)
                df.loc[idx, (a+en_list, dfmetric)] = bal_acc 
                df.loc[idx, (a+en_list, dfstd)] = conf_int(bal_acc, len(alg[en_list]))
        elif metric == 'Acc':
            dfmetric = 'Accuracy'
            dfstd = 'Acc CI'
            ### MLL ###
            acc = accuracy_score(mll[en_list][predmll[pred]],
                                 mll[en_list]['pred_' + predmll[pred]])
            df.loc[idx, ('mll'+en_list, dfmetric)] = acc 
            df.loc[idx, ('mll'+en_list, dfstd)] = conf_int(acc, len(mll[en_list]))
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                acc = accuracy_score(alg[en_list]['TrueY'], alg[en_list][A])
                df.loc[idx, (a+en_list, dfmetric)] = acc
                df.loc[idx, (a+en_list, dfstd)] = conf_int(acc, len(alg[en_list]))
        else: #Other classification metrics
            # TODO
            df = df
    return df

def reg_metrics(df, idx, mll, knn, dtr, pred, metric):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Error'
    errname = 'AbsError'    
    for en_list in ['_short', '_auto', '_long']:
        if metric == 'MAE':
            dfmetric = 'Neg MAE'
            dfstd = 'MAE Std'
            ### MLL ###
            df.loc[idx, ('mll'+en_list, dfmetric)] = -mean_absolute_error(mll[en_list][predmll[pred]], 
                                                                          mll[en_list]['pred_' + predmll[pred]])
            df.loc[idx, ('mll'+en_list, dfstd)] = mll[en_list][predmll[pred] + llmetric].std()
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[idx, (a+en_list, dfmetric)] = -mean_absolute_error(alg[en_list]['TrueY'], 
                                                                          alg[en_list][A])
                df.loc[idx, (a+en_list, dfstd)] = alg[en_list][errname].std()
        elif metric == 'MedAE':
            dfmetric1 = 'Neg MedAE SK'
            dfmetric2 = 'Neg MedAE'
            dfiqr1 = 'MedAE IQR_25'
            dfiqr2 = 'MedAE IQR_75'
            ### MLL ###
            q = ['25%', '75%']
            med = '50%'
            df.loc[idx, ('mll'+en_list, dfmetric1)] = -median_absolute_error(mll[en_list][predmll[pred]], 
                                                                             mll[en_list]['pred_' + predmll[pred]])
            col = mll[en_list][predmll[pred] + llmetric]
            df.loc[idx, ('mll'+en_list, dfmetric2)] = -col.describe()[med]
            df.loc[idx, ('mll'+en_list, dfiqr1)] = -col.describe()[q[0]]
            df.loc[idx, ('mll'+en_list, dfiqr2)] = -col.describe()[q[1]]
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[idx, (a+en_list, dfmetric1)] = -median_absolute_error(alg[en_list]['TrueY'], 
                                                                             alg[en_list][A])
                col = alg[en_list][errname]
                df.loc[idx, (a+en_list, dfmetric2)] = -col.describe()[med]
                df.loc[idx, (a+en_list, dfiqr1)] = -col.describe()[q[0]]
                df.loc[idx, (a+en_list, dfiqr2)] = -col.describe()[q[1]]
        else: #MAPE
            dfmetric1 = 'Neg MAPE SK'
            dfmetric2 = 'Neg MAPE'
            dfstd = 'MAPE Std'
            ### MLL ###
            df.loc[idx, ('mll'+en_list, dfmetric1)] = -mean_absolute_percentage_error(mll[en_list][predmll[pred]], 
                                                                                      mll[en_list]['pred_' + predmll[pred]])
            mape, std = MAPE(mll[en_list][predmll[pred]], mll[en_list]['pred_' + predmll[pred]])
            df.loc[idx, ('mll'+en_list, dfmetric2)] = -mape
            df.loc[idx, ('mll'+en_list, dfstd)] = std
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[idx, (a+en_list, dfmetric1)] = -mean_absolute_percentage_error(alg[en_list]['TrueY'], 
                                                                                      alg[en_list][A])
                mape, std = MAPE(alg[en_list]['TrueY'], alg[en_list][A])
                df.loc[idx, (a+en_list, dfmetric2)] = -mape
                df.loc[idx, (a+en_list, dfstd)] = std
    return df

def rxtr_randerr(df, idx, mll, knn, dtr, pred, metric):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Score'
    errname = 'AbsError'    
    mll_errs = [1, 5, 10, 15, 20]
    sk_errs = [0, 0.3, 0.7, 1, 2, 4, 6, 8, 10, 13, 17, 20]
    if metric = 'BalAcc':
        dfmetric = 'Balanced Accuracy'
        dfstd = 'BalAcc CI'
        ### MLL ###
        if idx in mll_errs:
            bal_acc = balanced_accuracy_score(mll[predmll[pred]],
                                              mll['pred_' + predmll[pred]],
                                              adjusted=True)
            df.loc[idx, ('mll', dfmetric)] = bal_acc
            df.loc[idx, ('mll', dfstd)] = conf_int(bal_acc, len(mll))
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                bal_acc = balanced_accuracy_score(alg['TrueY'], alg[A], adjusted=True)
                df.loc[idx, (a, dfmetric)] = bal_acc 
                df.loc[idx, (a, dfstd)] = conf_int(bal_acc, len(alg))
    elif metric == 'Acc':
        dfmetric = 'Accuracy'
        dfstd = 'Acc CI'
        ### MLL ###
        if idx in mll_errs:
            acc = accuracy_score(mll[predmll[pred]], mll['pred_' + predmll[pred]])
            df.loc[idx, ('mll', dfmetric)] = acc 
            df.loc[idx, ('mll', dfstd)] = conf_int(acc, len(mll))
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                acc = accuracy_score(alg['TrueY'], alg[A])
                df.loc[idx, (a, dfmetric)] = acc
                df.loc[idx, (a, dfstd)] = conf_int(acc, len(alg))
    else: #Other classification metrics
        # TODO
        df = df
    return df

def reg_randerr(df, idx, mll, knn, dtr, pred, metric):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Error'
    errname = 'AbsError'    
    mll_errs = [1, 5, 10, 15, 20]
    sk_errs = [0, 0.3, 0.7, 1, 2, 4, 6, 8, 10, 13, 17, 20]
    if metric == 'MAE':
        dfmetric = 'Neg MAE'
        dfstd = 'MAE Std'
        ### MLL ###
        if idx in mll_errs:
            df.loc[idx, ('mll', dfmetric)] = -mean_absolute_error(mll[predmll[pred]], 
                                                                  mll['pred_' + predmll[pred]])
            df.loc[idx, ('mll', dfstd)] = mll[predmll[pred] + llmetric].std()
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[idx, (a, dfmetric)] = -mean_absolute_error(alg['TrueY'], alg[A])
                df.loc[idx, (a, dfstd)] = alg[errname].std()
    elif metric == 'MedAE':
        dfmetric1 = 'Neg MedAE SK'
        dfmetric2 = 'Neg MedAE'
        dfiqr1 = 'MedAE IQR_25'
        dfiqr2 = 'MedAE IQR_75'
        ### MLL ###
        if idx in mll_errs:
            q = ['25%', '75%']
            med = '50%'
            df.loc[idx, ('mll', dfmetric1)] = -median_absolute_error(mll[predmll[pred]], 
                                                                     mll['pred_' + predmll[pred]])
            col = mll[predmll[pred] + llmetric]
            df.loc[idx, ('mll', dfmetric2)] = -col.describe()[med]
            df.loc[idx, ('mll', dfiqr1)] = -col.describe()[q[0]]
            df.loc[idx, ('mll', dfiqr2)] = -col.describe()[q[1]]
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[idx, (a, dfmetric1)] = -median_absolute_error(alg['TrueY'], alg[A])
                col = alg[errname]
                df.loc[idx, (a, dfmetric2)] = -col.describe()[med]
                df.loc[idx, (a, dfiqr1)] = -col.describe()[q[0]]
                df.loc[idx, (a, dfiqr2)] = -col.describe()[q[1]]
    else: #MAPE
        dfmetric1 = 'Neg MAPE SK'
        dfmetric2 = 'Neg MAPE'
        dfstd = 'MAPE Std'
        ### MLL ###
        if idx in mll_errs:
            df.loc[idx, ('mll', dfmetric1)] = -mean_absolute_percentage_error(mll[predmll[pred]], 
                                                                              mll['pred_' + predmll[pred]])
            mape, std = MAPE(mll[predmll[pred]], mll['pred_' + predmll[pred]])
            df.loc[idx, ('mll', dfmetric2)] = -mape
            df.loc[idx, ('mll', dfstd)] = std
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[idx, (a, dfmetric1)] = -mean_absolute_percentage_error(alg['TrueY'], alg[A])
                mape, std = MAPE(alg['TrueY'], alg[A])
                df.loc[idx, (a, dfmetric2)] = -mape
                df.loc[idx, (a, dfstd)] = std
    return df
