#! /usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def MAPE(y_true,y_pred):
    ape = np.abs((y_true - y_pred)/y_true)*100
    std = ape.std()
    mape = ape.mean()
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
    for en_list in ['short', 'auto', 'long']:
        if metric == 'BalAcc':
            dfmetric = 'Balanced Accuracy'
            dfstd = 'BalAcc CI'
            ### MLL ###
            bal_acc = balanced_accuracy_score(mll[en_list][predmll[pred]],
                                              mll[en_list]['pred_' + predmll[pred]],
                                              adjusted=True)
            df.loc[idx, ('mll_'+en_list, dfmetric)] = bal_acc
            df.loc[idx, ('mll_'+en_list, dfstd)] = conf_int(bal_acc, len(mll[en_list]))
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                bal_acc = balanced_accuracy_score(alg[en_list]['TrueY'], 
                                                  alg[en_list][A], 
                                                  adjusted=True)
                df.loc[idx, (a+'_'+en_list, dfmetric)] = bal_acc 
                df.loc[idx, (a+'_'+en_list, dfstd)] = conf_int(bal_acc, len(alg[en_list]))
        elif metric == 'Acc':
            dfmetric = 'Accuracy'
            dfstd = 'Acc CI'
            ### MLL ###
            acc = accuracy_score(mll[en_list][predmll[pred]],
                                 mll[en_list]['pred_' + predmll[pred]])
            df.loc[idx, ('mll_'+en_list, dfmetric)] = acc 
            df.loc[idx, ('mll_'+en_list, dfstd)] = conf_int(acc, len(mll[en_list]))
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                acc = accuracy_score(alg[en_list]['TrueY'], alg[en_list][A])
                df.loc[idx, (a+'_'+en_list, dfmetric)] = acc
                df.loc[idx, (a+'_'+en_list, dfstd)] = conf_int(acc, len(alg[en_list]))
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
    for en_list in ['short', 'auto', 'long']:
        if metric == 'MAE':
            dfmetric = 'Neg MAE'
            dfstd = 'MAE Std'
            ### MLL ###
            col = mll[en_list][predmll[pred] + llmetric]
            df.loc[idx, ('mll_'+en_list, dfmetric)] = -col.mean()
            df.loc[idx, ('mll_'+en_list, dfstd)] = col.std()
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                col = alg[en_list][errname]
                df.loc[idx, (a+'_'+en_list, dfmetric)] = -col.mean()
                df.loc[idx, (a+'_'+en_list, dfstd)] = col.std()
        elif metric == 'MedAE':
            dfmetric = 'Neg MedAE'
            dfiqr1 = 'IQR_25'
            dfiqr2 = 'IQR_75'
            ### MLL ###
            q = ['25%', '75%']
            med = '50%'
            col = mll[en_list][predmll[pred] + llmetric]
            df.loc[idx, ('mll_'+en_list, dfmetric)] = -col.describe()[med]
            df.loc[idx, ('mll_'+en_list, dfiqr1)] = -col.describe()[q[0]]
            df.loc[idx, ('mll_'+en_list, dfiqr2)] = -col.describe()[q[1]]
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                col = alg[en_list][errname]
                df.loc[idx, (a+'_'+en_list, dfmetric)] = -col.describe()[med]
                df.loc[idx, (a+'_'+en_list, dfiqr1)] = -col.describe()[q[0]]
                df.loc[idx, (a+'_'+en_list, dfiqr2)] = -col.describe()[q[1]]
        else: #MAPE
            dfmetric = 'Neg MAPE'
            dfstd = 'MAPE Std'
            ### MLL ###
            mape, std = MAPE(mll[en_list][predmll[pred]], mll[en_list]['pred_' + predmll[pred]])
            df.loc[idx, ('mll_'+en_list, dfmetric)] = -mape
            df.loc[idx, ('mll_'+en_list, dfstd)] = std
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                mape, std = MAPE(alg[en_list]['TrueY'], alg[en_list][A])
                df.loc[idx, (a+'_'+en_list, dfmetric)] = -mape
                df.loc[idx, (a+'_'+en_list, dfstd)] = std
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
    if metric == 'BalAcc':
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
            col = mll[predmll[pred] + llmetric]
            df.loc[idx, ('mll', dfmetric)] = -col.mean()
            df.loc[idx, ('mll', dfstd)] = col.std()
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                col = alg[errname]
                df.loc[idx, (a, dfmetric)] = -col.mean() 
                df.loc[idx, (a, dfstd)] = col.std()
    elif metric == 'MedAE':
        dfmetric = 'Neg MedAE'
        dfiqr1 = 'IQR_25'
        dfiqr2 = 'IQR_75'
        q = ['25%', '75%']
        med = '50%'
        ### MLL ###
        if idx in mll_errs:
            col = mll[predmll[pred] + llmetric]
            df.loc[idx, ('mll', dfmetric)] = -col.describe()[med]
            df.loc[idx, ('mll', dfiqr1)] = -col.describe()[q[0]]
            df.loc[idx, ('mll', dfiqr2)] = -col.describe()[q[1]]
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                col = alg[errname]
                df.loc[idx, (a, dfmetric)] = -col.describe()[med]
                df.loc[idx, (a, dfiqr1)] = -col.describe()[q[0]]
                df.loc[idx, (a, dfiqr2)] = -col.describe()[q[1]]
    else: #MAPE
        dfmetric = 'Neg MAPE'
        dfstd = 'MAPE Std'
        ### MLL ###
        if idx in mll_errs:
            mape, std = MAPE(mll[predmll[pred]], mll['pred_' + predmll[pred]])
            df.loc[idx, ('mll', dfmetric)] = -mape
            df.loc[idx, ('mll', dfstd)] = std
        ### Scikit ###
        if idx in sk_errs:
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                mape, std = MAPE(alg['TrueY'], alg[A])
                df.loc[idx, (a, dfmetric)] = -mape
                df.loc[idx, (a, dfstd)] = std
    return df

def reg_rxtr_type(df, pred, rxtr, mll, knn, dtr):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Error'
    errname = 'AbsError'    
    
    ### 1. MAE ###
    dfmetric = 'MAE'
    dfstd = 'MAE Std'
    ### MLL ###
    col = mll[predmll[pred] + llmetric]
    df.loc[(pred, rxtr), ('mll', dfmetric)] = col.mean()
    df.loc[(pred, rxtr), ('mll', dfstd)] = col.std()
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        col = alg[errname]
        df.loc[(pred, rxtr), (a, dfmetric)] = col.mean() 
        df.loc[(pred, rxtr), (a, dfstd)] = col.std()
    
    ### 2. MedAE ###
    dfmetric = 'MedAE'
    dfiqr1 = 'IQR_25'
    dfiqr2 = 'IQR_75'
    q = ['25%', '75%']
    med = '50%'
    ### MLL ###
    col = mll[predmll[pred] + llmetric]
    df.loc[(pred, rxtr), ('mll', dfmetric)] = col.describe()[med]
    df.loc[(pred, rxtr), ('mll', dfiqr1)] = col.describe()[q[0]]
    df.loc[(pred, rxtr), ('mll', dfiqr2)] = col.describe()[q[1]]
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        col = alg[errname]
        df.loc[(pred, rxtr), (a, dfmetric)] = col.describe()[med]
        df.loc[(pred, rxtr), (a, dfiqr1)] = col.describe()[q[0]]
        df.loc[(pred, rxtr), (a, dfiqr2)] = col.describe()[q[1]]
    
    ### 3. MAPE ###
    dfmetric = 'MAPE'
    dfstd = 'MAPE Std'
    ### MLL ###
    mape, std = MAPE(mll[predmll[pred]], mll['pred_' + predmll[pred]])
    df.loc[(pred, rxtr), ('mll', dfmetric)] = mape
    df.loc[(pred, rxtr), ('mll', dfstd)] = std
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        mape, std = MAPE(alg['TrueY'], alg[A])
        df.loc[(pred, rxtr), (a, dfmetric)] = mape
        df.loc[(pred, rxtr), (a, dfstd)] = std
    return df

def rxtr_sfco(df, case, pred, mll, knn, dtr):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Score'
    errname = 'AbsError'    
    
    ### 1. Bal Accuracy ###    
    dfmetric = 'Balanced Accuracy'
    dfstd = 'BalAcc CI'
    ### MLL ###
    bal_acc = balanced_accuracy_score(mll[predmll[pred]],
                                      mll['pred_' + predmll[pred]],
                                      adjusted=True)
    if bal_acc < 0:
        bal_acc = 0.0
    df.loc[(case, pred), ('mll', dfmetric)] = bal_acc
    df.loc[(case, pred), ('mll', dfstd)] = conf_int(bal_acc, len(mll))
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        bal_acc = balanced_accuracy_score(alg['TrueY'], alg[A], adjusted=True)
        if bal_acc < 0:
            bal_acc = 0.0
        df.loc[(case, pred), (a, dfmetric)] = bal_acc 
        df.loc[(case, pred), (a, dfstd)] = conf_int(bal_acc, len(alg))

    ### 2. Accuracy ###    
    dfmetric = 'Accuracy'
    dfstd = 'Acc CI'
    ### MLL ###
    acc = accuracy_score(mll[predmll[pred]], mll['pred_' + predmll[pred]])
    df.loc[(case, pred), ('mll', dfmetric)] = acc 
    df.loc[(case, pred), ('mll', dfstd)] = conf_int(acc, len(mll))
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        acc = accuracy_score(alg['TrueY'], alg[A])
        df.loc[(case, pred), (a, dfmetric)] = acc
        df.loc[(case, pred), (a, dfstd)] = conf_int(acc, len(alg))
    
    return df

def reg_sfco(df, case, pred, mll, knn, dtr):
    predmll = {'reactor' : 'ReactorType', 
               'burnup' : 'Burnup', 
               'enrichment' : 'Enrichment', 
               'cooling' : 'CoolingTime'
               }
    llmetric = '_Error'
    errname = 'AbsError'    
    
    ### 1. MAE ###
    dfmetric = 'MAE'
    dfstd = 'MAE Std'
    ### MLL ###
    col = mll[predmll[pred] + llmetric]
    df.loc[(case, pred), ('mll', dfmetric)] = col.mean()
    df.loc[(case, pred), ('mll', dfstd)] = col.std()
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        col = alg[errname]
        df.loc[(case, pred), (a, dfmetric)] = col.mean() 
        df.loc[(case, pred), (a, dfstd)] = col.std()
    
    ### 2. MedAE ###
    dfmetric = 'MedAE'
    dfiqr1 = 'IQR_25'
    dfiqr2 = 'IQR_75'
    q = ['25%', '75%']
    med = '50%'
    ### MLL ###
    col = mll[predmll[pred] + llmetric]
    df.loc[(case, pred), ('mll', dfmetric)] = col.describe()[med]
    df.loc[(case, pred), ('mll', dfiqr1)] = col.describe()[q[0]]
    df.loc[(case, pred), ('mll', dfiqr2)] = col.describe()[q[1]]
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        col = alg[errname]
        df.loc[(case, pred), (a, dfmetric)] = col.describe()[med]
        df.loc[(case, pred), (a, dfiqr1)] = col.describe()[q[0]]
        df.loc[(case, pred), (a, dfiqr2)] = col.describe()[q[1]]
    
    ### 3. MAPE ###
    dfmetric = 'MAPE'
    dfstd = 'MAPE Std'
    ### MLL ###
    mape, std = MAPE(mll[predmll[pred]], mll['pred_' + predmll[pred]])
    df.loc[(case, pred), ('mll', dfmetric)] = mape
    df.loc[(case, pred), ('mll', dfstd)] = std
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        mape, std = MAPE(alg['TrueY'], alg[A])
        df.loc[(case, pred), (a, dfmetric)] = mape
        df.loc[(case, pred), (a, dfstd)] = std
    return df
