#! /usr/bin/env python3

import pickle
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, median_absolute_error

def rxtr_metrics(i, df, d, mll, knn, dtr):
    predmll = ['ReactorType', 'Burnup', 'Enrichment', 'CoolingTime']
    llmetric = '_Score'
    dfmetric = 'Accuracy'
    dfstd = 'Acc Std'
    errname = 'AbsError'    
    ### MLL ###
    df.loc[d, ('mll', dfmetric)] = balanced_accuracy_score(mll[predmll[i]], 
                                                           mll['pred_' + predmll[i]], 
                                                           adjusted=True)
    df.loc[d, ('mll', dfstd)] = mll[predmll[i] + llmetric].std()
    ### Scikit ###
    for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
        df.loc[d, (a, dfmetric)] = balanced_accuracy_score(alg['TrueY'],
                                                           alg[A], 
                                                           adjusted=True)
        df.loc[d, (a, dfstd)] = alg[errname].std()
    return df

def reg_metrics(i, df, d, mll, knn, dtr, mean_or_med):
    predmll = ['ReactorType', 'Burnup', 'Enrichment', 'CoolingTime']
    llmetric = '_Error'
    dfmetric = 'Neg MAE'
    dfstd = 'MAE Std'
    errname = 'AbsError'    
    dfmetric1 = 'Neg MedAE SK'
    dfmetric2 = 'Neg MedAE'
    dfiqr1 = 'MedAE IQR_25'
    dfiqr2 = 'MedAE IQR_75'
    if mean_or_med == 'mean':
        ### MLL ###
        df.loc[d, ('mll', dfmetric)] = -mean_absolute_error(mll[predmll[i]], mll['pred_' + predmll[i]])
        df.loc[d, ('mll', dfstd)] = mll[predmll[i] + llmetric].std()
        ### Scikit ###
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a, dfmetric)] = -mean_absolute_error(alg['TrueY'], alg[A])
            df.loc[d, (a, dfstd)] = alg[errname].std()
    else:
        q = ['25%', '75%']
        med = '50%'
        ### MLL ###
        df.loc[d, ('mll', dfmetric1)] = -median_absolute_error(mll[predmll[i]], mll['pred_' + predmll[i]])
        col = mll[predmll[i] + llmetric]
        df.loc[d, ('mll', dfmetric2)] = -col.describe()[med]
        df.loc[d, ('mll', dfiqr1)] = -col.describe()[q[0]]
        df.loc[d, ('mll', dfiqr2)] = -col.describe()[q[1]]
        ### Scikit ###
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a, dfmetric1)] = -median_absolute_error(alg['TrueY'], alg[A])
            col = alg[errname]
            df.loc[d, (a, dfmetric2)] = -col.describe()[med]
            df.loc[d, (a, dfiqr1)] = -col.describe()[q[0]]
            df.loc[d, (a, dfiqr2)] = -col.describe()[q[1]]
    return df

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_path = rdrive + 'mll/nuc_conc/train/'
    learn_path = rdrive + 'scikit/nuc_conc/rand_err/'
    csv_end = '_tset1.0_nuc29_random_error.csv'
    pred = ['reactor', 'burnup', 'enrichment', 'cooling']
    # for dataframes
    algcol = ['knn', 'dtree', 'mll']
    scrcol  = ['Accuracy', 'Acc Std']
    errcol = ['Neg MAE', 'MAE Std']
    
    #####################
    #### MLL Results ####
    #####################
    job_act = 'Job0_unc0.05'
    mll = pd.read_csv(mll_path + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    results = {}
    for i, p in enumerate(pred):
        if p == 'reactor':
            levels = [algcol, scrcol]
        else:
            levels = [algcol, errcol]
        df = pd.DataFrame(index=, columns=pd.MultiIndex.from_product(levels, names=["Algorithm", "Metric"]))
        ### Scikit Results
        knncsv = p + '_knn' + csv_end
        dtrcsv = p + '_dtree' + csv_end
        knn = pd.read_csv(learn_path + knncsv).drop(columns='Unnamed: 0')
        dtr = pd.read_csv(learn_path + dtrcsv).drop(columns='Unnamed: 0')
        ### Error Calcs
        if p == 'reactor':
            df = rxtr_metrics(i, df, d, mll, knn, dtr)
        else:
            df = reg_metrics(i, df, d, mll, knn, dtr, 'mean')
        results[p] = df
        print('{} pred df complete'.format(p), flush=True)
    
    #with open(rdrive + 'processed_results/median_err_randerr_metrics_results_dict_mll_scikit_compare.pkl', 'wb') as pkl:
    with open(rdrive + 'processed_results/randerr_metrics_results_dict_mll_scikit_compare.pkl', 'wb') as pkl:
        pickle.dump(results, pkl)

    return

if __name__ == "__main__":
    main()

