#! /usr/bin/env python3

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error

def rxtr_metrics(df, d, knn, dtr):
    dfmetric = 'Accuracy'
    dfstd = 'Acc Std'
    errname = 'AbsError'    
    for en_list in ['_short', '_auto', '_long']:
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a+en_list, dfmetric)] = balanced_accuracy_score(alg[en_list]['TrueY'], alg[en_list][A], adjusted=True)
            df.loc[d, (a+en_list, dfstd)] = alg[en_list][errname].std()
    return df

def reg_metrics(df, d, knn, dtr):
    dfmetric = 'Neg MAE'
    dfstd = 'MAE Std'
    errname = 'AbsError'    
    for en_list in ['_short', '_auto', '_long']:
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a+en_list, dfmetric)] = -mean_absolute_error(alg[en_list]['TrueY'], alg[en_list][A])
            df.loc[d, (a+en_list, dfstd)] = alg[en_list][errname].std()
    return df

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    learn_gam = rdrive + 'scikit/gam_spec/cv_pred/cv5/'
    learn_nuc = rdrive + 'scikit/nuc_conc/cv_pred/cv5/'
    cv = 5
    tfrac = '_tset1.0_'
    csv_end = '_predictions.csv'
    # for loops
    dets = ['nuc29', 'act32', 'act7/12', 'd1_hpge', 'd2_hpge', 'd3_czt', 'd6_sri2', 'd5_labr3', 'd4_nai']
    pred = ['reactor', 'burnup', 'enrichment', 'cooling']
    # for dataframes
    algcol = ['knn_auto', 'dtree_auto', 'knn_short', 'dtree_short', 'knn_long', 'dtree_long']
    scrcol  = ['Accuracy', 'Acc Std']
    errcol = ['Neg MAE', 'MAE Std']
    results = {}
    
    for p in pred:
        if p == 'reactor':
            levels = [algcol, scrcol]
            dfmetric = 'Accuracy'
            dfstd = 'Acc Std'
        else: 
            levels = [algcol, errcol]
            dfmetric = 'Neg MAE'
            dfstd = 'MAE Std'
        df = pd.DataFrame(index=dets, columns=pd.MultiIndex.from_product(levels, names=["Algorithm", "Metric"]))
        for d in dets:
            learn_path = learn_gam
            knncsv = p + '_knn' + tfrac + 'cv' + str(cv) + '_' + d + csv_end
            dtrcsv = p + '_dtree' + tfrac + 'cv' + str(cv) + '_' + d + csv_end
            if 'nuc' in d or 'act' in d:
                if d == 'act7/12':
                    knncsv = p + '_knn' + tfrac + 'cv' + str(cv) + '_' + 'act7' + csv_end
                    dtrcsv = p + '_dtree' + tfrac + 'cv' + str(cv) + '_' + 'act7' + csv_end
                    knn_short = pd.read_csv(learn_path + 'act7/' + knncsv).drop(columns='Unnamed: 0')
                    dtr_short = pd.read_csv(learn_path + 'act7/' + dtrcsv).drop(columns='Unnamed: 0')
                    knncsv = p + '_knn' + tfrac + 'cv' + str(cv) + '_' + 'act12' + csv_end
                    dtrcsv = p + '_dtree' + tfrac + 'cv' + str(cv) + '_' + 'act12' + csv_end
                    ##### applying act12 to auto, although this isn't accurate #####
                    knn_auto = knn_long = pd.read_csv(learn_path + 'act12/' + knncsv).drop(columns='Unnamed: 0')
                    dtr_auto = dtr_long = pd.read_csv(learn_path + 'act12/' + dtrcsv).drop(columns='Unnamed: 0')
                else:
                    if 'nuc' in d:
                        learn_path = learn_nuc
                    knn_auto = knn_short = knn_long = pd.read_csv(learn_path + d + '/' + knncsv).drop(columns='Unnamed: 0')
                    dtr_auto = dtr_short = dtr_long = pd.read_csv(learn_path + d + '/' + dtrcsv).drop(columns='Unnamed: 0')
            else:
                knn_auto = pd.read_csv(learn_path + 'auto/' + knncsv).drop(columns='Unnamed: 0')
                dtr_auto = pd.read_csv(learn_path + 'auto/' + dtrcsv).drop(columns='Unnamed: 0')
                knn_short = pd.read_csv(learn_path + 'short/' + knncsv).drop(columns='Unnamed: 0')
                dtr_short = pd.read_csv(learn_path + 'short/' + dtrcsv).drop(columns='Unnamed: 0')
                knn_long = pd.read_csv(learn_path + 'long/' + knncsv).drop(columns='Unnamed: 0')
                dtr_long = pd.read_csv(learn_path + 'long/' + dtrcsv).drop(columns='Unnamed: 0')
            # Error Calcs
            knn = {'short' : knn_short, '_auto' : knn_auto, 'short' : knn_long}
            dtr = {'short' : dtr_short, '_auto' : dtr_auto, 'short' : dtr_long}
            if p == 'reactor':        
                df = rxtr_metrics(df, d, knn, dtr)
            else:
                df = reg_metrics(df, d, knn, dtr)
        results[cv][p] = df
        print('CV {}, {} pred df complete'.format(cv, p), flush=True)
    
    with open(rdrive + 'processed_results/cv_pred.pkl', 'wb') as pkl:
        pickle.dump(results, pkl)

    return

if __name__ == "__main__":
    main()

