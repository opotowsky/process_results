#! /usr/bin/env python3

import pickle
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

def rxtr_metrics(df, d, knn, dtr):
    dfmetric = 'Accuracy'
    dfstd = 'Acc Std'
    errname = 'AbsError'    
    for en_list in ['31', '_auto', '113']:
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a+en_list, dfmetric)] = balanced_accuracy_score(alg[en_list]['TrueY'], alg[en_list][A], adjusted=True)
            df.loc[d, (a+en_list, dfstd)] = alg[en_list][errname].std()
    return df

def reg_metrics(df, d, knn, dtr):
    dfmetric = 'Neg MAE'
    dfstd = 'MAE Std'
    errname = 'AbsError'    
    for en_list in ['31', '_auto', '113']:
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a+en_list, dfmetric)] = -mean_absolute_error(alg[en_list]['TrueY'], alg[en_list][A])
            df.loc[d, (a+en_list, dfstd)] = alg[en_list][errname].std()
    return df

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    learn_gam = rdrive + 'scikit/gam_spec/cv_pred/'
    learn_nuc = rdrive + 'scikit/nuc_conc/cv_pred/'
    tfrac = '_tset1.0_'
    csv_end = '_predictions.csv'
    # for loops
    dets = ['nuc29', 'act32', 'act4/9', 'd1_hpge', 'd2_hpge', 'd3_czt', 'd6_sri2', 'd5_labr3', 'd4_nai']
    pred = ['reactor', 'burnup', 'enrichment', 'cooling']
    # for dataframes
    algcol = ['knn_auto', 'dtree_auto', 'knn31', 'dtree31', 'knn113', 'dtree113']
    scrcol  = ['Accuracy', 'Acc Std']
    errcol = ['Neg MAE', 'MAE Std']
    # loops for results, dicts of dataframes    
    results5 = {}
    results10 = {}
    results15 = {}
    results = {'5' : results5, '10' : results10, '15' : results15}
    
    for cv, cv_dir in zip(['5', '10', '15'], ['cv5/', 'cv10/', 'cv15/']):
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
                learn_path = learn_gam + cv_dir
                knncsv = p + '_knn' + tfrac + 'cv' + str(cv) + '_' + d + csv_end
                dtrcsv = p + '_dtree' + tfrac + 'cv' + str(cv) + '_' + d + csv_end
                if 'nuc' in d or 'act' in d:
                    if d == 'act4/9':
                        knncsv = p + '_knn' + tfrac + 'cv' + str(cv) + '_' + 'act4' + csv_end
                        dtrcsv = p + '_dtree' + tfrac + 'cv' + str(cv) + '_' + 'act4' + csv_end
                        knn31 = pd.read_csv(learn_path + 'act4/' + knncsv).drop(columns='Unnamed: 0')
                        dtr31 = pd.read_csv(learn_path + 'act4/' + dtrcsv).drop(columns='Unnamed: 0')
                        knncsv = p + '_knn' + tfrac + 'cv' + str(cv) + '_' + 'act9' + csv_end
                        dtrcsv = p + '_dtree' + tfrac + 'cv' + str(cv) + '_' + 'act9' + csv_end
                        ##### applying act9 to auto, although this isn't accurate #####
                        knn_auto = knn113 = pd.read_csv(learn_path + 'act9/' + knncsv).drop(columns='Unnamed: 0')
                        dtr_auto = dtr113 = pd.read_csv(learn_path + 'act9/' + dtrcsv).drop(columns='Unnamed: 0')
                    else:
                        if 'nuc' in d:
                            learn_path = learn_nuc + cv_dir
                        knn_auto = knn31 = knn113 = pd.read_csv(learn_path + d + '/' + knncsv).drop(columns='Unnamed: 0')
                        dtr_auto = dtr31 = dtr113 = pd.read_csv(learn_path + d + '/' + dtrcsv).drop(columns='Unnamed: 0')
                else:
                    knn_auto = pd.read_csv(learn_path + 'auto/' + knncsv).drop(columns='Unnamed: 0')
                    dtr_auto = pd.read_csv(learn_path + 'auto/' + dtrcsv).drop(columns='Unnamed: 0')
                    knn31 = pd.read_csv(learn_path + 'n31/' + knncsv).drop(columns='Unnamed: 0')
                    dtr31 = pd.read_csv(learn_path + 'n31/' + dtrcsv).drop(columns='Unnamed: 0')
                    knn113 = pd.read_csv(learn_path + 'n113/' + knncsv).drop(columns='Unnamed: 0')
                    dtr113 = pd.read_csv(learn_path + 'n113/' + dtrcsv).drop(columns='Unnamed: 0')
                # Error Calcs
                knn = {'31' : knn31, '_auto' : knn_auto, '113' : knn113}
                dtr = {'31' : dtr31, '_auto' : dtr_auto, '113' : dtr113}
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

