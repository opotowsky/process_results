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
    for en_list in ['31', '_auto', '113']:
        ### MLL ###
        df.loc[d, ('mll'+en_list, dfmetric)] = balanced_accuracy_score(mll[en_list][predmll[i]], 
                                                                       mll[en_list]['pred_' + predmll[i]], 
                                                                       adjusted=True)
        df.loc[d, ('mll'+en_list, dfstd)] = mll[en_list][predmll[i] + llmetric].std()
        ### Scikit ###
        for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
            df.loc[d, (a+en_list, dfmetric)] = balanced_accuracy_score(alg[en_list]['TrueY'], 
                                                                       alg[en_list][A], 
                                                                       adjusted=True)
            df.loc[d, (a+en_list, dfstd)] = alg[en_list][errname].std()
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
    for en_list in ['31', '_auto', '113']:
        if mean_or_med == 'mean':
            ### MLL ###
            df.loc[d, ('mll'+en_list, dfmetric)] = -mean_absolute_error(mll[en_list][predmll[i]], 
                                                                        mll[en_list]['pred_' + predmll[i]])
            df.loc[d, ('mll'+en_list, dfstd)] = mll[en_list][predmll[i] + llmetric].std()
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[d, (a+en_list, dfmetric)] = -mean_absolute_error(alg[en_list]['TrueY'], 
                                                                        alg[en_list][A])
                df.loc[d, (a+en_list, dfstd)] = alg[en_list][errname].std()
        else:
            ### MLL ###
            q = ['25%', '75%']
            med = '50%'
            df.loc[d, ('mll'+en_list, dfmetric1)] = -median_absolute_error(mll[en_list][predmll[i]], 
                                                                          mll[en_list]['pred_' + predmll[i]])
            col = mll[en_list][predmll[i] + llmetric]
            df.loc[d, ('mll'+en_list, dfmetric2)] = -col.describe()[med]
            df.loc[d, ('mll'+en_list, dfiqr1)] = -col.describe()[q[0]]
            df.loc[d, ('mll'+en_list, dfiqr2)] = -col.describe()[q[1]]
            ### Scikit ###
            for a, A, alg in zip(['knn', 'dtree'], ['kNN', 'DTree'], [knn, dtr]):
                df.loc[d, (a+en_list, dfmetric1)] = -median_absolute_error(alg[en_list]['TrueY'], 
                                                                          alg[en_list][A])
                col = alg[en_list][errname]
                df.loc[d, (a+en_list, dfmetric2)] = -col.describe()[med]
                df.loc[d, (a+en_list, dfiqr1)] = -col.describe()[q[0]]
                df.loc[d, (a+en_list, dfiqr2)] = -col.describe()[q[1]]
    return df

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_gam = rdrive + 'mll/gam_spec/'
    mll_nuc = rdrive + 'mll/nuc_conc/'
    learn_gam = rdrive + 'scikit/gam_spec/test_0.067_only/'
    learn_nuc = rdrive + 'scikit/nuc_conc/test_0.067_only/'
    tfrac = '_tset1.0_'
    csv_end = '_mimic_mll.csv'
    # for loops
    dets = ['nuc29', 'act32', 'act4/9', 'd1_hpge', 'd2_hpge', 'd3_czt', 'd6_sri2', 'd5_labr3', 'd4_nai']
    pred = ['reactor', 'burnup', 'enrichment', 'cooling']
    # for dataframes
    algcol = ['knn_auto', 'dtree_auto', 'mll_auto', 'knn31', 'dtree31', 'mll31', 'knn113', 'dtree113', 'mll113']
    scrcol  = ['Accuracy', 'Acc Std']
    errcol = ['Neg MAE', 'MAE Std']
    
    #####################
    #### MLL Results ####
    #####################
    job_act = 'Job0_unc0.05'
    job = 'Job1_unc0.0'
    # nuclide masses
    n29mll = pd.read_csv(mll_nuc + 'train/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    # activities
    a32mll = pd.read_csv(mll_gam + 'act32/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    a9mll = pd.read_csv(mll_gam + 'act9/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    a4mll = pd.read_csv(mll_gam + 'act4/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    mll31 = [n29mll, a32mll, a4mll]
    mll_auto = [n29mll, a32mll, a9mll]
    mll113 = [n29mll, a32mll, a9mll]
    for d in ['d1', 'd2', 'd3', 'd6', 'd5', 'd4']:
        mll_auto.append(pd.read_csv(mll_gam + d + '_auto/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
        mll31.append(pd.read_csv(mll_gam + d + '_n31/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
        mll113.append(pd.read_csv(mll_gam + d + '_n113/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
    mll = {'31' : mll31, '_auto' : mll_auto, '113' : mll113}
    
    results = {}
    for i, p in enumerate(pred):
        if p == 'reactor':
            levels = [algcol, scrcol]
        else:
            levels = [algcol, errcol]
        df = pd.DataFrame(index=dets, columns=pd.MultiIndex.from_product(levels, names=["Algorithm", "Metric"]))
        for j, d in enumerate(dets):
            ######################
            ### Scikit Results ###
            ######################
            learn_path = learn_gam
            knncsv = p + '_knn' + tfrac + d + csv_end
            dtrcsv = p + '_dtree' + tfrac + d + csv_end
            if 'nuc' in d or 'act' in d:
                # scikit for 113 and 31 lists will both have knn nuc29 & act32 
                # as their starting points then act4, act9 get applied to 
                # n31, n113 respectively
                # P.S. applying act9 to auto, although this isn't accurate
                if d == 'act4/9':
                    knncsv = p + '_knn' + tfrac + 'act4' + csv_end
                    dtrcsv = p + '_dtree' + tfrac + 'act4' + csv_end
                    knn31 = pd.read_csv(learn_path + 'act4/' + knncsv).drop(columns='Unnamed: 0')
                    dtr31 = pd.read_csv(learn_path + 'act4/' + dtrcsv).drop(columns='Unnamed: 0')
                    knncsv = p + '_knn' + tfrac + 'act9' + csv_end
                    dtrcsv = p + '_dtree' + tfrac + 'act9' + csv_end
                    ##### applying act9 to auto, although this isn't accurate #####
                    knn_auto = knn113 = pd.read_csv(learn_path + 'act9/' + knncsv).drop(columns='Unnamed: 0')
                    dtr_auto = dtr113 = pd.read_csv(learn_path + 'act9/' + dtrcsv).drop(columns='Unnamed: 0')
                else:
                    if 'nuc' in d:
                        learn_path = learn_nuc
                    knn_auto = knn31 = knn113 = pd.read_csv(learn_path + d + '/' + knncsv).drop(columns='Unnamed: 0')
                    dtr_auto = dtr31 = dtr113 = pd.read_csv(learn_path + d + '/' + dtrcsv).drop(columns='Unnamed: 0')
            else:
                knn_auto = pd.read_csv(learn_path + 'auto/' + knncsv).drop(columns='Unnamed: 0')
                dtr_auto = pd.read_csv(learn_path + 'auto/' + dtrcsv).drop(columns='Unnamed: 0')
                knn31 = pd.read_csv(learn_path + 'n31/' + knncsv).drop(columns='Unnamed: 0')
                dtr31 = pd.read_csv(learn_path + 'n31/' + dtrcsv).drop(columns='Unnamed: 0')
                knn113 = pd.read_csv(learn_path + 'n113/' + knncsv).drop(columns='Unnamed: 0')
                dtr113 = pd.read_csv(learn_path + 'n113/' + dtrcsv).drop(columns='Unnamed: 0')
            ###################
            ### Error Calcs ###
            ###################
            mll_d = {'31' : mll31[j], '_auto' : mll_auto[j], '113' : mll113[j]}
            knn = {'31' : knn31, '_auto' : knn_auto, '113' : knn113}
            dtr = {'31' : dtr31, '_auto' : dtr_auto, '113' : dtr113}
            if p == 'reactor':
                df = rxtr_metrics(i, df, d, mll_d, knn, dtr)
            else:
                df = reg_metrics(i, df, d, mll_d, knn, dtr, 'med')
        results[p] = df
        print('{} pred df complete'.format(p), flush=True)
    
    with open(rdrive + 'processed_results/median_err_metrics_results_dict_mll_scikit_compare.pkl', 'wb') as pkl:
        pickle.dump(results, pkl)

    return

if __name__ == "__main__":
    main()

