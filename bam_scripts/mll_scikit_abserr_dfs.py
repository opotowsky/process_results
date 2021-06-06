#! /usr/bin/env python3

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

def main():
    # for filepaths
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
    algcol = ['knn', 'dtree', 'mll']
    en_dict = {'31' : 'n31/', '_auto' : 'auto/', '113' : 'n113/'}
    
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
    
    for i, p in enumerate(pred):
        for en_list in ['31', '_auto', '113']:
            levels = [algcol, dets]
            df = pd.DataFrame(columns=pd.MultiIndex.from_product(levels, names=['Algorithm', 'Detector']))
            for j, d in enumerate(dets):
                ######################
                ### Scikit Results ###
                ######################
                learn_path = learn_gam
                en_dir = en_dict[en_list]
                # scikit for 113 and 31 lists will both have knn nuc29 & act32 
                # as their starting points then act4, act9 get applied to 
                # n31, n113 respectively
                # P.S. applying act9 to auto, although this isn't accurate
                if 'act' in d:
                    if d == 'act32':
                        en_dir = 'act32/'
                    else:
                        if en_list == '31':
                            d = 'act4'
                            en_dir = 'act4/'
                        else:
                            d = 'act9'
                            en_dir = 'act9/'
                if d == 'nuc29':
                    learn_path = learn_nuc
                    en_dir = 'nuc29/'
                knncsv = p + '_knn' + tfrac + d + csv_end
                dtrcsv = p + '_dtree' + tfrac + d + csv_end
                knn = pd.read_csv(learn_path + en_dir + knncsv).drop(columns='Unnamed: 0')
                dtr = pd.read_csv(learn_path + en_dir + dtrcsv).drop(columns='Unnamed: 0')
                #######################
                ### Make DataFrames ###
                #######################
                predmll = ['ReactorType', 'Burnup', 'Enrichment', 'CoolingTime']
                errname = 'AbsError'
                if p == 'reactor':
                    llmetric = '_Score'
                else:
                    llmetric = '_Error'
                # oh life is jusy messy
                if d == 'act4' or d == 'act9':
                    d = 'act4/9'
                df.loc[:, (algcol[0], d)] = knn[errname]
                df.loc[:, (algcol[1], d)] = dtr[errname]
                df.loc[:, (algcol[2], d)] = mll[en_list][j][predmll[i] + llmetric]
            saveme = 'processed_results/' + p + '_'  + en_dir[:-1] + '_abserr_mll_scikit_compare.pkl'
            df.to_pickle(rdrive + saveme, protocol=4)
            print('{} pred, {} en-list df complete'.format(p, en_list), flush=True)

    return

if __name__ == "__main__":
    main()

