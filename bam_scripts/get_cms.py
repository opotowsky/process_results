#! /usr/bin/env python3

import sys
import pickle
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():
    
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_gam = rdrive + 'mll/gam_spec/'
    mll_nuc = rdrive + 'mll/nuc_conc/'
    learn_gam = rdrive + 'scikit/gam_spec/cv_pred/cv5/'
    learn_nuc = rdrive + 'scikit/nuc_conc/cv_pred/cv5/'
    tfrac = '_tset1.0_'
    csv_end = '_predictions.csv'
    # for loops
    dets = ['nuc29', 'act32', 'act7/12', 'd1_hpge', 'd2_hpge', 'd3_czt', 'd6_sri2', 'd5_labr3', 'd4_nai']
    cm = {}
    
    #####################
    #### MLL Results ####
    #####################
    job_nuc = 'Job0_unc0.01'
    job_act = 'Job2_unc0.01'
    job = 'Job2_unc0.0'
    # nuclide masses
    n29mll = pd.read_csv(mll_nuc + 'nuc29/' + job_nuc + '/' + job_nuc + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    # activities
    a32mll = pd.read_csv(mll_gam + 'act32/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    a12mll = pd.read_csv(mll_gam + 'act12/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    a7mll = pd.read_csv(mll_gam + 'act7/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    mll_short = [n29mll, a32mll, a7mll]
    mll_auto = [n29mll, a32mll, a12mll]
    mll_long = [n29mll, a32mll, a12mll]
    for d in ['d1', 'd2', 'd3', 'd6', 'd5', 'd4']:
        mll_auto.append(pd.read_csv(mll_gam + d + '_auto/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
        mll_short.append(pd.read_csv(mll_gam + d + '_short/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
        mll_long.append(pd.read_csv(mll_gam + d + '_long/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
    mll = {'short' : mll_short, 'auto' : mll_auto, 'long' : mll_long}
    for i, d in enumerate(dets):
        ######################
        ### Scikit Results ###
        ######################
        pred = 'reactor'
        learn_path = learn_gam
        knncsv = pred + '_knn' + tfrac + d + csv_end
        dtrcsv = pred + '_dtree' + tfrac + d + csv_end
        if 'nuc' in d or 'act' in d:
            # results for all lists will both have nuc29 & act32 as their
            # starting points then act7, act12 get applied to short, long
            # respectively
            if d == 'act7/12':
                knncsv = pred + '_knn' + tfrac + 'act7' + csv_end
                dtrcsv = pred + '_dtree' + tfrac + 'act7' + csv_end
                knn_short = pd.read_csv(learn_path + 'act7/' + knncsv).drop(columns='Unnamed: 0')
                dtr_short = pd.read_csv(learn_path + 'act7/' + dtrcsv).drop(columns='Unnamed: 0')
                knncsv = pred + '_knn' + tfrac + 'act12' + csv_end
                dtrcsv = pred + '_dtree' + tfrac + 'act12' + csv_end
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
        ##########################
        ### Confusion Matrices ###
        ##########################
        mll_d = {'short' : mll_short[i], 'auto' : mll_auto[i], 'long' : mll_long[i]}
        knn = {'short' : knn_short, 'auto' : knn_auto, 'long' : knn_long}
        dtr = {'short' : dtr_short, 'auto' : dtr_auto, 'long' : dtr_long}
        cm_labels = ['pwr', 'bwr', 'phwr']
        cm_det = {}
        for enlist in ['auto', 'short', 'long']:
            knn_cm = confusion_matrix(knn[enlist]['TrueY'], knn[enlist]['kNN'], labels=cm_labels)
            dtr_cm = confusion_matrix(dtr[enlist]['TrueY'], dtr[enlist]['DTree'], labels=cm_labels)
            mll_cm = confusion_matrix(mll_d[enlist]['ReactorType'], mll_d[enlist]['pred_ReactorType'], labels=cm_labels)
            cm_det[enlist] = [knn_cm, dtr_cm, mll_cm]
        cm[d] = cm_det
    
    pklname = rdrive + 'processed_results/confusion_matrices_all_detectors.pkl'
    with open(pklname, 'wb') as pkl:
        pickle.dump(cm, pkl, protocol=4)

    return

if __name__ == "__main__":
    main()

