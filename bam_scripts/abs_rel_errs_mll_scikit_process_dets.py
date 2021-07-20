#! /usr/bin/env python3

import sys
import pickle
import argparse
import numpy as np
import pandas as pd

def ape(y_true, y_pred):
    ape = np.abs((y_true - y_pred)/y_true)*100
    return ape

def main():
    parser = argparse.ArgumentParser(description='Processes mll and scikit absolute and relative errors into .pkl files.')
    parser.add_argument('pred', metavar='prediction', choices = ['burnup', 'enrichment', 'cooling'],
                        help='string indicating which prediction error to process')
    parser.add_argument('enlist', metavar='energy-list', choices = ['auto', 'short', 'long'],
                        help='string indicating which energy list to process')
    args = parser.parse_args(sys.argv[1:])
    
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
    pred = args.pred
    enlist = args.enlist
    # make empty dataframe
    ens = {'auto' : '_auto', 'short' : '_short', 'long' : '_long'}
    algcol = ['knn' + ens[enlist], 'dtree' + ens[enlist], 'mll' + ens[enlist]]
    levels = [dets, algcol]
    df_abs = pd.DataFrame(columns=pd.MultiIndex.from_product(levels, names=['Detector', 'Algorithm']))
    df_rel = pd.DataFrame(columns=pd.MultiIndex.from_product(levels, names=['Detector', 'Algorithm']))

    
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
    if enlist == 'short':
        a7mll = pd.read_csv(mll_gam + 'act7/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        mll = [n29mll, a32mll, a7mll]
    else:
        a12mll = pd.read_csv(mll_gam + 'act12/' + job_act + '/' + job_act + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        mll = [n29mll, a32mll, a12mll]
    for d in ['d1', 'd2', 'd3', 'd6', 'd5', 'd4']:
        mll.append(pd.read_csv(mll_gam + d + ens[enlist]  + '/' + job + '/' + job + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1']))
    for i, d in enumerate(dets):
        ######################
        ### Scikit Results ###
        ######################
        learn_path = learn_gam
        knncsv = pred + '_knn' + tfrac + d + csv_end
        dtrcsv = pred + '_dtree' + tfrac + d + csv_end
        if 'nuc' in d or 'act' in d:
            # results for all lists will both have nuc29 & act32 as their
            # starting points then act7, act12 get applied to short, long
            # respectively
            if d == 'act7/12':
                if enlist == 'short':
                    knncsv = pred + '_knn' + tfrac + 'act7' + csv_end
                    dtrcsv = pred + '_dtree' + tfrac + 'act7' + csv_end
                    knn = pd.read_csv(learn_path + 'act7/' + knncsv).drop(columns='Unnamed: 0')
                    dtr = pd.read_csv(learn_path + 'act7/' + dtrcsv).drop(columns='Unnamed: 0')
                else:
                    knncsv = pred + '_knn' + tfrac + 'act12' + csv_end
                    dtrcsv = pred + '_dtree' + tfrac + 'act12' + csv_end
                    ##### applying act12 to auto, although this isn't accurate #####
                    knn = pd.read_csv(learn_path + 'act12/' + knncsv).drop(columns='Unnamed: 0')
                    dtr = pd.read_csv(learn_path + 'act12/' + dtrcsv).drop(columns='Unnamed: 0')
            else:
                if 'nuc' in d:
                    learn_path = learn_nuc
                knn = pd.read_csv(learn_path + d + '/' + knncsv).drop(columns='Unnamed: 0')
                dtr = pd.read_csv(learn_path + d + '/' + dtrcsv).drop(columns='Unnamed: 0')
        else:
            knn = pd.read_csv(learn_path + enlist + '/' + knncsv).drop(columns='Unnamed: 0')
            dtr = pd.read_csv(learn_path + enlist + '/' + dtrcsv).drop(columns='Unnamed: 0')
        ##############
        ### Errors ###
        ##############
        predmll = {'burnup' : 'Burnup', 
                   'enrichment' : 'Enrichment', 
                   'cooling' : 'CoolingTime'
                   }
        llpred = 'pred_'
        llerr = '_Error'
        truename = 'TrueY'
        errname = 'AbsError'
        
        df_abs.loc[:, (d, 'knn' + ens[enlist])] = knn[errname]
        df_abs.loc[:, (d, 'dtree' + ens[enlist])] = dtr[errname]
        df_abs.loc[:, (d, 'mll' + ens[enlist])] = mll[i][predmll[pred]+llerr]

        df_rel.loc[:, (d, 'knn' + ens[enlist])] = ape(knn[truename], knn['kNN'])
        df_rel.loc[:, (d, 'dtree' + ens[enlist])] = ape(dtr[truename], dtr['DTree'])
        df_rel.loc[:, (d, 'mll' + ens[enlist])] = ape(mll[i][predmll[pred]], mll[i][llpred+predmll[pred]])

    print('{} {} pred df complete'.format(pred, enlist), flush=True)
    
    pklname = args.pred + '_abserr_mll_scikit_compare_' + enlist  + '.pkl'
    df_abs.to_pickle(rdrive + 'processed_results/abserr/' + pklname, protocol=4)
    pklname = args.pred + '_relerr_mll_scikit_compare_' + enlist  + '.pkl'
    df_rel.to_pickle(rdrive + 'processed_results/relerr/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

