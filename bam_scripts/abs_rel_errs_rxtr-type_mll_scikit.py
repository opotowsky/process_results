#! /usr/bin/env python3

import pickle
import numpy as np
import pandas as pd

def ape(y_true, y_pred):
    ape = np.abs((y_true - y_pred)/y_true)*100
    return ape

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_path = rdrive + 'mll/nuc_conc/rxtr-type/err05/'
    learn_path = rdrive + 'scikit/nuc_conc/rxtr-type/err05/'
    rxtrs = ['bwr', 'pwr', 'phwr']
    # create empty dataframe
    preds = ['burnup', 'cooling', 'enrichment']
    algcol = ['knn', 'dtree', 'mll']
    levels = [algcol, rxtrs]
    df_abs = pd.DataFrame(columns=pd.MultiIndex.from_product(levels, names=['Algorithm', 'Rxtr Type']))
    df_rel = pd.DataFrame(columns=pd.MultiIndex.from_product(levels, names=['Algorithm', 'Rxtr Type']))
    
    for pred in preds:
        for rxtr in rxtrs:
            #### MLL Results
            mll = pd.read_csv(mll_path + rxtr + '/' + rxtr + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
            ### Scikit Results
            csv_end = '_tset1.0_nuc29_' + rxtr + '_predictions.csv'
            knncsv = pred + '_knn' + csv_end
            dtrcsv = pred + '_dtree' + csv_end
            knn = pd.read_csv(learn_path + knncsv).drop(columns='Unnamed: 0')
            dtr = pd.read_csv(learn_path + dtrcsv).drop(columns='Unnamed: 0')
    
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
            
            df_abs.loc[:, ('knn', rxtr)] = knn[errname]
            df_abs.loc[:, ('dtree', rxtr)] = dtr[errname]
            df_abs.loc[:, ('mll', rxtr)] = mll[predmll[pred]+llerr]

            df_rel.loc[:, ('knn', rxtr)] = ape(knn[truename], knn['kNN'])
            df_rel.loc[:, ('dtree', rxtr)] = ape(dtr[truename], dtr['DTree'])
            df_rel.loc[:, ('mll', rxtr)] = ape(mll[predmll[pred]], mll[llpred+predmll[pred]])

        print('{} pred df complete'.format(pred), flush=True)
        
        pklname = pred + '_rxtr-type_abserr_mll_scikit_compare.pkl'
        df_abs.to_pickle(rdrive + 'processed_results/abserr/' + pklname, protocol=4)
        pklname = pred + '_rxtr-type_relerr_mll_scikit_compare.pkl'
        df_rel.to_pickle(rdrive + 'processed_results/relerr/' + pklname, protocol=4)
    
    return

if __name__ == "__main__":
    main()

