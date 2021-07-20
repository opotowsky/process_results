#! /usr/bin/env python3

import numpy as np
import pandas as pd

def ape(y_true, y_pred):
    ape = np.abs((y_true - y_pred)/y_true)*100
    return ape

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_csv = rdrive + 'mll/nuc_conc/nuc29/Job1_unc0.05/Job1_unc0.05.csv'
    mll = pd.read_csv(mll_csv).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    learn_path = rdrive + 'scikit/nuc_conc/cv_pred/cv5/nuc29/'

    # create empty dataframe
    preds = ['burnup', 'cooling', 'enrichment']
    algcol = ['knn', 'dtree', 'mll']
    df_abs = pd.DataFrame(columns=algcol)
    df_rel = pd.DataFrame(columns=algcol)
    
    for pred in preds:
        csv_end = '_tset1.0_nuc29_predictions.csv'
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
        
        df_abs['knn'] = knn[errname]
        df_abs['dtree'] = dtr[errname]
        df_abs['mll'] = mll[predmll[pred]+llerr]

        df_rel['knn'] = ape(knn[truename], knn['kNN'])
        df_rel['dtree'] = ape(dtr[truename], dtr['DTree'])
        df_rel['mll'] = ape(mll[predmll[pred]], mll[llpred+predmll[pred]])

        print('{} pred df complete'.format(pred), flush=True)
        
        pklname = pred + '_err05_abserr_mll_scikit_compare.pkl'
        df_abs.to_pickle(rdrive + 'processed_results/abserr/' + pklname, protocol=4)
        pklname = pred + '_err05_relerr_mll_scikit_compare.pkl'
        df_rel.to_pickle(rdrive + 'processed_results/relerr/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

