#! /usr/bin/env python3

import pickle
import pandas as pd
from tools import rxtr_sfco, reg_sfco

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_path = rdrive + 'mll/nuc_conc/learncurve/err05/'
    learn_path = rdrive + 'scikit/nuc_conc/learncurve/err05/'
    tsizes = [20, 40, 60, 80, 100]
    #mll100 = rdrive + 'mll/nuc_conc/nuc29/Job0_unc0.01/Job0_unc0.01.csv'
    #learn100 = rdrive + 'scikit/nuc_conc/cv_pred/cv5/nuc29/'
    mll100 = rdrive + 'mll/nuc_conc/nuc29/Job1_unc0.05/Job1_unc0.05.csv'
    learn100 = rdrive + 'scikit/nuc_conc/rand_err/'
    # create empty dataframe
    preds = ['reactor', 'burnup', 'cooling', 'enrichment']
    algcol = ['knn', 'dtree', 'mll']
    levels = [algcol, []]
    df = pd.DataFrame(index=pd.MultiIndex.from_product([tsizes, preds], names=['PredParam', 'RxtrType']), 
                      columns=pd.MultiIndex.from_product(levels, names=['Algorithm', 'Metric']))
    
    for pred in preds:
        for size in tsizes:
            if size == 100:
                #### MLL Results
                mll = pd.read_csv(mll100).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
                ### Scikit Results
                #csv_end = '_tset1.0_nuc29_predictions.csv'
                csv_end = '_tset1.0_nuc29_err5_random_error.csv'
                knncsv = pred + '_knn' + csv_end
                dtrcsv = pred + '_dtree' + csv_end
                knn = pd.read_csv(learn100 + knncsv).drop(columns='Unnamed: 0')
                dtr = pd.read_csv(learn100 + dtrcsv).drop(columns='Unnamed: 0')
            else:
                #### MLL Results
                mll = pd.read_csv(mll_path + str(size) + '/' + str(size) + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
                ### Scikit Results
                csv_end = '_tset1.0_nuc29_' + str(size) + '_predictions.csv'
                knncsv = pred + '_knn' + csv_end
                dtrcsv = pred + '_dtree' + csv_end
                knn = pd.read_csv(learn_path + knncsv).drop(columns='Unnamed: 0')
                dtr = pd.read_csv(learn_path + dtrcsv).drop(columns='Unnamed: 0')
            ### Error Calcs
            if pred == 'reactor':
                df = rxtr_sfco(df, size, pred, mll, knn, dtr)
            else:
                df = reg_sfco(df, size, pred, mll, knn, dtr)
    
    #pklname = 'learncurve_err01_mll_scikit_compare.pkl'
    pklname = 'learncurve_err05_mll_scikit_compare.pkl'
    df.to_pickle(rdrive + 'processed_results/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

