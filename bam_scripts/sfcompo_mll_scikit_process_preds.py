#! /usr/bin/env python3

import pickle
import pandas as pd
from tools import reg_sfco, rxtr_sfco

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_path = rdrive + 'mll/nuc_conc/sfco/'
    learn_path = rdrive + 'scikit/nuc_conc/sfco/'
    jobs = ['Job0_unc0.01_impnull', 'Job1_unc0.01_0null']
    # create empty dataframe
    cases = ['imputed-null', 'zero-null']
    c = ['impnull', '0null']
    preds = ['reactor', 'burnup', 'enrichment']
    algcol = ['knn', 'dtree', 'mll']
    levels = [algcol, []]
    df = pd.DataFrame(index=pd.MultiIndex.from_product([cases, preds], names=['NullHandling', 'PredParam']), 
                      columns=pd.MultiIndex.from_product(levels, names=['Algorithm', 'Metric']))
    
    for i, case in enumerate(cases):
        for pred in preds:
            #### MLL Results
            mll = pd.read_csv(mll_path + jobs[i] + '/' + jobs[i] + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
            ### Scikit Results
            csv_end = '_tset1.0_nuc29_' + c[i] + '_ext_test_compare.csv'
            knncsv = pred + '_knn' + csv_end
            dtrcsv = pred + '_dtree' + csv_end
            knn = pd.read_csv(learn_path + knncsv).drop(columns='Unnamed: 0')
            dtr = pd.read_csv(learn_path + dtrcsv).drop(columns='Unnamed: 0')
            ### Error Calcs
            if pred == 'reactor':
                df = rxtr_sfco(df, case, pred, mll, knn, dtr)
            else:
                df = reg_sfco(df, case, pred, mll, knn, dtr)
    
    pklname = 'sfcompo_mll_scikit_compare.pkl'
    df.to_pickle(rdrive + 'processed_results/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

