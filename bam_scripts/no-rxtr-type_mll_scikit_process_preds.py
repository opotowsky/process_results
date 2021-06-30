#! /usr/bin/env python3

import pickle
import pandas as pd
from tools import reg_rxtr_type

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    learn_path = rdrive + 'scikit/nuc_conc/cv_pred/cv5/nuc29/'
    
    # get indices of rxtr types:
    train = pd.read_pickle(rdrive + 'sim_grams_nuc29.pkl')
    pwr = train.loc[train['ReactorType'] == 'pwr'].index.tolist()
    bwr = train.loc[train['ReactorType'] == 'bwr'].index.tolist()
    phwr = train.loc[train['ReactorType'] == 'phwr'].index.tolist()

    #### MLL Results
    mll_path = rdrive + 'mll/nuc_conc/nuc29/Job0_unc0.01/Job0_unc0.01.csv'
    mll = pd.read_csv(mll_path).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    
    # create empty dataframe
    rxtrs = ['pwr', 'bwr', 'phwr']
    preds = ['burnup', 'cooling', 'enrichment']
    algcol = ['knn', 'dtree', 'mll']
    levels = [algcol, []]
    df = pd.DataFrame(index=pd.MultiIndex.from_product([preds, rxtrs], names=['PredParam', 'RxtrType']), 
                      columns=pd.MultiIndex.from_product(levels, names=['Algorithm', 'Metric']))
    
    idx = {'pwr' : pwr, 'bwr' : bwr, 'phwr' : phwr}
    for pred in preds:
        ### Scikit Results
        csv_end = '_tset1.0_nuc29_predictions.csv'
        knncsv = pred + '_knn' + csv_end
        dtrcsv = pred + '_dtree' + csv_end
        knn = pd.read_csv(learn_path + knncsv).drop(columns='Unnamed: 0')
        dtr = pd.read_csv(learn_path + dtrcsv).drop(columns='Unnamed: 0')
        for rxtr in rxtrs:
            # filter indices of rxtr type
            mll_sub = mll.loc[idx[rxtr]]
            knn_sub = knn.loc[idx[rxtr]]
            dtr_sub = dtr.loc[idx[rxtr]]
            ### Error Calcs
            df = reg_rxtr_type(df, pred, rxtr, mll_sub, knn_sub, dtr_sub)
    
    pklname = 'no-rxtr-type_mll_scikit_compare.pkl'
    df.to_pickle(rdrive + 'processed_results/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

