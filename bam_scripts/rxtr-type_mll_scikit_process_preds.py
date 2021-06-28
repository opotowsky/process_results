#! /usr/bin/env python3

import pickle
import pandas as pd
from tools import reg_rxtr_type

def main():
    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_path = rdrive + 'mll/nuc_conc/rxtr-type/'
    learn_path = rdrive + 'scikit/nuc_conc/rxtr-type/'
    rxtrs = ['pwr', 'bwr', 'phwr']
    # create empty dataframe
    preds = ['burnup', 'cooling', 'enrichment']
    algcol = ['knn', 'dtree', 'mll']
    levels = [algcol, []]
    df = pd.DataFrame(index=[preds, rxtrs], columns=pd.MultiIndex.from_product(levels, names=["Algorithm", "Metric"]))
    
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
            ### Error Calcs
            df = reg_rxtr_type(df, pred, rxtr, mll, knn, dtr)
    
    pklname = 'rxtr-type_mll_scikit_compare.pkl'
    df.to_pickle(rdrive + 'processed_results/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

