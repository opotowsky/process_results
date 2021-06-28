#! /usr/bin/env python3

import sys
import pickle
import argparse
import pandas as pd
from tools import rxtr_randerr, reg_randerr

def main():
    parser = argparse.ArgumentParser(description='Processes mll and scikit prediction errors into .pkl files.')
    parser.add_argument('pred', metavar='prediction', choices = ['reactor', 'burnup', 'enrichment', 'cooling'],
                        help='string indicating which prediction error to process')
    parser.add_argument('metric', metavar='error-metric', choices = ['MAE', 'MedAE', 'MAPE', 'BalAcc', 'Acc'],
                        help='string indicating which error metric to use')
    args = parser.parse_args(sys.argv[1:])

    # for filepaths
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    mll_path = rdrive + 'mll/nuc_conc/nuc29/'
    learn_path = rdrive + 'scikit/nuc_conc/rand_err/'
    pred = args.pred
    # injected errors
    jobs = ['Job0_unc0.01', 'Job1_unc0.05', 'Job2_unc0.1', 'Job3_unc0.15', 'Job4_unc0.2']
    mll_errs = [1, 5, 10, 15, 20]
    sk_errs = [0, 0.3, 0.7, 1, 2, 4, 6, 8, 10, 13, 17, 20]
    all_errs = sorted(list(set(mll_errs) | set(sk_errs)))
    # create empty dataframe
    algcol = ['knn', 'dtree', 'mll']
    levels = [algcol, []]
    df = pd.DataFrame(index=all_errs, columns=pd.MultiIndex.from_product(levels, names=["Algorithm", "Metric"]))
    
    for err in all_errs:    
        #### MLL Results
        if err in mll_errs:
            i = mll_errs.index(err)
            mll = pd.read_csv(mll_path + jobs[i] + '/' + jobs[i] + '.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        else:
            mll = None
        ### Scikit Results
        if err in sk_errs:
            csv_end = '_tset1.0_nuc29_err' + str(err) + '_random_error.csv'
            knncsv = pred + '_knn' + csv_end
            dtrcsv = pred + '_dtree' + csv_end
            knn = pd.read_csv(learn_path + knncsv).drop(columns='Unnamed: 0')
            dtr = pd.read_csv(learn_path + dtrcsv).drop(columns='Unnamed: 0')
        else:
            knn = dtr = None
        ### Error Calcs
        if pred == 'reactor':
            df = rxtr_randerr(df, err, mll, knn, dtr, pred, args.metric)
        else:
            df = reg_randerr(df, err, mll, knn, dtr, pred, args.metric)
    print('{} {} pred df complete'.format(pred, args.metric), flush=True)
    
    pklname = args.pred + '_randerr_mll_scikit_compare_' + args.metric  + '.pkl'
    df.to_pickle(rdrive + 'processed_results/' + pklname, protocol=4)

    return

if __name__ == "__main__":
    main()

