#! /usr/bin/env python3

import sys
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
#from mll_calc.all_jobs import job_dirs

def calc_errors(pred_df, true_lbls):
    """
    Given a dataframe containing predictions and log-likelihood value,
    calculates absolute error between predictions and ground truth (or boolean
    where applicable)

    Parameters
    ----------
    pred_df : dataframe with ground truth and predicted labels
    true_lbls : list of ground truth column labels 
    
    Returns
    -------
    pred_df : dataframe with ground truth, predictions, and errors between the 
              two
    
    """
    pred_lbls = ["pred_" + s for s in true_lbls] 
    for true, pred in zip(true_lbls, pred_lbls):
        if 'Reactor' in true:
            col_name = true + '_Score'
            pred_df[col_name] = np.where(pred_df.loc[:, true] == pred_df.loc[:, pred], True, False)
        else: 
            col_name = true + '_Error'
            pred_df[col_name] = np.abs(pred_df.loc[:, true]  - pred_df.loc[:, pred])
    return pred_df

def main():
    """
    
    
    """

    parser = argparse.ArgumentParser(description='Concatenates the contents of all CSVs in a directory')
    parser.add_argument('results_dir', metavar='results-directory',
                        help='name of directory with results, e.g. gam_spec/d1_n113')
    parser.add_argument('unc_dir', metavar='uncertainty-directory',
                        help='name of uncertainty directory within results dir, e.g. Job1_unc0.0')
    args = parser.parse_args(sys.argv[1:])
    
    results_path = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/mll/' \
                   + args.results_dir + '/' + args.unc_dir + '/'
    csvs = sorted(glob.glob(results_path + '*.csv'))
    pred_df = pd.concat((pd.read_csv(csv, header = 0) for csv in csvs))
    
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    pred_df = calc_errors(pred_df, lbls) 
    
    pred_df.to_csv(results_path + args.unc_dir + '.csv')

    for csv in csvs:
        subprocess.run(['rm', csv])

    return

if __name__ == "__main__":
    main()
