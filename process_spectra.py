#! /usr/bin/env python3

import sys
import pickle
import argparse
import numpy as np
import pandas as pd

def find_nearest_energy(target_en, energy_bins):
    close_idx = (np.abs(np.asarray(energy_bins) - target_en)).argmin()
    return close_idx

def get_energy_bins(bindata):
    with open(bindata) as binfile:
        contents = binfile.readlines()
    binlist = []
    for ebin in contents[1].strip().split(' '):
        binlist.append(float(ebin))
    return binlist

def main():
    """
    """
    parser = argparse.ArgumentParser(description='Processes a database of gamma spectra into count windows.')
    parser.add_argument('detector', metavar='detector-descrip', 
                        help='string indicating a detector for which the spectra are being processed')
    args = parser.parse_args(sys.argv[1:])
    
    detect_info = {'d1' : {'en_windows' : 'd1_hpge_energy_list_113.pkl', # this was created in a jupyter notebook using idx 88087
                           'det_path' : 'd1_hpge/',
                           'en_delta' : 2, # energy window is plus/minus 2 keV
                           'pkl_name' : 'd1_hpge_spectra_peaks_trainset.pkl.gz'
                           },
                   'test':{'en_windows' : 'd1_hpge_energy_list_113.pkl', # this was created in a jupyter notebook using idx 88087
                           'det_path' : 'd1_hpge/',
                           'en_delta' : 2, # energy window is plus/minus 2 keV
                           'pkl_name' : 'test.pkl.gz'
                           },
                   }
    
    path = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/detector_response/'
    results_path = path + detect_info[args.detector]['det_path']
    en_windows_fname = detect_info[args.detector]['en_windows']
    with open(path + en_windows_fname, 'rb') as filehandle:
        en_windows = pickle.load(filehandle)
    
    en_delta = detect_info[args.detector]['en_delta']
    # output of `ls -1 | wc -l` in d1 directory is 5008 
    # need to exclude energy_bins.dat: range is 0 --> 5006(+1)
    energy_bins = get_energy_bins(results_path + 'energy_bins.dat')
    for i in range(0, 5007):
    #for i in range(0,10): #for shorter train set
        gz = results_path + str(i) + '.dat.gz'
        gzdf = pd.read_csv(gz, sep=' ', index_col=0, header=None, usecols=range(0, 8193), names=['DbIdx',]+energy_bins, compression='gzip')
        windf = pd.DataFrame(columns=en_windows)
        for en in en_windows:
            low_idx = find_nearest_energy(en - en_delta, energy_bins)
            high_idx = find_nearest_energy(en + en_delta, energy_bins)
            windf[en] = gzdf.iloc[:, low_idx:high_idx+1].sum(axis=1)
        if i == 0:
            peaksdf = windf.copy()
        else:
            peaksdf = peaksdf.append(windf)
    
    # label the tracked peaks
    actsdf = pd.read_pickle('nuc32_activities_scaled_1g_reindex.pkl')
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 
            'OrigenReactor', 'AvgPowerDensity', 'ModDensity', 'UiWeight'
            ]
    lbls_df = actsdf[lbls]
    #lbls_df = actsdf.iloc[0:peaksdf.index.tolist()[-1]+1][lbls] #for making shorter train set
    labeled_peaksdf = pd.concat([lbls_df, peaksdf], axis=1)
    labeled_peaksdf.to_pickle(path + detect_info[args.detector]['pkl_name'], compression='gzip')

    return

if __name__ == "__main__":
    main()