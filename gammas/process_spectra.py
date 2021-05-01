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
    parser.add_argument('detector', choices=['d1', 'd2', 'd3', 'd4', 'd5', 'd6'], 
                        help='string indicating a detector for which the spectra are being processed')
    args = parser.parse_args(sys.argv[1:])
    
    # grab energy bins to create list of peaks training set
    # created in en windows notebook using source activities of idx 88087
    path = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    gad_path = path + 'detector_response/'
    #en_windows_fname = 'idx88087_energy_list_113.pkl'
    en_windows_fname = 'idx88087_energy_list_31.pkl'
    with open(gad_path + en_windows_fname, 'rb') as filehandle:
        en_windows = pickle.load(filehandle)
    # grab labels for the tracked peaks training set
    actsdf = pd.read_pickle(path + 'nuc32_activities_scaled_1g_reindex.pkl')
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 
            'OrigenReactor', 'AvgPowerDensity', 'ModDensity', 'UiWeight'
            ]
    lbls_df = actsdf[lbls]

    detect_info = {'d1' : {'det_path' : 'd1_hpge/',
                           'en_delta' : 2,
                           'num_channels' : 8192,
                           'pkl_name' : 'd1_hpge_spectra_31peaks_trainset.pkl.gz'
                           },
                   'd2' : {'det_path' : 'd2_detective_hpge/',
                           'en_delta' : 3,
                           'num_channels' : 8192,
                           'pkl_name' : 'd2_hpge_spectra_31peaks_trainset.pkl.gz'
                           },
                   'd3' : {'det_path' : 'd3_czt/',
                           'en_delta' : 8,
                           'num_channels' : 1024,
                           'pkl_name' : 'd3_czt_spectra_31peaks_trainset.pkl.gz'
                           },
                   'd4' : {'det_path' : 'd4_nai/',
                           'en_delta' : 20,
                           'num_channels' : 1024,
                           'pkl_name' : 'd4_nai_spectra_peaks_trainset.pkl.gz'
                           },
                   'd5' : {'det_path' : 'd5_labr3/',
                           'en_delta' : 20,
                           'num_channels' : 1024,
                           'pkl_name' : 'd5_labr3_spectra_peaks_trainset.pkl.gz'
                           },
                   'd6' : {'det_path' : 'd6_sri2/',
                           'en_delta' : 20,
                           'num_channels' : 1024,
                           'pkl_name' : 'd6_sri2_spectra_peaks_trainset.pkl.gz'
                           },
                   }
    
    detect_path = gad_path + detect_info[args.detector]['det_path']
    energy_bins = get_energy_bins(detect_path + 'energy_bins.dat')
    en_delta = detect_info[args.detector]['en_delta']
    nchan = detect_info[args.detector]['num_channels']
    # output of `ls -1 | wc -l` in d1 directory is 5008 
    # need to exclude energy_bins.dat: range is 0 --> 5006(+1)
    for i in range(0, 5007):
    #for i in range(0,1): #for shorter train set
        gz = detect_path + str(i) + '.dat.gz'
        gzdf = pd.read_csv(gz, sep=' ', index_col=0, header=None, usecols=range(0, nchan+1), names=['DbIdx',]+energy_bins, compression='gzip')
        windf = pd.DataFrame(columns=en_windows)
        for en in en_windows:
            low_idx = find_nearest_energy(en - en_delta, energy_bins)
            high_idx = find_nearest_energy(en + en_delta, energy_bins)
            windf[en] = gzdf.iloc[:, low_idx:high_idx+1].sum(axis=1)
        if i == 0:
            peaksdf = windf.copy()
        else:
            peaksdf = peaksdf.append(windf)
    
    #lbls_df = actsdf.iloc[0:peaksdf.index.tolist()[-1]+1][lbls] #for making shorter train set
    labeled_peaksdf = pd.concat([lbls_df, peaksdf], axis=1)
    labeled_peaksdf.to_pickle(gad_path + detect_info[args.detector]['pkl_name'], compression='gzip')

    return

if __name__ == "__main__":
    main()
