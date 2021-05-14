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
    parser.add_argument('detpath', metavar='detector-path', 
                        help='string indicating a detector for which the spectra are being processed')
    parser.add_argument('winlist', metavar = 'window-list',
                        help='list of energy windows to process spectra into training set')
    parser.add_argument('delta', metavar = 'energy-delta', type=int,
                        help='value in keV that forms size of energy window')
    parser.add_argument('nchannel', metavar = 'num-channels', type=int,
                        help='number of channels of spectra for given detector')
    parser.add_argument('pklname', metavar = 'pickle-name',
                        help='filename for trainset output')
    args = parser.parse_args(sys.argv[1:])
    
    # grab energy bins to create list of peaks training set
    # created in en windows notebook using source activities of idx 88087
    path = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    gad_path = path + 'detector_response/'
    en_windows_fname = args.winlist
    with open(gad_path + en_windows_fname, 'rb') as filehandle:
        en_windows = pickle.load(filehandle)
    # grab labels for the tracked peaks training set
    actsdf = pd.read_pickle(path + 'nuc32_activities_scaled_1g_reindex.pkl')
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 
            'OrigenReactor', 'AvgPowerDensity', 'ModDensity', 'UiWeight'
            ]
    lbls_df = actsdf[lbls]

    detect_path = gad_path + args.detpath
    energy_bins = get_energy_bins(detect_path + 'energy_bins.dat')
    en_delta = args.delta
    nchan = args.nchannel
    # output of `ls -1 | wc -l` in det directories is 5008
    # need to exclude energy_bins.dat: range is 0 --> 5006(+1)
    for i in range(0, 5007):
    #for i in range(0,1): #for shorter train set to test code
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
    
    #lbls_df = actsdf.iloc[0:peaksdf.index.tolist()[-1]+1][lbls] #for making shorter train set to test code
    labeled_peaksdf = pd.concat([lbls_df, peaksdf], axis=1)
    labeled_peaksdf.to_pickle(gad_path + args.pklname, compression='gzip')

    return

if __name__ == "__main__":
    main()
