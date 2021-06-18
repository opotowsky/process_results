#! /usr/bin/env python3

import argparse
import pickle
import numpy as np
import pandas as pd
import glob
import csv
import os

def format_gdf(filename):
    """
    This takes a filepath and reads the csv data in as a dataframe.
    There are different requirements for the gamma file bc opus gives
    stupid output - so can't use pandas functionality

    Parameters
    ----------
    filename : str of simulation output in a csv file

    Returns
    -------
    data : pandas dataframe containing csv entries

    """
    time_idx = []
    spectrum = []
    spectra = []
    gamma_bins = ()
    with open(filename) as f:
        gamma = csv.reader(f, delimiter=',')
        for i, row in enumerate(gamma, 1):
            if len(row) > 0:
                if i < 6:
                    pass
                elif i == 6:
                    time_idx.append(row[0])
                elif row[1]=='days':
                    spectra.append(spectrum)
                    time_idx.append(row[0])
                    spectrum = []
                else:
                    if i in range(7, 209):
                        if (i > 7 and gamma_bins[-1]==row[0]):
                            row[0] = row[0] + '.1'
                        gamma_bins = gamma_bins + (row[0],)    
                    spectrum.append(row[1])
        spectra.append(spectrum)
    data = pd.DataFrame(spectra, index=time_idx, columns=gamma_bins)
    return data

def format_ndf(filename):
    """
    This takes a filepath and reads the csv data in as a dataframe.

    Parameters
    ----------
    filename : str of simulation output in a csv file

    Returns
    -------
    data : pandas dataframe containing csv entries

    """
    
    data = pd.read_csv(filename, header=5, index_col=0).T
    if 'subtotal' in data.columns:
        data.drop('subtotal', axis=1, inplace=True)
    return data

def label_data(labels, data):
    """
    Takes the labels for and a dataframe of the simulation results; 
    adds these labels as additional columns to the dataframe.

    Parameters
    ----------
    labels : dict representing the labels for a simulation
    data : dataframe of simulation results

    Returns
    -------
    data : dataframe of simulation results + label entries in columns

    """
    
    col = len(data.columns)
    cooling_tmp = [0] + list(labels['CoolingInts']) + [0]
    burnups =  [ burnup for burnup in labels['Burnups'] for cooling in cooling_tmp ]
    coolings = cooling_tmp * len(labels['Burnups'])

    # the above process puts an extra entry on the end of each list
    burnups.pop()
    coolings.pop()

    #still need a pre-0 to start
    burnups.insert(0, 0)
    coolings.insert(0, 0)
    
    # inserting 4 prediction labels into columns
    data.insert(loc = col, column = 'ReactorType', value = labels['ReactorType'])
    data.insert(loc = col+1, column = 'CoolingTime', value = tuple(coolings))
    data.insert(loc = col+2, column = 'Enrichment', value = labels['Enrichment'])
    data.insert(loc = col+3, column = 'Burnup', value = tuple(burnups))
    # added the origen reactor for indepth purposes
    data.insert(loc = col+4, column = 'OrigenReactor', value = labels['OrigenReactor'])
    # added additional labels, but not for prediction
    data.insert(loc = col+5, column = 'ModDensity', value = labels['ModDensity'])
    data.insert(loc = col+6, column = 'AvgPowerDensity', value = labels['AvgPower'])
    data.insert(loc = col+7, column = 'UiWeight', value = labels['UiWeight'])

    return data

def dataframeXY(train_labels, info):
    """" 
    Takes list of all files in a directory (and rxtr-labeled subdirectories) 
    and produces a dataframe that has both the data features (X) and labeled 
    data (Y).

    Parameters
    ----------
    train_labels : list of dicts holding training lables and filenames
    info : string indicating the information source of the training data

    Returns
    -------
    dfXY : dataframe that has all features and labels for all simulations in a 
           directory

    """

    all_data = []
    for training_set in train_labels:
        if info == '_gamma':
            data = format_gdf(training_set['filename'])
        else:
            data = format_ndf(training_set['filename'])
        labeled = label_data(training_set, data)
        labeled.drop_duplicates(keep='last', inplace=True)
        all_data.append(labeled)
    dfXY = pd.concat(all_data, sort=True)
    dfXY.fillna(value=0, inplace=True)
    return dfXY

def main():
    """
    Takes all origen files in the hard-coded datapath and compiles them into
    the appropriate dataframe for a data set ready for pandas/scikit learn use.
    Saves the data set as a pickle file.

    """
    parser = argparse.ArgumentParser(description='Takes a set of origen sims in a directory and saves all sim results together in a .pkl file.')
    parser.add_argument('data_dir', help='directory name in origen directory (e.g., 20mar2019_description)')
    parser.add_argument('pkl_name', help='name of pickle file in origen directory that contains simulation parameters (e.g., varied_tset.pkl)')
    args = parser.parse_args()
        
    origen_dir = '../origen-data/'
    data_dir = args.data_dir
    datapath = origen_dir + data_dir + '/' 
    print('Is {} the correct data set directory?\n'.format(datapath), flush=True)
    # Grab data set labels
    if 'test' in datapath:
        testset = True
    else:
        testset = False
    pkl_labels = datapath + args.pkl_name
    data_set_labels = pickle.load(open(pkl_labels, 'rb'))
    opus_files = {'_nuc29' : '_masses.pkl', '_nuc62' : '_activities.pkl'}
    for ofile, pkl_end in opus_files.items():
        for i, sim in enumerate(data_set_labels, 1):
            o_rxtr = sim['OrigenReactor']
            enrich = sim['Enrichment']
            mod_d = sim['ModDensity']
            power = sim['AvgPower']
            rxtrpath = datapath + o_rxtr + "/"
            csv_base = '_enr' + str(enrich) + '_mod' + str(mod_d) + '_pwr' + str(power) + ofile
            if testset == True:
                csvfile = o_rxtr + '_' + str(i)  + csv_base + '_' + str(i) + '.csv'
            else:
                csvfile = o_rxtr + csv_base + '.csv'
            filepath = os.path.join(rxtrpath, csvfile)
            sim['filename'] = filepath
        dataXY = dataframeXY(data_set_labels, ofile)
        if '00' in ofile:
            ofile = 'gspec'
        pkl_set = 'not-scaled' + ofile + pkl_end
        pickle.dump(dataXY, open(pkl_set, 'wb'), protocol=4)
    return

if __name__ == "__main__":
    main()
