#!/bin/bash

# run script using ./loop_dets.sh

# This script runs the process_spectra.py script for all the input arg rows in
# params_process_spectra.txt. 

# already done
# d1_hpge/,idx88087_energy_list_113.pkl,2,8192,d1_hpge_spectra_xxpeaks_trainset.pkl.gz
# d2_detective_hpge/,idx88087_energy_list_113.pkl,3,8192,d2_hpge_spectra_xxpeaks_trainset.pkl.gz

while read p; do
    time ./process_spectra.py $p
done <params_process_spectra.txt
