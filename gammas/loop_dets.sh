#!/bin/bash

# run script using ./loop_dets.sh

# This script runs the process_spectra.py script for all the input arg rows in
# params_process_spectra.txt. 

# already done

# not yet done
#d4_nai/ idxYY_energy_list_xx.pkl 12 1024 d4_nai_spectra_xxpeaks_trainset.pkl.gz
#d5_labr3/ idxYY_energy_list_xx.pkl 12 1024 d5_labr3_spectra_xxpeaks_trainset.pkl.gz
#d6_sri2/ idxYY_energy_list_xx.pkl 10 1024 d6_sri2_spectra_xxpeaks_trainset.pkl.gz

while read p; do
    time ./process_spectra.py $p
done <params_process_spectra.txt
