#!/bin/bash

# run script using ./loop_dets.sh

# This script runs the process_spectra.py script for all the input arg rows in
# params_process_spectra.txt. 

# already done
#d1_hpge/ energy_list_short.pkl 2 8192 d1_hpge_spectra_short_peaks_trainset.pkl.gz
#d2_detective_hpge/ energy_list_short.pkl 3 8192 d2_hpge_spectra_short_peaks_trainset.pkl.gz
#d3_czt/ energy_list_short.pkl 8 1024 d3_czt_spectra_short_peaks_trainset.pkl.gz
#d4_nai/ energy_list_short.pkl 12 1024 d4_nai_spectra_short_peaks_trainset.pkl.gz
#d5_labr3/ energy_list_short.pkl 12 1024 d5_labr3_spectra_short_peaks_trainset.pkl.gz
#d6_sri2/ energy_list_short.pkl 10 1024 d6_sri2_spectra_short_peaks_trainset.pkl.gz
#d1_hpge/ energy_list_long.pkl 2 8192 d1_hpge_spectra_long_peaks_trainset.pkl.gz
#d2_detective_hpge/ energy_list_long.pkl 3 8192 d2_hpge_spectra_long_peaks_trainset.pkl.gz
#d3_czt/ energy_list_long.pkl 8 1024 d3_czt_spectra_long_peaks_trainset.pkl.gz
#d4_nai/ energy_list_long.pkl 12 1024 d4_nai_spectra_long_peaks_trainset.pkl.gz
#d5_labr3/ energy_list_long.pkl 12 1024 d5_labr3_spectra_long_peaks_trainset.pkl.gz
#d6_sri2/ energy_list_long.pkl 10 1024 d6_sri2_spectra_long_peaks_trainset.pkl.gz
#d1_hpge/ idx133717_energy_list_d1auto.pkl 2 8192 d1_hpge_spectra_auto_peaks_trainset.pkl.gz
#d2_detective_hpge/ idx133717_energy_list_d2auto.pkl 3 8192 d2_hpge_spectra_auto_peaks_trainset.pkl.gz
#d3_czt/ idx133717_energy_list_d3auto.pkl 8 1024 d3_czt_spectra_auto_peaks_trainset.pkl.gz
#d4_nai/ idx133717_energy_list_d4auto.pkl 12 1024 d4_nai_spectra_auto_peaks_trainset.pkl.gz
#d5_labr3/ idx133717_energy_list_d5auto.pkl 12 1024 d5_labr3_spectra_auto_peaks_trainset.pkl.gz
#d6_sri2/ idx133717_energy_list_d6auto.pkl 10 1024 d6_sri2_spectra_auto_peaks_trainset.pkl.gz

# not yet done

while read p; do
    time ./process_spectra.py $p
done <params_process_spectra.txt
