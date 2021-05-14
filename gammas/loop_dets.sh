#!/bin/bash

# run script using ./loop_dets.sh

# This script runs the process_spectra.py script for all the input arg rows in
# params_process_spectra.txt. 

# already done

while read p; do
    time ./process_spectra.py $p
done <params_process_spectra.txt
