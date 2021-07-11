
while read p; do
    time ./abs_rel_errs_mll_scikit_process.py $p
done <preds_enlists.txt
