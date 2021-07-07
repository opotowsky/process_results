
while read p; do
    #time ./mll_scikit_process_preds.py $p
    time ./randerr_mll_scikit_process_preds.py $p
done <preds_metrics.txt
