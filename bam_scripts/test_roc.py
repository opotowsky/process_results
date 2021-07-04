#! /usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, label_binarize
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, auc

def main():
    rdrive = '/mnt/researchdrive/BOX_INTERNAL/opotowsky/'
    CV = 5
    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'OrigenReactor']
    nonlbls = ['AvgPowerDensity', 'ModDensity', 'UiWeight']
    train = pd.read_pickle(rdrive + 'sim_grams_nuc29.pkl')
    # process train set
    X = train.drop(lbls, axis=1)
    for nonlbl in nonlbls+['total']:
        if nonlbl in X.columns:
            X.drop(nonlbl, axis=1, inplace=True)
    y = train.loc[:, lbls[0]]
    err = 0.10
    errs = np.random.uniform(1 - err, 1 + err, (X.shape[0], X.shape[1]))
    X = X * errs
    X = scale(X)
    # scikit init
    cv = StratifiedKFold(n_splits=CV, shuffle=True)
    knn_init = KNeighborsClassifier(n_neighbors=4, weights='distance', p=1, metric='minkowski')
    dtr_init = DecisionTreeClassifier(max_depth=56, max_features=None, class_weight='balanced')
    # binarize y
    rxtrs = ['bwr', 'phwr', 'pwr']
    y_bin = label_binarize(y, classes=rxtrs)
    n_classes = 3
    # cv predict
    y_score_knn = cross_val_predict(knn_init, X, y, cv=cv, method='predict_proba')
    y_score_dtr = cross_val_predict(dtr_init, X, y, cv=cv, method='predict_proba')
    # ROC process
    fpr_knn = dict()
    tpr_knn = dict()
    auc_knn = dict()
    fpr_dtr = dict()
    tpr_dtr = dict()
    auc_dtr = dict()
    for i in range(n_classes):
        fpr_knn[i], tpr_knn[i], kth = roc_curve(y_bin[:, i], y_score_knn[:, i])
        fpr_dtr[i], tpr_dtr[i], dth = roc_curve(y_bin[:, i], y_score_dtr[:, i])
        auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])
        auc_dtr[i] = auc(fpr_dtr[i], tpr_dtr[i])
        print(len(kth), len(dth))
    # save the dicts
    dict_list = [fpr_knn, tpr_knn, auc_knn, 
                 fpr_dtr, tpr_dtr, auc_dtr]
    pklname = rdrive + 'processed_results/roc_nuc29_err10.pkl'
    with open(pklname, 'wb') as pkl:
        pickle.dump(dict_list, pkl, protocol=4)

    return

if __name__ == "__main__":
    main()

