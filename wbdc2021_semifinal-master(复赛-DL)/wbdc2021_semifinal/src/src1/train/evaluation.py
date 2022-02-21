import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata
from joblib import Parallel, delayed


@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def uAUC(y_true, y_pred, userids):
    num_labels = y_pred.shape[1]
    y_true = y_true.reset_index(drop = True)
    y_pred = y_pred.reset_index(drop = True)
    def uAUC_infunc(i):
        print(i)
        uauc_df = pd.DataFrame()
        uauc_df['userid'] = userids
        uauc_df['y_true'] = y_true.iloc[:, i]
        uauc_df['y_pred'] = y_pred.iloc[:, i]


        label_nunique = uauc_df.groupby(by='userid')['y_true'].transform('nunique')
        uauc_df = uauc_df[label_nunique == 2]
        aucs = uauc_df.groupby(by='userid').apply(
            lambda x: auc(x['y_true'].values, x['y_pred'].values))
        return np.mean(aucs)

    uauc = Parallel(n_jobs=4)(delayed(uAUC_infunc)(i) for i in range(num_labels))
    return np.average(uauc, weights=[4, 3, 2, 1,1,1,1]), uauc


    
    
def evaluate_deepctr(val_labels,val_pred_ans,userid_list,target):

    weight_auc = uAUC(val_labels,val_pred_ans,userid_list)
    print("【UAUC：%s】" % (weight_auc[0]), weight_auc[1] )
    return weight_auc

def evaluate_deepctr_single(y_true,y_pred,userid_list):
    uauc_df = pd.DataFrame()
    uauc_df['userid'] = userid_list
    uauc_df['y_true'] = y_true
    uauc_df['y_pred'] = y_pred


    label_nunique = uauc_df.groupby(by='userid')['y_true'].transform('nunique')
    uauc_df = uauc_df[label_nunique == 2]
    aucs = uauc_df.groupby(by='userid').apply(
        lambda x: auc(x['y_true'].values, x['y_pred'].values))
    return np.mean(aucs)