'''
Created on Oct 17, 2012

@author: georgianadinu
'''

import numpy as np
from scipy import stats
   

def score(gold, prediction, method):
    if len(gold) != len(prediction):
        raise ValueError("The two arrays must have the same length!")
    
    gold = np.array(gold)
    prediction = np.array(prediction)
    
    if method == "pearson":
        return pearson(gold, prediction)[0]
    elif method == "spearman":
        return spearman(gold, prediction)[0]
    elif method == "auc":
        return auc(gold, prediction)
    else:
        raise NotImplementedError("Unknown scoring measure:%s" % method)

def pearson(gold, prediction):
    return stats.pearsonr(gold, prediction)

def spearman(gold, prediction):
    return stats.spearmanr(gold, prediction, 0)

def auc(gold, prediction):

    positive = float(gold[gold == 1].size)
    negative = float(gold.size - positive)
     
    total_count = gold.size
    point_set = np.empty(total_count, dtype = [('gold',float),('score',float)])
    for i in range(total_count):
        point_set[i]=(gold[i], prediction[i])
         
    point_set.sort(order = 'score')
    
    xi = 1.0
    yi = 1.0
    xi_old = 1.0
    true_positive = positive
    false_positive = negative
    auc = 0
    
    for i in range(total_count):
        if (point_set[i][0] == 1):
            true_positive -= 1
            yi = true_positive / positive
        else:
            false_positive -= 1
            xi = false_positive / negative
            auc += (xi_old - xi) * yi
            xi_old = xi
            
    return auc
 