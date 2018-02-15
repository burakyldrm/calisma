'''
Created on Nov 2, 2016

@author: dicle
'''

import numpy as np
import sklearn.metrics as mtr


def get_performance(ytrue, ypred, verbose=True):
    
    acc = mtr.accuracy_score(ytrue, ypred)
    # fscore = mtr.f1_score(ytrue, ypred, average="weighted", pos_label=None) #, labels=list(set(ytrue)))
    fscore = mtr.f1_score(ytrue, ypred, average="micro", pos_label=None)  # , labels=list(set(ytrue)))

    # fscore=0
    
    confmatrix = mtr.confusion_matrix(ytrue, ypred)
    report = mtr.classification_report(ytrue, ypred)
    
    if verbose:
        print("accuracy: ", acc)
        print("f1 score: ", fscore)
        print("confusion matrix\n", confmatrix)
        print("classification report\n", report)
        
        print(mtr.precision_recall_fscore_support(ytrue, ypred))
    
    '''
    c = list(set(ytrue))[1]
    print("conf matrix for class ", c)
    print(mtr.confusion_matrix(ytrue, ypred, labels=[c]))
    #tn, fp, fn, tp = mtr.confusion_matrix(ytrue, ypred, labels=[c]).ravel()
    #print(tn, fp, fn, tp)
    
    labels = list(set(ytrue))
    
    for i, label in enumerate(labels):
        print(i, " counts for label ", label)
        tp, fp, fn, tn, ntrue = get_item_counts(ytrue, ypred, i)
        print("tp, fp, fn, tn, ntrue : ", tp, fp, fn, tn, ntrue)
        print()
    '''
    return acc, fscore



def get_item_counts(ytrue, ypred, labelindex):
    
        
    confmat = mtr.confusion_matrix(ytrue, ypred)
    
    n_all_instances = np.sum(confmat)
    n_true = sum(confmat[labelindex, :])
    n_predicted = sum(confmat[:, labelindex])
    
    tp = confmat[labelindex, labelindex]
    fn = n_true - tp
    fp = n_predicted - tp
    tn = n_all_instances - (tp + fp + fn)
    
    return tp, fp, fn, tn, n_true



if __name__ == "__main__":
    
    print(get_performance([1, 2], [1, 1]))



