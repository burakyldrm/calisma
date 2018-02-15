'''
Created on Nov 2, 2016

@author: dicle
'''

import numpy as np
import sklearn.metrics as mtr


class Category:
    def __init__(self):
        self.name = ''
        self.description=''
        self.size = 3
        self.accuracy =0
        self.precision =0
        self.recall =0
        self.truePositive=0
        self.trueNegative=0
        self.falsePositive=0
        self.falseNegative=0
        self.keywords = ''    
        
class Root:
    def __init__(self):
        self.categoryList =  []
        self.size = 0
        self.accuracy =0
        self.precision =0
        self.recall =0
        self.truePositive=0
        self.trueNegative=0
        self.falsePositive=0
        self.falseNegative=0
        self.keywords = ''

def roundFloat(val):
    return int(round(float(val)*100)) 

def get_performance(ytrue, ypred, verbose=True):
    
    root=Root()
    root.name='ROOT'
    root.size=0
    
    acc = mtr.accuracy_score(ytrue, ypred)
    # fscore = mtr.f1_score(ytrue, ypred, average="weighted", pos_label=None) #, labels=list(set(ytrue)))
    fscore = mtr.f1_score(ytrue, ypred, average="macro", pos_label=None)  # , labels=list(set(ytrue)))

    root.accuracy=roundFloat(fscore)
    # fscore=0
    
    confmatrix = mtr.confusion_matrix(ytrue, ypred)
    report = mtr.classification_report(ytrue, ypred)
    
    #sonuclar alınıyor
    map_classification_report(report,root)
    
    if verbose:        
        print("accuracy: ", acc)
        print("f1 score: ", fscore)
        print("confusion matrix\n", confmatrix)
        print("classification report\n", report)
        
        print(mtr.precision_recall_fscore_support(ytrue, ypred))
    
    
    #c = list(set(ytrue))[1]
    #print("conf matrix for class ", c)
    #print(mtr.confusion_matrix(ytrue, ypred, labels=[c]))
    #tn, fp, fn, tp = mtr.confusion_matrix(ytrue, ypred, labels=[c]).ravel()
    #print(tn, fp, fn, tp)
    
    labels = list(set(ytrue))
    
    for i, label in enumerate(labels):
               
        print(i, " counts for label ", label)
        tp, fp, fn, tn, ntrue = get_item_counts(ytrue, ypred, i)
        print("tp, fp, fn, tn, ntrue : ", tp, fp, fn, tn, ntrue)
        
        '''
        for x in root.categoryList:
            print("root label name : " +x.name)
            print(type(x.name))
            print(type(label))
            if(str(x.name)==str(label)):
                print("kosul saglandi : "+x.name)
        '''   
        
        #burası uygun labela gore ekleme yapıyor
        lsCategory=[x for x in root.categoryList if x.name == str(label).strip()]
        category=lsCategory[0]
        category.truePositive=tp
        category.falsePositive=fp
        category.falseNegative=fn
        category.trueNegative=tn
        #print()
    
    '''
    for i in range(len(root.categoryList)):
        print("ornek "+str(i))
        _ct=root.categoryList[i]
        print("categor bilgileri")
        print(_ct.name)
        print(_ct.precision)
        print(_ct.recall)
        print(_ct.accuracy)
        print(_ct.size)
        print(_ct.truePositive)
        print(_ct.falsePositive)
    '''
    print("root accuracy")
    print(root.accuracy)
    return acc, fscore,root


def map_classification_report(report, root):
    
    lines = report.split('\n')
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        class_val=t[:-4]
        className = ' '.join(class_val)
        cate=Category()
        cate.name=className
        cate.precision=roundFloat(t[-4])
        cate.recall=roundFloat(t[-3])
        cate.accuracy=roundFloat(t[-2])
        cate.size=int(t[-1])  
        root.size+=cate.size      
        root.categoryList.append(cate)
        '''
        className = str(t[0].replace(' ', ''))
        cate=Category()
        cate.name=className
        cate.precision=roundFloat(t[1])
        cate.recall=roundFloat(t[2])
        cate.accuracy=roundFloat(t[3])
        cate.size=int(t[4])  
        root.size+=cate.size      
        root.categoryList.append(cate)
        '''
        '''
        print('classname')
        print(className)
        print(t[1])
        print(t[2])
        print(t[3])
        print(t[4])
        '''
        
        #statistic.precisionMap[className] = t[1]
        #statistic.recallMap[className] = t[2]
        #statistic.f1ScoreMap[className] = t[3]
        #statistic.supportMap[className] = t[4]

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



