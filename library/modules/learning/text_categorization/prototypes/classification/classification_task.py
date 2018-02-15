'''
Created on Jan 19, 2017

@author: dicle
'''

import sys
sys.path.append("..")



import os
from time import time

import numpy as np



import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline
import sklearn.calibration as skcalibrated
from sklearn.externals import joblib


from modules.learning.text_categorization.prototypes.classification import CLSF_CONSTANTS, prep_config
from modules.learning.text_categorization.utils import tc_utils

from abc import ABCMeta, abstractmethod



class _ClassificationTask():
    
    __metaclass__ = ABCMeta
    def __init__(self, feature_config=None, classifier=None, task_name=""):
        # classifier can be inside feature_config
        
        self.task_name = task_name
        
        if feature_config:
            self.feature_config = feature_config
        else:
            self.feature_config = self.get_default_feature_config()
        
        
        # not very safe!! this should make sure feature_config is assigned
        self.feature_union = self._generate_feature_extraction_pipeline()
        
        if classifier:
            self.clsf_name = classifier.__class__.__name__
            self.main_classifier = classifier
            self.classifier = skcalibrated.CalibratedClassifierCV(classifier, 
                                                                  cv=CLSF_CONSTANTS._calibration_nfolds, 
                                                                  method=CLSF_CONSTANTS._calibration_method)
        else:
            self.classifier = self.get_default_classifier()
    
    @abstractmethod
    def _generate_feature_extraction_pipeline(self):
        
        if self.feature_config is None:
            self.feature_config = self.get_default_feature_config()
        pass
    
    @abstractmethod
    def get_default_classifier(self):
        self.main_classifier = sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
        # classifier = nb.MultinomialNB()
        self.clsf_name = self.main_classifier.__class__.__name__
        classifier = skcalibrated.CalibratedClassifierCV(self.main_classifier, 
                                                         cv=CLSF_CONSTANTS._calibration_nfolds, 
                                                         method=CLSF_CONSTANTS._calibration_method)
        
        return classifier
    
    @abstractmethod
    def get_default_feature_config(self):
        feat_config = prep_config.FeatureChoice()   # default values
                
        return feat_config

    
    def update_feature_config(self, new_feature_config):

        self.feature_config = new_feature_config

    def cross_validated_classify(self, instances, labels, nfolds=3):
            
        print("n_instances: ", len(instances))
        print("classifier: ", self.clsf_name)
        t0 = time()
        
        #print(self.feature_pipelines.get_params())
    
        model = skpipeline.Pipeline([('features', self.feature_union),
                                     ('clsf', self.classifier)])
    
        print("Start", nfolds, "-fold cross-validation \n..")
        
        #print(model.get_params())
        #print("LABELS", list(set(labels)))
        ypred = cv.cross_val_predict(model, instances, labels, cv=nfolds)
        accuracy, fscore,root = self.get_performance(labels, ypred, verbose=True)
        t1 = time()
        duration = round(t1 - t0, 2)
        print(nfolds, "-fold cross-validated classification took ", duration, "sec.")
        
        return accuracy, fscore, duration,root
    
    
    def measure_performance(self, instances, labels, nfolds=5):
            
        print("n_instances: ", len(instances))
        print("classifier: ", self.clsf_name)
        t0 = time()
        
        #print(self.feature_pipelines.get_params())
    
        
    
        print("Start", nfolds, "-cross-validation \n..")
        
        #print(model.get_params())
        #print("LABELS", list(set(labels)))
        import sklearn.model_selection as skmodelselect

        Xtrain, Xtest, ytrain, ytest = skmodelselect.train_test_split(instances, labels, test_size=0.4, random_state=0)
        
        main_model = skpipeline.Pipeline([('features', self.feature_union),
                                     ('main_clsf', self.main_classifier)])
        
        main_model.fit(Xtrain, ytrain)
        
        cv_model = skpipeline.Pipeline([('features', self.feature_union),
                                     ('main_clsf', self.classifier)])
        
        tsize = int(len(ytest) / 2)
        print(tsize, type(Xtest), len(Xtest), len(ytest))
        cv_model.fit(Xtest[:tsize], ytest[:tsize])
        ypred = cv_model.predict(Xtest[tsize:])
        accuracy, fscore = self.get_performance(ytest[tsize:], ypred, verbose=True)
        t1 = time()
        duration = round(t1 - t0, 2)
        print(nfolds, "-cross validated classification took ", duration, "sec.")
        
        '''
        import sklearn.model_selection as skmodelselect

        Xtrain, Xtest, ytrain, ytest = skmodelselect.train_test_split(instances, labels, test_size=0.4, random_state=0)
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        accuracy, fscore = self.get_performance(ytest, ypred, verbose=True)
        t1 = time()
        duration = round(t1 - t0, 2)
        print(nfolds, "-cross validated classification took ", duration, "sec.")
        '''
        return accuracy, fscore, duration

    def train(self, train_instances, train_labels):
           
        print("n_train_instances: ", len(train_instances))
        print("classifier: ", self.clsf_name)
        
        t0 = time()
    
        model = skpipeline.Pipeline([('features', self.feature_union),
                                     ('clsf', self.classifier)])
    
        #print(model.get_params())
        
        print("Start training\n..")
        model.fit(train_instances, train_labels)
                
        t1 = time()
        print("Traning took ", round(t1 - t0, 2), "sec.")
        return model

    
    
    def predict(self, model, test_instances) :  #, test_labels=None):
        
        t0 = time()
        print("Start prediction")
        prediction_probabilities = model.predict_proba(test_instances)
        t1 = time()
        
        print("Prediction took ", round(t1 - t0, 2), "sec.")


        categories = model.classes_
        '''
        print(categories)
        print(len(test_instances))
        print(prediction_probabilities)
        '''
        predicted_labels = []
        prediction_map = []
        for i in range(len(test_instances)):
            label_index = np.argmax(prediction_probabilities[i])
            label = categories[label_index]
            prob = prediction_probabilities[i] #np.max(prediction_probabilities[i])
            predicted_labels.append(label)
            
            cats_probs = {label : probability for label, probability in zip(categories, prediction_probabilities[i])}
            prediction_map.append({"predicted_label" : label,
                                   "prediction_probability" : cats_probs})
    
        
        
        comparasion = '{\"results\":['
        categories = model.classes_
        k=0;
        for i in range(len(test_instances)):
            #print("Prediction probabilities for test instance #", i)
            for j, label in enumerate(categories):     
                if k != 0:
                    comparasion += ','
                comparasion +=  "{ \"category\":\""+ str(label)+ "\","+ " \"probability\": " + str(roundFloat(prediction_probabilities[i][j])) + "}"                           
                #print(label, " : ", prediction_probabilities[i][j])
                k+=1
        comparasion += ']}'
    
        '''
        predicted_labels = []
        prediction_map = []
        for i in range(len(test_instances)):
            label_index = np.argmax(prediction_probabilities[i])
            label = categories[label_index]
            prob = prediction_probabilities[i]
            predicted_labels.append(label)
            prediction_map.append({"predicted_label" : label,
                                   "prediction_probability" : prob})
        
        '''
        '''           
        if test_labels:
            self.get_performance(test_labels, predicted_labels, verbose=True)
        '''            
        return predicted_labels, prediction_map, comparasion


    
    def getValues(classList, lists, categoryMap):
        comparasion = '{\"results\":['
        i = 0
        for className in classList:
                 
                 if i != 0:
                     comparasion += ','
                 if (className != None and className != ''):     
                     #comparasion +=  "{ \"category\":\"" + (str(className.decode())+ "\"," + " \"probability\": " + str(roundFloat(lists[i]))) + ", \"description\": \"" + str(categoryMap[className.decode()]) + "\"}"
                     comparasion +=  "{ \"category\":\"" + (str(className.decode())+ "\"," + " \"probability\": " + str(roundFloat(lists[i]))) + "\"}"                 
                     i+=1
                 
        comparasion += ']}'
        
        return comparasion

    
    def get_performance(self, ytrue, ypred, verbose=True):
        
        return tc_utils.get_performance(ytrue, ypred, verbose)


def roundFloat(val):
    return int(round(val*100))  
    
    
    
