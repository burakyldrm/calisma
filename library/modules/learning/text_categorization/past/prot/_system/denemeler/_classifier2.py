'''
Created on May 8, 2017

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

from dataset import corpus_io, io_utils
import prep_config
from text_categorization.prototypes import CLSF_CONSTANTS 
from text_categorization import tc_utils2


from abc import ABCMeta, abstractmethod



'''
    task_name = "" 
    
    feature_extraction_choice = None # FeatureChoice
    
    classifier = None
    
    
    _feature_union = None   # FeatureUnion()
        
    _text_classifier = None   # TextClassifier()
   
    
    @abstractmethod
    def __init__(self, task_name, 
                 feature_config, classifier):
        self.task_name = task_name
        self.feature_extraction_choice = feature_config
        self.classifier = classifier

'''
  

class _ClassificationTask():
    
    __metaclass__ = ABCMeta
    def __init__(self, task_name="", feature_config=None, classifier=None):
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
        main_classifier = sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
        # classifier = nb.MultinomialNB()
        self.clsf_name = main_classifier.__class__.__name__
        classifier = skcalibrated.CalibratedClassifierCV(main_classifier, 
                                                         cv=CLSF_CONSTANTS._calibration_nfolds, 
                                                         method=CLSF_CONSTANTS._calibration_method)
        
        return classifier
    
    @abstractmethod
    def get_default_feature_config(self):
        feat_config = prep_config.FeatureChoice()   # default values
                
        return feat_config


    def cross_validated_classify(self, instances, labels, nfolds=5):
            
        print("n_instances: ", len(instances))
        print("classifier: ", self.clsf_name)
        t0 = time()
        
        #print(self.feature_pipelines.get_params())
    
        model = skpipeline.Pipeline([('features', self.feature_union),
                                     ('clsf', self.classifier)])
    
        print("Start classification\n..")
        
        #print(model.get_params())
        #print("LABELS", list(set(labels)))
        ypred = cv.cross_val_predict(model, instances, labels, cv=nfolds)
        accuracy, fscore = self.get_performance(labels, ypred, verbose=True)
        t1 = time()
        duration = round(t1 - t0, 2)
        print(nfolds, "-cross validated classification took ", duration, "sec.")
        
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

    
    
    def predict(self, model, test_instances, test_labels=None):
        
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
            prob = np.max(prediction_probabilities[i])
            predicted_labels.append(label)
            prediction_map.append({"predicted_label" : label,
                                   "prediction_probability" : prob})
                    
        if test_labels:
            self.get_performance(test_labels, predicted_labels, verbose=True)
                       
        return predicted_labels, prediction_map


    


    
    def get_performance(self, ytrue, ypred, verbose=True):
        
        return tc_utils2.get_performance(ytrue, ypred, verbose)


################################################



class _ClassificationSystem():
    
    
    def __init__(self):
        
        
        



def dump_classification_system(model,
                                 task_obj,
                                 picklefolder,
                                 modelname):
    
    
    recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
    
    modelpath = os.path.join(recordfolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
    classifierpath = os.path.join(recordfolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)

    joblib.dump(model, modelpath)
    joblib.dump(task_obj, classifierpath)
    
    return recordfolder
    
def load_classification_system(picklefolder):
    

    
    
    modelpath = os.path.join(picklefolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
    classifierpath = os.path.join(picklefolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
    
    model = joblib.load(modelpath)
    task_obj = joblib.load(classifierpath)
    
    return model, task_obj

def train_and_save_model(texts, labels, 
                          task_obj,
                          picklefolder,
                          modelname):
    
    learning_model = task_obj.train(texts, labels)
    
    print("Training finished.")
    
    
    modelfolder = dump_classification_system(learning_model, task_obj, picklefolder, modelname)
    print("Model written on the disc.")
    #clsf_system.cross_validated_classify(texts, labels)

    return learning_model, modelfolder


def predict_offline(modelfolder, test_instances, test_labels):
    
    model, task_obj = load_classification_system(modelfolder)
    
    predicted_labels, prediction_map = task_obj.predict(model, test_instances, test_labels)
    return predicted_labels, prediction_map



###########################################################




    
    
    
def get_tr_sentiment_data(folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint",
                            fname="sentiment_reviews.csv",
                            sep="\t",
                            text_col="text",
                            cat_col="polarity"):
    
    path = os.path.join(folder, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(path, sep, text_col, cat_col)
    
    return instances, labels        

if __name__ == '__main__':

    print()
    
    
    
