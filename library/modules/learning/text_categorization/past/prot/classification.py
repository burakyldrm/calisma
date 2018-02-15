'''
Created on Jan 19, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import os
from time import time

import numpy as np

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import io_utils
#import text_categorization.prototypes.preprocessor as preprocessor

#import text_preprocessor
import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline
from text_categorization import tc_utils2
import sklearn.calibration as skcalibrated

#import CLSF_CONSTANTS

from text_categorization.prototypes import CLSF_CONSTANTS 

# this is not necessarily abstract
class TextClassifier():
    
   
    feature_pipelines = None
    classifier = None
    clsf_name = ""
    
    def __init__(self, feature_pipelines=None, classifier=None):
    

        if feature_pipelines:
            self.feature_pipelines = feature_pipelines
        else:
            self.feature_pipelines = self.get_default_features()

        if classifier:
            #self.classifier = classifier
            self.clsf_name = classifier.__class__.__name__
            self.classifier = skcalibrated.CalibratedClassifierCV(classifier, 
                                                                  cv=CLSF_CONSTANTS._calibration_nfolds, 
                                                                  method=CLSF_CONSTANTS._calibration_method)
        else:
            self.classifier = self.get_default_classifier()
            
    
    
       
    
    def cross_validated_classify(self, instances, labels, nfolds=5):
            
        print("n_instances: ", len(instances))
        print("classifier: ", self.clsf_name)
        t0 = time()
        
        #print(self.feature_pipelines.get_params())
    
        model = skpipeline.Pipeline([('features', self.feature_pipelines),
                                     ('clsf', self.classifier)])
    
        print("Start classification\n..")
        
        print(model.get_params())
        print("LABELS", list(set(labels)))
        ypred = cv.cross_val_predict(model, instances, labels, cv=nfolds)
        accuracy, fscore = self.get_performance(labels, ypred, verbose=True)
        t1 = time()
        duration = round(t1 - t0, 2)
        print(nfolds, "-cross validated classification took ", duration, "sec.")
        
        return accuracy, fscore, duration
     
    
    
    def get_default_classifier(self):
        
        main_classifier = sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
        # classifier = nb.MultinomialNB()
        self.clsf_name = main_classifier.__class__.__name__
        classifier = skcalibrated.CalibratedClassifierCV(main_classifier, 
                                                         cv=CLSF_CONSTANTS._calibration_nfolds, 
                                                         method=CLSF_CONSTANTS._calibration_method)
        
        return classifier
    
     
    def get_default_features(self):
        
        tfidfvect = TfidfVectorizer()  
        
        features = skpipeline.FeatureUnion([('word_tfidf', tfidfvect),
                                            ])
        
        return features
     
    
    
    def train(self, train_instances, train_labels):
           
        print("n_train_instances: ", len(train_instances))
        print("classifier: ", self.clsf_name)
        
        t0 = time()
    
        model = skpipeline.Pipeline([('features', self.feature_pipelines),
                                     ('clsf', self.classifier)])
    
        print(model.get_params())
        
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

    
        # this runs prediction twice; one for labels, one probabilities. above fixed.
#===============================================================================
#     def predict(self, model, test_instances, test_labels=None):
#         
#         t0 = time()
#         print("Start prediction")
#         print("model type:", type(model))
#         predicted_labels = model.predict(test_instances)
#         t1 = time()
#         print("Prediction took ", round(t1 - t0, 2), "sec.")
#         
#         if test_labels:
#             self.get_performance(test_labels, predicted_labels, verbose=True)
# 
#         print("n_test_instances: ", len(test_instances))
#         
#         prediction_probabilities = model.predict_proba(test_instances)
# 
#         categories = model.classes_
#         '''
#         print(categories)
#         print(len(test_instances))
#         print(prediction_probabilities)
#         '''
#         labels_probs = []
#         for i in range(len(test_instances)):
#             label_index = np.argmax(prediction_probabilities[i])
#             label = categories[label_index]
#             prob = np.max(prediction_probabilities[i])
#             labels_probs.append((label, prob))
#         
#         print(labels_probs)
#         
#         #prediction_map = []   # [{predicted_label: .., prediction_probability: ..}]
#         prediction_map = [{"predicted_label" : label,
#                            "prediction_probability" : max(probs)} 
#                           for label,probs in zip(predicted_labels, prediction_probabilities)]
#         
#         
#             
#         return predicted_labels, prediction_map
#===============================================================================

    
    def save_model(self, model, path):    
        joblib.dump(model, path)
        
    
    def read_model(self, path):
        model = joblib.load(path)
        return model
    
    
    def train_and_save_model(self, train_instances, train_labels,
                             picklefolder="", modelname=""):
        
        model = self.train(train_instances, train_labels)

        recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
        recordpath = os.path.join(recordfolder, "model.b")
        self.save_model(model, recordpath)
        
        return model, recordpath


    def get_performance(self, ytrue, ypred, verbose=True):
        
        return tc_utils2.get_performance(ytrue, ypred, verbose)





def get_text_classifier(features, classifier):
    
    text_classifier = TextClassifier(feature_pipelines=features,
                                     classifier=classifier)
    return text_classifier



def dump_classification_system(model,
                                 text_classifier,
                                 picklefolder,
                                 modelname):
    
    
    recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
    
    modelpath = os.path.join(recordfolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
    classifierpath = os.path.join(recordfolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)

    joblib.dump(model, modelpath)
    joblib.dump(text_classifier, classifierpath)
    
    return recordfolder
    
def load_classification_system(picklefolder):
    

    
    
    modelpath = os.path.join(picklefolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
    classifierpath = os.path.join(picklefolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
    
    model = joblib.load(modelpath)
    text_classifier = joblib.load(classifierpath)
    
    return model, text_classifier




def train_and_save_model(texts, labels,
                                features_pipeline,
                                classifier,
                                picklefolder,
                                modelname):
    


    analyser = TextClassifier(features_pipeline, classifier)
        
    model = analyser.train(texts, labels)
    
    modelfolder = dump_classification_system(model, analyser, picklefolder, modelname)
    
    #analyser.cross_validated_classify(texts, labels)

    return model, modelfolder



def train_and_save_model2(texts, labels, 
                          clsf_system,
                          picklefolder,
                          modelname):
    
    model = clsf_system.train(texts, labels)
    
    print("Training finished.")
    
    
    modelfolder = dump_classification_system(model, clsf_system, picklefolder, modelname)
    print("Model written on the disc.")
    #clsf_system.cross_validated_classify(texts, labels)

    return model, modelfolder



'''
def build_cross_val_classification_system(texts, labels,
                                config_dict,
                                pipeline_builder,
                                picklefolder,
                                modelname):
    

    
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = pipeline_builder(feature_params)
    classifier = config_dict[conf.classifier_key]
    
    #features_pipeline, classifier = pipeline_builder()
    analyser = TextClassifier(features_pipeline, classifier)
        
    model = analyser.train(texts, labels)
    
    modelfolder = dump_classification_system(model, analyser, picklefolder, modelname)
    
    analyser.cross_validated_classify(texts, labels)

    return model, modelfolder
'''

def run_saved_model(modelfolder,
              test_instances, test_labels=None):

    model, analyser = load_classification_system(modelfolder)
    
    ypred, prediction_map = analyser.predict(model, test_instances)
    
    '''
    ytrue = test_labels
    if not ytrue:
        ytrue = [None]*len(ypred)
         
        
    print("Actual\tPredicted")
    for actual,predicted in zip(ytrue, ypred):
        print(actual,"\t",predicted)
    
    
    print("\n****Prediction map per instance")
    print(prediction_map)
    print("****")
    '''
    
    if test_labels:
        print("Testing performance:")
        analyser.get_performance(test_labels, ypred, verbose=True)
    
    return ypred, prediction_map


def cross_validated_classify(texts, labels,
                                features_pipeline,
                                classifier):


    analyser = TextClassifier(features_pipeline, classifier)
                
    accuracy, fscore, duration = analyser.cross_validated_classify(texts, labels)
    return accuracy, fscore, duration






if __name__ == '__main__':

    print()
    
    
    
    
    
