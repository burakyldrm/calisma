'''
Created on Jan 19, 2017

@author: dicle
'''
import os,json
from time import time

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline
import sklearn.calibration as skcalibrated

from modules.dataset import io_utils
from modules.dataset import tc_utils2
from modules.prototypes import clsf_constants 





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
                                                                  cv=clsf_constants._calibration_nfolds, 
                                                                  method=clsf_constants._calibration_method)
        else:
            self.classifier = self.get_default_classifier()
            
    
       
    
    def cross_validated_classify(self, instances, labels):
            
        print("n_instances: ", len(instances))
        print("classifier: ", self.clsf_name)
        #print("classifier: ", self.classifier.__class__.__name__)
       
        t0 = time()
    
        model = skpipeline.Pipeline([('features', self.feature_pipelines),
                                     ('clsf', self.classifier)])
    
        print("Start classification\n..")
        nfolds = 5
        print("labels",list(set(labels)))
        ypred = cv.cross_val_predict(model, instances, labels, cv=nfolds)
        accuracy, fscore = self.get_performance(labels, ypred, verbose=True)
        t1 = time()
        duration = round(t1 - t0, 2)
        print("Classification took ", duration, "sec.")
        
        return accuracy, fscore, duration
     
    
    
    def get_default_classifier(self):
        
        classifier = sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
        # classifier = nb.MultinomialNB()
        return classifier
    
     
    def get_default_features(self):
        
        tfidfvect = TfidfVectorizer()  
        
        #features = skpipeline.FeatureUnion([('word_tfidf', tfidfvect)])
        features = skpipeline.FeatureUnion([('word_tfidf', tfidfvect),])
        return features
     
    
    
    def train(self, train_instances, train_labels):
           
        print("n_train_instances: ", len(train_instances))
        print("classifier: ", self.clsf_name)
        #print("classifier: ", self.classifier.__class__.__name__)
        
        t0 = time()
    
        model = skpipeline.Pipeline([('features', self.feature_pipelines),
                                     ('clsf', self.classifier)])
    
        print("Start training\n..")
        model.fit(train_instances, train_labels)
        
        
        t1 = time()
        print("Traning took ", round(t1 - t0, 2), "sec.")
        return model

    
    
    def predict(self, model, test_instances, test_labels=None):
        
        t0 = time()
        print("Start prediction")
        predicted_labels = model.predict(test_instances)
        t1 = time()
        print("Prediction took ", round(t1 - t0, 2), "sec.")
        
        if test_labels:
            self.get_performance(test_labels, predicted_labels, verbose=True)

        print("n_test_instances: ", len(test_instances))
        
        prediction_probabilities = model.predict_proba(test_instances)
        #print(type)
        #print(prediction_probabilities)
        #print(model.get_params())
        #print(model.classes_)
                
        ''''
        categories = model.classes_
        print("**********")
        for i in range(len(test_instances)):
            print("Prediction probabilities for test instance #", i)
            for j, label in enumerate(categories):                                
                print(label, " : ", prediction_probabilities[i][j])
        '''        
        #print("**********")        
        comparasion = '{\"results\":['
        categories = model.classes_
        k=0;
        for i in range(len(test_instances)):
            #print("Prediction probabilities for test instance #", i)
            for j, label in enumerate(categories):     
                if k != 0:
                    comparasion += ','
                comparasion +=  "{ \"category\":\""+ label+ "\","+ " \"probability\": " + str(roundFloat(prediction_probabilities[i][j])) + ", \"description\":"  + "null"  + "}"                           
                #print(label, " : ", prediction_probabilities[i][j])
                k+=1
        comparasion += ']}'
        
        
        '''
        prediction_map = []   # [{predicted_label: .., prediction_probability: ..}]
        prediction_map = [{"predicted_label" : label,
                           "prediction_probability" : max(probs)} 
                          for label,probs in zip(predicted_labels, prediction_probabilities)]
        '''
        return predicted_labels, comparasion

    
    def save_model(self, model, path):    
        joblib.dump(model, path)
        
    
    def read_model(self, path):
        model = joblib.load(path)
        return model
    
    def get_performance(self, ytrue, ypred, verbose=True):    
        return tc_utils2.get_performance(ytrue, ypred, verbose)
    
'''    
def train_and_save_model(self, train_instances, train_labels,
                         picklefolder="", modelname=""):
    
    model = self.train(train_instances, train_labels)

    recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
    recordpath = os.path.join(recordfolder, "model.b")
    self.save_model(model, recordpath)
    
    return model, recordpath
'''




def roundFloat(val):
    return int(round(val*100))  


def get_text_classifier(features, classifier):
    
    text_classifier = TextClassifier(feature_pipelines=features,
                                     classifier=classifier)
    return text_classifier


def dump_classification_system(model,
                                 text_classifier,
                                 picklefolder,
                                 modelname):
    
    
    recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
    
    modelpath = os.path.join(recordfolder, clsf_constants.MODEL_FILE_NAME)
    classifierpath = os.path.join(recordfolder, clsf_constants.CLASSIFIER_FILE_NAME)

    joblib.dump(model, modelpath)
    joblib.dump(text_classifier, classifierpath)
    
    return recordfolder
    
def load_classification_system(picklefolder):
    

    
    
    modelpath = os.path.join(picklefolder, clsf_constants.MODEL_FILE_NAME)
    classifierpath = os.path.join(picklefolder, clsf_constants.CLASSIFIER_FILE_NAME)
    
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
    
    analyser.cross_validated_classify(texts, labels)

    return model, modelfolder

def train_and_save_model2(texts, labels, 
                          clsf_system,
                          picklefolder,
                          modelname):
    
        
    model = clsf_system.train(texts, labels)
    
    modelfolder = dump_classification_system(model, clsf_system, picklefolder, modelname)
    
    #clsf_system.cross_validated_classify(texts, labels)

    return model, modelfolder

def run_saved_model(modelfolder,
              test_instances, test_labels=None):

    model, analyser = load_classification_system(modelfolder)
    
    ypred, prediction_map = analyser.predict(model, test_instances)
    ytrue = test_labels
    if not ytrue:
        ytrue = [None]*len(ypred)
         
    
    print("predict*********")
    print(ypred)   
    #print("Actual\tPredicted")
    #for actual,predicted in zip(ytrue, ypred):
    #    print(actual,"\t",predicted)
    
    
    print("\n****Prediction map per instance")
    print(prediction_map)
    print("****")
       
    
    return ypred, prediction_map    

def cross_validated_classify(texts, labels,
                                features_pipeline,
                                classifier):


    analyser = TextClassifier(features_pipeline, classifier)
                
    accuracy, fscore, duration = analyser.cross_validated_classify(texts, labels)
    return accuracy, fscore, duration

model_sentiment_tr = None
analyser_sentiment_tr = None


model_sentiment_tr_twit = None
analyser_sentiment_tr_twit = None

def run_saved_model_sentiment_tr(modelfolder, test_instances, test_labels=None):
    global model_sentiment_tr, analyser_sentiment_tr
    if model_sentiment_tr is None or analyser_sentiment_tr is None:
        model_sentiment_tr, analyser_sentiment_tr = load_classification_system(modelfolder)

    ypred, prediction_map = analyser_sentiment_tr.predict(model_sentiment_tr, test_instances)
    ytrue = test_labels
    if not ytrue:
        ytrue = [None] * len(ypred)

    return ypred, prediction_map


def run_saved_model_sentiment_tr_twit(modelfolder, test_instances, test_labels=None):
    global model_sentiment_tr_twit, analyser_sentiment_tr_twit
    if model_sentiment_tr_twit is None or analyser_sentiment_tr_twit is None:
        model_sentiment_tr_twit, analyser_sentiment_tr_twit = load_classification_system(modelfolder)

    ypred, prediction_map = analyser_sentiment_tr_twit.predict(model_sentiment_tr_twit, test_instances)
    ytrue = test_labels
    if not ytrue:
        ytrue = [None] * len(ypred)

    return ypred, prediction_map

if __name__ == '__main__':

    print()
    
    
    
    
    
