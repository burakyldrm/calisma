'''
Created on May 11, 2017

@author: dicle
'''

import sys
sys.path.append("..")



import os

from sklearn.externals import joblib

from modules.learning.dataset import corpus_io, io_utils
from modules.learning.text_categorization.prototypes.classification import CLSF_CONSTANTS 

from abc import ABCMeta, abstractmethod

class ClassificationSystem():
    
    clsf_task = None   # _ClassificationTask()
    
    
    
    def __init__(self, Task,
                       task_name,
                       config_obj):
        
        self.clsf_task = Task(feature_config=config_obj,
                              classifier=config_obj.classifier,   # here or outside??
                              task_name=task_name
                              ) 
    
    
    '''
    def read_dataset(self,   train_data_folder,
                             train_data_fname,
                             text_col,
                             cat_col,
                             csvsep,
                             shuffle_dataset):
        texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, 
                                          csvsep, text_col, cat_col, shuffle_dataset) 
        return texts, labels
    '''
    
    
    
    def get_cross_validated_clsf_performance(self, instances, labels, nfolds=3):
        '''
        returns acc, fscore, duration as the results of nfolds-fold cross-validated classification.
        '''
        
        accuracy, fscore, duration,root = self.clsf_task.cross_validated_classify(instances, labels, nfolds)
        return accuracy, fscore, duration,root
    
    def build_system(self, 
                     Task,
                     task_name,
                     config_obj,
                     #classifier,
                     train_data_folder,
                     train_data_fname,
                     text_col,
                     cat_col,
                     csvsep,
                     shuffle_dataset,
                     cross_val_performance,
                     modelfolder,
                     modelname,
                     N=None):
         
        clsf_task = Task(feature_config=config_obj,
                                      classifier=config_obj.classifier,   # here or outside??
                                      task_name=task_name
                                      )
        texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, 
                                          csvsep, text_col, cat_col, shuffle_dataset)  # make this a member

        if N:
            texts = texts[:N]
            labels = labels[:N]
            modelname = modelname + "_" + str(N)
         
        model, modelpath = self.run_classification_system(clsf_task, texts, labels, modelfolder, modelname, cross_val_performance)
        
        return model, modelpath 
    
    def _dump_classification_system(self, model,
                                 task_obj,
                                 picklefolder,
                                 modelname):
    
    
        recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
        
        modelpath = os.path.join(recordfolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
        classifierpath = os.path.join(recordfolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
    
        joblib.dump(model, modelpath)
        joblib.dump(task_obj, classifierpath)
        
        return recordfolder
        
    def _load_classification_system(self, picklefolder):
        
    
        
        
        modelpath = os.path.join(picklefolder, CLSF_CONSTANTS.MODEL_FILE_NAME)
        classifierpath = os.path.join(picklefolder, CLSF_CONSTANTS.CLASSIFIER_FILE_NAME)
        
        model = joblib.load(modelpath)
        task_obj = joblib.load(classifierpath)
        
        return model, task_obj

    def train_and_save_model(self, texts, labels, 
                              picklefolder,
                              modelname):
        
        learning_model = self.clsf_task.train(texts, labels)
        
        print("Training finished.")
        
        
        modelfolder = self._dump_classification_system(learning_model, self.clsf_task, picklefolder, modelname)
        print("Model written on the disc.")
        #clsf_system.cross_validated_classify(texts, labels)
    
        return learning_model, modelfolder
    
 

    def predict_offline(self, modelfolder, test_instances):
        #global model, task_obj, model_folder
        #if model is None or task_obj is None or model_folder !=modelfolder:
        #    print("model yukleme yaptim")
        #model_folder=modelfolder
        model, task_obj = self._load_classification_system(modelfolder)
        
        predicted_labels, prediction_map, comparasion = task_obj.predict(model, test_instances)
        return predicted_labels, prediction_map, comparasion
    
    
    def predict_online(self, model, test_instances):
        
        return self.clsf_task.predict(model, test_instances)
    
    
    def run_classification_system(self,  clsf_task,
                                     texts,
                                     labels,
                                     modelfolder,
                                     modelname,
                                     cross_val_performance=True):
            
        if cross_val_performance:
            clsf_task.cross_validated_classify(texts, labels)
            #clsf_task.measure_performance(texts, labels)
            
        model, modelpath = self.train_and_save_model(texts, labels, clsf_task, modelfolder, modelname)
        
        return model, modelpath
    
#model=None
#task_obj=None
#model_folder=None       
if __name__ == "__main__":
    
    print(ClassificationSystem())
    
    
        
    