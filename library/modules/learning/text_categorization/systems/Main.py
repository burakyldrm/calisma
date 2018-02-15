'''
Created on May 12, 2017

@author: dicle
'''

import sys
sys.path.append("..")

from modules.learning.text_categorization.utils import tc_utils


def main(clsf_system, 
         texts, labels,
         modelrootfolder, modelname,
         test_instances, test_labels):
    
 
        
    # 2- APPLY CROSS-VALIDATED CLASSIFICATION and GET PERFORMANCE
    accuracy, fscore, duration = clsf_system.get_cross_validated_clsf_performance(texts, labels, nfolds=3)
    
    
    # 3- TRAIN THE MODEL WITH THE ABOVE PARAMETERS; SAVE IT ON THE FOLDER modelrootfolder/modelname
    model, modelfolder = clsf_system.train_and_save_model(texts, labels, modelrootfolder, modelname)
    '''
    import os
    modelfolder = os.path.join(modelrootfolder, modelname)
    '''
    
    # 4.a- PREDICT ONLINE (the model is in the memory)
    predicted_labels, prediction_map = clsf_system.predict_online(model, test_instances)
    
    # 4.b- PREDICT OFFLINE (the model is loaded from the disc)
    predicted_labels, prediction_map = clsf_system.predict_offline(modelfolder, test_instances)
    
    print(prediction_map)
    
    # 4.c- GET PREDICTION PERFORMANCE IF TRUE LABELS ARE AVAILABLE
    if test_labels:
        test_acc, test_fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
    # -- if paramaters are modified (tr_sent_config_obj), create a new ClassificationSystem() and run necessary methods.
    
    