'''
Created on May 3, 2017

@author: dicle
'''

import text_categorization.prototypes.classification as clsf


if __name__ == '__main__':
    
    
    model_folder = ""
    
    test_instances = []
    test_labels = []
    
    model, analyser = clsf.load_classification_system(model_folder)
    
    ypred, prediction_map = analyser.predict(model, test_instances)
    
    if test_labels:
        print("Testing performance:")
        analyser.get_performance(test_labels, ypred, verbose=True)
        
        
        
        
        