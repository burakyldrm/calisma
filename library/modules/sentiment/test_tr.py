'''
Created on Apr 3, 2017

@author: dicle
'''

import sys
sys.path.append("..")



import os



import sentiment_conf as conf







if __name__ == '__main__':
    
    config_dict = conf.tr_sentiment_params
    feature_params = config_dict[conf.feat_params_key]
    #features_pipeline = _tr_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    
    print("config_dict")
    print(config_dict)
    print("classifier")
    print(classifier)
    
    #return features_pipeline, classifier
    
    