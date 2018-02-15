'''
Created on Mar 15, 2017

@author: dicle
'''

import os

import pandas as pd

import SENTIMENT_CONF as conf
import text_categorization.prototypes.classification as clf
from text_categorization.sentiment_analysis import ar_sentiment_classification

from dataset import corpus_io, io_utils

def run_pipeline_choices(confs, instances, labels, pipeline_generator, recordpath):  #, instances, labels):


    result_rows = []
    for config in confs:
            
        # merge nested dicts in one.
        import copy
        config_copy = copy.deepcopy(config)
        #config_copy = config.copy()
        features = config_copy.pop(conf.feat_params_key, None)
        
        prep = features.pop(conf.prep_key, None)
        
        features.update(prep)  # join all the feature keys in one dict.
        
        config_copy.update(features)   # join classifier + feature keys in one dict.
        
        
        feature_params = config[conf.feat_params_key]
        features_pipeline = pipeline_generator(feature_params)
        
        classifier = config[conf.classifier_key]
        
        email_classifier = clf.TextClassifier(feature_pipelines=features_pipeline,
                                          classifier=classifier)

        acc, fscore, duration = email_classifier.cross_validated_classify(instances, labels)
        # @TODO get precision and recall
        config_copy["accuracy"] = acc
        config_copy["f1-score"] = fscore
        config_copy["duration"] = duration
        
        result_rows.append(config_copy)
        
        
        
    results = pd.DataFrame(result_rows)  
    
    if recordpath:
        results.to_csv(recordpath, sep="\t", index=False)
    
    return results



def ar_generate_and_run_grid():
    
    # twitter
    
    datasetname = "twitter2"
    fpath = "/home/dicle/Documents/arabic_nlp/datasets/sentiment/Twitter/ar_2000polartweets.csv"
    
    
    # OCA
    '''
    datasetname = "OCA"
    fname = "ar_polar-moviereviewsOCAcorpus.csv"
    infolder = "/home/dicle/Documents/arabic_nlp/datasets/OCA-corpus"
    fpath = os.path.join(infolder, fname)
    '''
    
    textcol = "text"
    catcol = "polarity"
    sep = "\t"
    instances, labels = corpus_io.read_labelled_texts_csv(fpath, sep, textcol, catcol)
    
    mainrecordpath = "/home/dicle/Documents/experiments/ar_sentiment/grid"
    recordpath = os.path.join(mainrecordpath, "grid_"+datasetname+".csv")
    
    param_choices = ar_sentiment_classification.grid_generate_confs()
    
    run_pipeline_choices(param_choices, instances, labels, 
                         pipeline_generator=ar_sentiment_classification._ar_sentiment_features_pipeline2,
                         recordpath=recordpath)


if __name__ == '__main__':
    

    ar_generate_and_run_grid()
    
    
    
