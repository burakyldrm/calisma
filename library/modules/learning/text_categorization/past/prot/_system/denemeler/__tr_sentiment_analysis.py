'''
Created on May 8, 2017

@author: dicle
'''


import os
from time import time 

import sklearn.pipeline as skpipeline
import sklearn.feature_extraction.text as sktext
import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb

from abc import abstractmethod



import text_categorization.prototypes.system._classification as clsf
import text_categorization.prototypes.system.prep_config as prepconfig
import text_categorization.prototypes.text_preprocessor as prep
import text_categorization.prototypes.text_based_transformers as tbt
import text_categorization.prototypes.token_based_multilang as obt


from _sentiment_analysis import SentimentAnalysis
import _dataset



tr_sent_config = dict(  lang="tr",
                        weights={"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "lexicon_count" : 1,
                           "char_tfidf" : 1}, 
                        stopword=True, 
                        more_stopwords=None, 
                        spellcheck=False,
                        stemming=True,
                        remove_numbers=True,
                        deasciify=True,
                        remove_punkt=True,
                        lowercase=True,
                        wordngramrange=(1,2),
                        charngramrange=(2,2),
                        nmaxfeature=None,
                        norm="l2",
                        use_idf=True,
                        classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),)


def tr_sentiment_analysis(   lang, 
                             weights, 
                             stopword, 
                             more_stopwords, 
                             spellcheck,
                             stemming,
                             remove_numbers,
                             deasciify,
                             remove_punkt,
                             lowercase,
                             wordngramrange,
                             charngramrange,
                             nmaxfeature,
                             norm,
                             use_idf,
                             classifier,
                             train_data_folder,
                             train_data_fname,
                             text_col,
                             cat_col,
                             csvsep,
                             shuffle_dataset,
                             cross_val_performance,
                             modelfolder,
                             modelname):
    
    
    conf_sentiment = prepconfig.FeatureChoice(lang, weights, 
                                              stopword, more_stopwords, 
                                              spellcheck,
                                              stemming,
                                              remove_numbers, deasciify, remove_punkt, lowercase,
                                              wordngramrange, charngramrange,  
                                              nmaxfeature, norm, use_idf)
 
    clsf_task = SentimentAnalysis(feature_config=conf_sentiment,
                                  classifier=classifier,
                                  task_name="TR Sentiment Analysis"
                                  )

    texts, labels = _dataset.read_csv(train_data_folder, train_data_fname, 
                                      csvsep, text_col, cat_col, shuffle_dataset)
    
     
    model, modelpath = clsf.build_classification_system(clsf_task, texts, labels, modelfolder, modelname, cross_val_performance)
    
    return model, modelpath
    
##################################################
# Alternative:


tr_sent_config_obj = prepconfig.FeatureChoice(lang="tr", weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 1,
                                                                   "char_tfidf" : 1}, 
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=True,
                                              remove_numbers=True, deasciify=True, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=(2,2),  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))

en_sent_config_obj = prepconfig.FeatureChoice(lang="en", weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 1,
                                                                   "char_tfidf" : 1}, 
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=True,
                                              remove_numbers=True, deasciify=False, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=(2,2),  
                                              nmaxfeature=None, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))

def sentiment_classification(task_name,
                             sent_config_obj,
                             classifier,
                             train_data_folder,
                             train_data_fname,
                             text_col,
                             cat_col,
                             csvsep,
                             shuffle_dataset,
                             cross_val_performance,
                             modelfolder,
                             modelname):
    

 
    
    clsf_task = SentimentAnalysis(feature_config=tr_sent_config_obj,
                                  classifier=classifier,
                                  task_name=task_name
                                  )

    texts, labels = _dataset.read_csv(train_data_folder, train_data_fname, 
                                      csvsep, text_col, cat_col, shuffle_dataset)
    
     
    model, modelpath = clsf.build_classification_system(clsf_task, texts, labels, modelfolder, modelname, cross_val_performance)
    
    return model, modelpath
    
    





    
    
if __name__ == '__main__':
    
    train_data_folder = "/home/dicle/Documents/data/tr_sentiment/Turkish_Products_Sentiment"
    train_data_fname = "tr_sentiment_product_reviews.csv"
    csvsep="\t"
    text_col="text"
    cat_col="polarity"
    shuffle_dataset = False
    cross_val_performance = True
    modelfolder = "/home/dicle/Documents/experiments/tr_sentiment_detection/models"
    modelname = "product1"
    
    tr_sentiment_analysis(lang=tr_sent_config["lang"], weights=tr_sent_config["weights"], 
                          stopword=tr_sent_config["stopword"], more_stopwords=tr_sent_config["more_stopwords"], 
                          spellcheck=tr_sent_config["spellcheck"], stemming=tr_sent_config["stemming"], 
                          remove_numbers=tr_sent_config["remove_numbers"], deasciify=tr_sent_config["deasciify"], 
                          remove_punkt=tr_sent_config["remove_punkt"], lowercase=tr_sent_config["lowercase"], 
                          wordngramrange=tr_sent_config["wordngramrange"], charngramrange=tr_sent_config["charngramrange"], 
                          nmaxfeature=tr_sent_config["nmaxfeature"], norm=tr_sent_config["norm"], 
                          use_idf=tr_sent_config["use_idf"], 
                          classifier=tr_sent_config["classifier"], 
                          train_data_folder=train_data_folder, train_data_fname=train_data_fname, 
                          text_col=text_col, cat_col=cat_col, csvsep=csvsep, shuffle_dataset=shuffle_dataset, 
                          cross_val_performance=cross_val_performance, 
                          modelfolder=modelfolder, modelname=modelname)
    
    
    
    ####
    # Alternative II
    classifier = sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    tr_sentiment_classifier = ClassificationSystem()
    tr_sentiment_classifier.build_classification_system( task_name="TR Sentiment Classification",
                                                         sent_config_obj=tr_sent_config_obj,
                                                         classifier=classifier,
                                                         train_data_folder=train_data_folder,
                                                         train_data_fname=train_data_fname,
                                                         text_col=text_col,
                                                         cat_col=cat_col,
                                                         csvsep=csvsep,
                                                         shuffle_dataset=shuffle_dataset,
                                                         cross_val_performance=cross_val_performance,
                                                         modelfolder=modelfolder,
                                                         modelname=modelname)
    
    