'''
Created on Mar 10, 2017

@author: dicle
'''


import sys
from text_categorization.sentiment_analysis import ar_sentiment_classification
sys.path.append("..")

from django_docker.learning.text_categorization.sentiment_analysis import arabic_datasets


import os
from time import time
import random
import pandas as pd

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline





from text_categorization.prototypes.classification import TextClassifier
import text_categorization.prototypes.text_preprocessor as prep
import text_categorization.prototypes.text_based_transformers as tbt
#import text_categorization.prototypes.token_based_transformers as obt
import text_categorization.prototypes.token_based_multilang as obt

import SENTIMENT_CONF as conf


from dataset import corpus_io, io_utils
from misc import list_utils, table_utils



def run_ar_sentiment_analyser2(instances,
                              labels,
                              config_dict=conf.ar_sentiment_params,
                              picklefolder="/home/dicle/Documents/karalama",
                              modelname="ar_sentiment1"):
    
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = ar_sentiment_classification._ar_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    analyser = TextClassifier(features_pipeline, classifier)
    
    analyser.cross_validated_classify(instances, labels)



def run_for_datasets():
    
    folder = "/home/dicle/Documents/arabic_nlp/datasets/sentiment/MASC Corpus/MASC Corpus/Excel Format"
    #fnames = ["MSAC corpus- Political.xlsx"]
    fnames = io_utils.getfilenames_of_dir(folder, removeextension=False)
    textcol = "Text"
    catcol = "Polarity"
    
    config_dict=conf.ar_sentiment_params
    
    picklefolder = "/home/dicle/Documents/experiments/ar_sentiment/models"
    
    for fname in fnames:
        
        p = os.path.join(folder, fname)
        
        df = pd.read_excel(p)
        
        
        
        instances = df[textcol].tolist()
        labels = df[catcol].tolist()
        instances, labels = corpus_io.shuffle_dataset(instances, labels)
        
        modelname = ".".join(fname.split(".")[:-1])
        
        print("\n\n")
        print("Classify ", modelname)
        
        g = df.groupby(by=[catcol])
        print("Category counts:\n ", g.count()[textcol])
        
        run_ar_sentiment_analyser2(instances, labels, config_dict, picklefolder, modelname)
    
    
    all_instances = []
    all_labels = []
    for fname in fnames:
        
        p = os.path.join(folder, fname)
        
        df = pd.read_excel(p)
                
        instances = df[textcol].tolist()
        labels = df[catcol].tolist()
        all_instances.extend(instances)
        all_labels.extend(labels)
    
    all_instances, all_labels = corpus_io.shuffle_dataset(all_instances, all_labels)
    
    print("Classify ALL")
    modelname = "3sets"
    run_ar_sentiment_analyser2(all_instances, all_labels, config_dict, picklefolder, modelname)



def grid_generate_confs():
    
    
    language = "ar"
    weight_params = {"word_tfidf" : 1,
                               "polyglot_value" : 0,
                               "polyglot_count" : 0,
                               "polarity_lexicon_count" : 0,
                               "emoticon_count" : 0,
                               "char_tfidf" : 1,
                               "named_entity_rate" : 0}
      
    ar_possible_prep_values = {
        conf.stopword_key : (True,),
        conf.more_stopwords_key : (None,),
        conf.spellcheck_key : (False,),
        conf.stemming_key : (True, False),
        conf.remove_numbers_key : (True,),
        conf.deasciify_key : (False,),
        conf.remove_punkt_key : (True,),
        conf.lowercase_key : (False,),
        
        conf.wordngramrange_key : ((1, 2),),
        conf.charngramrange_key : ((2, 2),),
        conf.nmaxfeature_key : (None, 10000,),
        conf.norm_key : ("l2",),
        conf.use_idf_key : (True,),
    }
    
    prep_choices = list_utils.multiplex_dict(ar_possible_prep_values)
    
    classifiers = [nb.MultinomialNB(),
                   sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
                   ]
    
    
    
    param_choices = []
    for classifier in classifiers:
        
        for prep_choice in prep_choices:

            ar_sentiment_params = {     
                conf.feat_params_key : {
                    conf.lang_key : language,
                    conf.weights_key : weight_params,
                    conf.prep_key : prep_choice,
                    },
                conf.classifier_key : classifier,
                }
            param_choices.append(ar_sentiment_params)
    
    return param_choices




def run_tweets():
    
    tweetspath = '/home/dicle/Documents/arabic_nlp/datasets/twitter_sentiment/ASTD-master/10K_polar-tweets.csv'
    cat_col = "label"
    text_col = "text"
    #tweets, labels, _ = arabic_datasets.read_tweets(tweetspath)
    _,_, df = arabic_datasets.read_tweets(tweetspath)
    cats = ["NEG", "POS"]   #, "NEUTRAL"]
    df = df.loc[df[cat_col].isin(cats), :]
    
    g = df.groupby(by=["label"])
    print("Category counts:\n ", g.count())
    
    
    df = table_utils.category_balance3(df, text_col, cat_col)
    g = df.groupby(by=["label"])
    print("Balanced - Category counts:\n ", g.count())
    
    tweets = df[text_col].tolist()
    labels = df[cat_col].tolist()
    
    run_ar_sentiment_analyser2(tweets, labels)
    
    
    
def run_oca():
    
    ocapath = '/home/dicle/Documents/arabic_nlp/datasets/OCA-corpus/ar_polar-moviereviewsOCAcorpus.csv'
    cat_col = "polarity"
    text_col = "text"
    sep= "\t"
    #tweets, labels, _ = arabic_datasets.read_tweets(tweetspath)
    reviews, labels = corpus_io.read_labelled_texts_csv(ocapath, sep, text_col, cat_col, shuffle=True)
    
    
    run_ar_sentiment_analyser2(reviews, labels)



def run_labr():
    
    labrpath = '/home/dicle/Documents/arabic_nlp/datasets/book_reviews_sentiment/LABR/ar_63K_book-reviews.tsv'
    cat_col = "label"
    text_col = "text"
    #tweets, labels, _ = arabic_datasets.read_tweets(tweetspath)
    _,_, df = arabic_datasets.read_tweets(labrpath)
    
    #cats = ["NEG", "POS"]   #, "NEUTRAL"]
    #df = df.loc[df[cat_col].isin(cats), :]
    
    g = df.groupby(by=["label"])
    print("Category counts:\n ", g.count())
    
    
    df = table_utils.category_balance3(df, text_col, cat_col)
    g = df.groupby(by=["label"])
    print("Balanced - Category counts:\n ", g.count())
    
    reviews = df[text_col].tolist()
    labels = df[cat_col].tolist()
    
    run_ar_sentiment_analyser2(reviews, labels)
    



    
if __name__ == '__main__':
    
    
    # tweets
    #run_tweets()
    #run_oca()
    #run_labr()
    
    '''
    l = grid_generate_confs()
    for i in l:
        print(i)
    '''
    run_for_datasets()
    
    
    '''
    #fpath = "/home/dicle/Documents/arabic_nlp/datasets/sentiment/MASC Corpus/MASC Corpus/MSAC corpus- Political.csv"
    fpath = "/home/dicle/Documents/arabic_nlp/datasets/sentiment/Twitter/ar_500polartweets.csv"
    textcol = "text"
    catcol = "polarity"
    sep = "\t"
    instances, labels = corpus_io.read_labelled_texts_csv(fpath, sep, textcol, catcol)
    run_ar_sentiment_analyser2(instances, labels)
    print()
    ''' 
    
    
    