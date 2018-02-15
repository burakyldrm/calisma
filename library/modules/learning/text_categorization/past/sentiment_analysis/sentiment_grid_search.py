'''
Created on Feb 1, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import os
import sklearn.naive_bayes as nb
import sklearn.linear_model as sklinear
import sklearn.neighbors as knn

import pandas as pd

from misc import list_utils
import SENTIMENT_CONF as conf


{'classifier': nb.MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
 'feature_params': {'lang': 'tr',
  'prep_params': {'case_choice': True,
   'charngramrange': (2, 2),
   'deasc_choice': True,
   'more_stopwords_list': None,
   'nmaxfeature': 10000,
   'norm': 'l2',
   'number_choice': False,
   'punct_choice': True,
   'spellcheck_choice': False,
   'stemming_choice': False,
   'stopword_choice': True,
   'use_idf': True,
   'wordngramrange': (1, 2)},
  'weights': {'char_tfidf': 1,
   'lexicon_count': 0,
   'polyglot_count': 0,
   'polyglot_value': 0,
   'word_tfidf': 1}}}



def generate_tr_sentiment_grid():
    
    weights= { 'char_tfidf': 1,
               'lexicon_count': 0,
               'polyglot_count': 0,
               'polyglot_value': 0,
               'word_tfidf': 1}
    
    possible_weights = { 'char_tfidf': [0, 1],
               'lexicon_count': [0, 1],
               'polyglot_count': [0, 1],
               'polyglot_value': [0, 1],
               'word_tfidf': [0, 1]}
    
    weight_choices = list_utils.multiplex_dict(possible_weights)
    
    
    
    possible_preps = {
        conf.stopword_key : (True, False),
        conf.more_stopwords_key : (None,),
        conf.spellcheck_key : (True, False),
        conf.stemming_key : (True, False),
        conf.remove_numbers_key : (True, False),
        conf.deasciify_key : (True, False),
        conf.remove_punkt_key : (True, False),
        conf.lowercase_key : (True, False),
        conf.wordngramrange_key : [(1, 2), (1, 1), (1, 3)],
        conf.charngramrange_key : [(1, 2), (2, 2)],   # this should be separate
        conf.nmaxfeature_key : [None, 10000],
        conf.norm_key : ("l2",),
        conf.use_idf_key : (True,), 
        }
    
    prep_choices = list_utils.multiplex_dict(possible_preps)
    
    

if __name__ == '__main__':
    