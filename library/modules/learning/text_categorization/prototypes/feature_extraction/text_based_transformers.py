'''
Created on Jan 18, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import re

from sklearn.base import BaseEstimator, TransformerMixin

import sklearn.feature_extraction.dict_vectorizer as dv
import sklearn.pipeline as skpipeline

from polyglot.text import Text

import modules.learning.text_categorization.prototypes.feature_extraction.sentiment_feature_extractors as sf

######## reverse text for arabic   ############






########  sentiment   #################
class PolyglotPolarityCountTransformer(BaseEstimator, TransformerMixin):
    
    lang = ""
    def __init__(self, lang):
        self.lang = lang
        
    
    def fit(self, X, y):
        
        return self
    
    def transform(self, rawtexts):
        
        return [self.polyglot_polarity_feat_dict(text) for text in rawtexts]  # @TODO can counting improve?
 


    def polyglot_polarity_feat_dict(self, text):
        
        npos, nneg = sf.get_polyglot_polarity_count(text, lang=self.lang)
        d = dict(
            polyglot_nneg=nneg,
            polyglot_npos=npos
            # polyglot_polarity_val = sf.get_polyglot_doc_polarity(text, lang=self.lang)
            )
        return d



class PolyglotPolarityValueTransformer(BaseEstimator, TransformerMixin):
    
    lang = ""
    def __init__(self, lang):
        self.lang = lang
        
    
    def fit(self, X, y):
        
        return self
    
    def transform(self, rawtexts):
        
        return [self.polyglot_polarity_feat_dict(text) for text in rawtexts]  # @TODO can counting improve?
 


    def polyglot_polarity_feat_dict(self, text):
        
        d = dict(
            polyglot_polarity_val=sf.get_polyglot_doc_polarity(text, lang=self.lang)
            )
        return d




def get_polylglot_polarity_value_pipe(lang):
    
    ptransformer = PolyglotPolarityValueTransformer(lang)
    tvect = dv.DictVectorizer()
    polaritypipe = skpipeline.Pipeline([('polyglotpolarityvfeat', ptransformer),
                                        ('polyglotpolarityvvect', tvect),
                                       ])

    return polaritypipe


def get_polylglot_polarity_count_pipe(lang):
    
    ptransformer = PolyglotPolarityCountTransformer(lang)
    tvect = dv.DictVectorizer()
    polaritypipe = skpipeline.Pipeline([('polyglotpolaritycfeat', ptransformer),
                                        ('polyglotpolaritycvect', tvect),
                                       ])

    return polaritypipe

#########################################







##### drop empty / na instances   #########

class DropNATransformer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self):
        return
        
    
    def fit(self, X, y):
        
       
        indices = [i for i, doc in enumerate(X) if len(doc) > 0]  # non-empty row indices
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]
        return self
    
    
    def transform(self, rawtexts):
        
        return rawtexts
 



##########





######  keyword presence    ##############
# we can change this later to search the keyword in tokens instead of the whole text where we apply regex search.

# checks if some given term exists in each text
class TermPresenceTransformer(BaseEstimator, TransformerMixin):
    word = ""
    def __init__(self, word):
        self.word = word
        
    
    def fit(self, X, y):
        
        return self
    
    def transform(self, rawtexts):
        
        return [self.word_presence_feat_dict(text) for text in rawtexts]  # @TODO can counting improve?
 


    def word_presence_feat_dict(self, text):
        
        # @TODO hızlandır. search in word lists - from the preprocessed
        val = len(re.findall(self.word, text, re.IGNORECASE)) > 0
        d = {"has_" + self.word : val}
        return d



def get_keyword_pipeline(word):
    
    ttransformer = TermPresenceTransformer(word)
    tvect = dv.DictVectorizer()
    wordpipe = skpipeline.Pipeline([('wordpresfeat', ttransformer),
                                    ('wordpresvect', tvect),
                                    ])
    return wordpipe




#########################################


'''
stylistic features
'''


# number of named entities
class NNEsTransformer(BaseEstimator, TransformerMixin):

    language = ""
    a = 5
    def __init__(self, lang, x=10):
        self.language = lang
        print("#", lang, self.language)
        self.a = x
        
    def fit(self, X, y):
        
        return self
    
    def transform(self, rawtexts):
        
        return [self.num_of_NEs_dict(text) for text in rawtexts]  # @TODO can counting improve?
 


    def num_of_NEs_dict(self, text):
        
        # should be weighted by number of words
        
        t = Text(text, hint_language_code=self.language)
        print("*", self.language, len(t.tokens), " * ", self.a)
        nnes = len(t.entities) 
        nwords = len(text.split())
        val = float(nnes) / nwords if nwords > 0 else 0.0       
        d = {"named_entity_weight": val}
        return d


def get_named_entity_weight_pipeline(language):
    
    ntransformer = NNEsTransformer(language)
    nvect = dv.DictVectorizer()
    ne_rate_pipe = skpipeline.Pipeline([('neratefeat', ntransformer),
                                    ('neratevect', nvect),
                                    ])
    return ne_rate_pipe





