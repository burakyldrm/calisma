'''
Created on Mar 1, 2017

@author: dicle
'''


import sys
sys.path.append("..")


from sklearn.base import BaseEstimator, TransformerMixin

import sklearn.feature_extraction.dict_vectorizer as dv
import sklearn.pipeline as skpipeline
from modules.sentiment import sentiment_feature_extractors as sf



#########   sentiment lexicon counting   ###################



def lexicon_count_feature_extraction_dict(tokens, lexicon_type):
        
    keys = ["pos_lex", "neg_lex"]
    
    nwords = len(tokens)
    # t = Text(text, hint_language_code=lang)
    # nentities = len(t.entities)
    
    if nwords == 0:
        d = {key : 0.0 for key in keys}
        return d
    
    #print("lexicon type", lexicon_type)
    npos_lex = 0
    nneg_lex = 0
    if lexicon_type in ["tr", "turkish"]:
        # boun lexicon
        npos_lex = sf.get_boun_polarity_count(tokens, "pos")
        nneg_lex = sf.get_boun_polarity_count(tokens, "neg")
    
    if lexicon_type in ["en", "english", "eng"]:
        npos_lex = sf.get_english_polarity_count(tokens, "pos")
        nneg_lex = sf.get_english_polarity_count(tokens, "neg")
    
    if lexicon_type in ["emoticon", "emot"]:
        npos_lex = sf.get_emoticon_polarity_count(tokens, "pos")
        nneg_lex = sf.get_emoticon_polarity_count(tokens, "neg")
    
    if lexicon_type in ["ar", "arab", "arabic"]:
        npos_lex, nneg_lex = sf.get_arabic_polarity_count1(tokens)
        
    
    
    values = [npos_lex, nneg_lex]  # , nentities]
    f = lambda x : round(float(x) / nwords, 5)
    # f = lambda x : round(float(x), 3)
    values = list(map(f, values))    

    
    value_dict = dict()
    for key, value in zip(keys, values):
        value_dict[key] = value

    return value_dict

# # should be called after a tokenizer/preprocessor
class CountBasedTransformer(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
      
    tokenizer = None
    lexicon_choice = ""   # en, tr or emoticon
    def __init__(self, lexicon_choice, tokenizer=None):  # tokenize can be a callable as in sklearn.tfidfvectorizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = lambda text : text.split()
        
        self.lexicon_choice = lexicon_choice
        if len(self.lexicon_choice.strip()) == 0:
            self.language = "en"
        
    def fit(self, X, y=None): 
        
              
        return self
    
    # tokens_list = [[tokens_of_doc_i]]
    def transform(self, X):   
        
        tokens_list = [self.tokenizer(doc) for doc in X]  # [[tokens_of_doc_i]]
        
        return [lexicon_count_feature_extraction_dict(tokens, self.lexicon_choice) for tokens in tokens_list]
   




# @TODO call this in eng and tr feature pipeline builders

def get_lexicon_count_pipeline(tokenizer, lexicontype):
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer(lexicontype, tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe


def get_TR_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer("tr", tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe


def get_EN_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer("en", tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe

def get_EMOT_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer("emoticon", tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe


def single_word_polarity(word, lang="tr"):
    
    label = ""
    lex_results = lexicon_count_feature_extraction_dict([word], lexicon_type=lang)
    for k,v in lex_results.items():
        if v == 1:
            label = k[:3]
            return label
            
    # lexicon_count_feature_extraction_dict() returns {pos_lex: ?, neg_lex= ?}. if the key having value 1 is the label.
    emoticon_results = lexicon_count_feature_extraction_dict([word], lexicon_type="emot")
    for k,v in emoticon_results.items():
        if v == 1:
            label = k[:3]
            return label

    label = "neutral"
    return label

#################################################






