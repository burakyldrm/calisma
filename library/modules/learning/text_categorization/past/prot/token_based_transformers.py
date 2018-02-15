'''
Created on Jan 18, 2017

@author: dicle
'''

import sys
sys.path.append("..")


from sklearn.base import BaseEstimator, TransformerMixin

import sklearn.feature_extraction.dict_vectorizer as dv
import sklearn.pipeline as skpipeline
import text_categorization.sentiment_analysis.sentiment_feature_extractors as sf


#########   sentiment lexicon counting   ###################
def lexicon_based_feature_extraction_dict(tokens):
        
    keys = ["pos_b", "neg_b", "pos_e", "neg_e"]
    
    nwords = len(tokens)
    # t = Text(text, hint_language_code=lang)
    # nentities = len(t.entities)
    
    if nwords == 0:
        d = {key : 0.0 for key in keys}
        return d
    
    # boun lexicon
    npos_boun = sf.get_boun_polarity_count(tokens, "pos")
    nneg_boun = sf.get_boun_polarity_count(tokens, "neg")
    
    
    # emoticon counts
    npos_emot = sf.get_emoticon_polarity_count(tokens, "pos")
    nneg_emot = sf.get_emoticon_polarity_count(tokens, "neg")
    
    values = [npos_boun, nneg_boun, npos_emot, nneg_emot]  # , nentities]
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
    def __init__(self, tokenizer=None):  # tokenize can be a callable as in sklearn.tfidfvectorizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = lambda text : text.split()
        
    def fit(self, X, y=None): 
        
              
        return self
    
    # tokens_list = [[tokens_of_doc_i]]
    def transform(self, X):   
        
        tokens_list = [self.tokenizer(doc) for doc in X]  # [[tokens_of_doc_i]]
        
        return [lexicon_based_feature_extraction_dict(tokens) for tokens in tokens_list]
   



def get_lexicon_count_pipeline(tokenizer):
    
    lexpipe = skpipeline.Pipeline([('lexfeatures', CountBasedTransformer(tokenizer)),
                                   ('lexvect', dv.DictVectorizer()),
                                   ])
    return lexpipe

#################################################






