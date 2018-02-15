'''
Created on Oct 18, 2016

@author: dicle
'''


import re

import nltk.tokenize as tokenizer
import nltk.stem as stemmer
from language_tools import  stopword_lists




# We have to have separate classes per language 
# because not all languages have the same facilities
# (for tr, we don't have stemming tools for now.)


def simple_tokenizer(text):
    return re.split(r"\s+", text)   # r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"


class _TRSimpleTokenHandler(object):
    
    apply_stemming = False
    lang = None
    eliminate_stopwords = False
    
    def __init__(self, stopword):
        self.lang = "tr"
        self.eliminate_stopwords = stopword
        return
    def __call__(self, doc):
        
       
        tokens = [token for token in simple_tokenizer(doc) if token.isalnum() and 
                                                len(token) > 0 and not token.isspace()] # we can eliminate punctuation  as well
        
        if self.eliminate_stopwords:
            stopwords = stopword_lists.get_stopwords(lang="tr")          
            tokens = [token for token in tokens if token not in stopwords]      
        return tokens
    
    

class ENSimpleTokenHandler(object):
    
    apply_stemming = False
    eliminate_stopwords = False
    
    def __init__(self, stopword):
        self.eliminate_stopwords = stopword
        return
    def __call__(self, doc):
        
        tokens = tokenizer.word_tokenize(doc)
        tokens = [token for token in tokens if token.isalnum() and 
                                                len(token) > 0 and not token.isspace()] 
    
        if self.eliminate_stopwords:
            
            stop_words = stopword_lists.get_stopwords(lang="en")        
            tokens = [token for token in tokens if token not in stop_words]
        
        if self.apply_stemming:
            snowball_stemmer = stemmer.SnowballStemmer("english")
            tokens = [snowball_stemmer.stem(token) for token in tokens]
            
        return tokens   
    