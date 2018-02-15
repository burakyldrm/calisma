'''
Created on Oct 11, 2016

@author: dicle
'''



import re

import nltk.tokenize as tokenizer
import nltk.stem as stemmer
from nltk.corpus import stopwords

# Only for English
# @todo improve for tools

def simple_tokenizer(text):
    return re.split(r"\s+", text)   # r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"

class _ENSimpleTokenHandler(object):
    
    apply_stemming = False
    eliminate_stopwords = False
    
    def __init__(self, stem, stopword):
        self.apply_stemming = stem
        self.eliminate_stopwords = stopword
        return
    def __call__(self, doc):
        
        tokens = tokenizer.word_tokenize(doc)
        tokens = [token for token in tokens if token.isalnum() and 
                                                len(token) > 0 and not token.isspace()] 
    
        if self.eliminate_stopwords:
            
            stop_words = stopwords.words("english")          
            tokens = [token for token in tokens if token not in stop_words]
        
        if self.apply_stemming:
            snowball_stemmer = stemmer.SnowballStemmer("english")
            tokens = [snowball_stemmer.stem(token) for token in tokens]
            
        return tokens
        
        
        