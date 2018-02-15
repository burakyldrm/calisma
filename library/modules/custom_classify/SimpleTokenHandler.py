# encoding: utf-8

'''
Created on Sep 21, 2016

@author: dicle
'''

import re
import string

from turkish.deasciifier import Deasciifier
import nltk.tokenize as tokenizer
import nltk.stem as stemmer
from polyglot.text import Text
from modules.custom_classify.language_tools import tr_stemmer
from modules.custom_classify import  stopword_lists

# We have to have separate classes per language 
# because not all languages have the same facilities
# (for tr, we don't have stemming tools for now.)

'''
def simple_tokenizer(text):
    return re.split(r"\s+", text)   # r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"
'''
# checks if a word has only punctuation char.s
def is_punctuation(word):
    punkts = string.punctuation
    tested_chars = [i for i in word if i in punkts]
    return len(word) == len(tested_chars)

def tokenizer(text, lang, remove_punkt=True):

    t = Text(text, hint_language_code=lang)
    tokens = list(t.tokens)

    if remove_punkt:
        tokens = [token for token in tokens if not is_punctuation(token)]
    
    tokens = [token.strip() for token in tokens]
    tokens = [token for token in tokens if len(token) > 0]
    return tokens

'''
class _TRSimpleTokenHandler(object):
    
    #apply_stemming = False
    lang = None
    eliminate_stopwords = False
    apply_stemming = False
    
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
    
'''



class TRSimpleTokenHandler(object):
    
    #apply_stemming = False
    lang = None
    eliminate_stopwords = False
    apply_stemming = False
    remove_numbers = False
    deasciify = False
    remove_punkt = False
    
    def __init__(self, stopword, stemming, remove_numbers=True, deasciify=True, remove_punkt=True):
        self.lang = "tr"
        self.eliminate_stopwords = stopword
        self.apply_stemming = stemming
        self.remove_numbers = remove_numbers
        self.deasciify = deasciify
        self.remove_punkt = remove_punkt
        return
    def __call__(self, doc):
        
        if not doc :
            return []
        else:
            tokens = [token for token in tokenizer(doc, self.lang, True) if token.isalnum() and 
                                                    len(token) > 0 and not token.isspace()] # we can eliminate punctuation  as well
            tokens = [token.lower() for token in tokens]
            
            if self.remove_numbers:
                number_pattern = "[a-zA-z]{,3}\d{6,}"
                tokens = [re.sub(number_pattern, "", token) for token in tokens]
            
            if self.eliminate_stopwords:
                stopwords = stopword_lists.get_stopwords(lang="tr")          
                tokens = [token for token in tokens if token not in stopwords]  
            
            if self.apply_stemming:
                tokens = [tr_stemmer.stem2(token) for token in tokens]    
            
            if self.deasciify:
                tokens = [Deasciifier(token).convert_to_turkish() for token in tokens]
            
            tokens = [token.strip() for token in tokens]
            tokens = [token for token in tokens if len(token) > 0] # or not token.isspace()]
            return tokens
    
        

class ENSimpleTokenHandler(object):
    
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
        tokens = [token.lower() for token in tokens]
        
        if self.eliminate_stopwords:
            
            stop_words = stopword_lists.get_stopwords(lang="en")        
            tokens = [token for token in tokens if token not in stop_words]
        
        if self.apply_stemming:
            snowball_stemmer = stemmer.SnowballStemmer("english")
            tokens = [snowball_stemmer.stem(token) for token in tokens]
            
        return tokens   
    
if __name__ == "__main__":
    
    print("Hellooooooooooooooooooooooooooooooo")
    