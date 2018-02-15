'''
Created on Sep 21, 2016

@author: dicle
'''

import re
import string

from turkish.deasciifier import Deasciifier
import nltk.tokenize as nltktokenizer
import nltk.stem as stemmer
from polyglot.text import Text



import  tr_stemmer, en_stemmer,stopword_lists

from spellchecker import en_spellchecker 

from abc import ABCMeta, abstractmethod



'''
implements preprocessing utilities for tr & en. can be reproduced for other langs

@todo: 
- handle sentences
- optimize for running time generally
  - english takes a bit long to tokenize
'''


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
    '''
    text : any string in lang
    lang : langauge of the string (english, turkish..) 
    remove_punkt: if true, remove the punctuation tokens
    '''

    # @TODO if lang not found, use english
    tokens = nltktokenizer.word_tokenize(text, language=lang)
    #t = Text(text, hint_language_code="tr")
    #tokens = list(t.tokens)
    
    if remove_punkt:
        tokens = [token for token in tokens if not is_punctuation(token)]
    
    tokens = eliminate_empty_strings(tokens)
    tokens = [token for token in tokens if token.isalnum()]
    return tokens


def stem_words(words, lang):

    if lang in ["tr", "turkish"]:
        words = [tr_stemmer.stem2(word) for word in words]
    
    if lang in ["en", "english"]:
        words = [en_stemmer.stem1(word) for word in words]
    
    return words


def eliminate_empty_strings(wordlist):
    l = [w.strip() for w in wordlist]
    l = [w for w in l if len(w) > 0]
    return l




def language_map(lang_shortcut):
    langmap = { "tr" : "turkish",
                "en" : "english",
                "eng" : "english"
              }
    
    return langmap[lang_shortcut]
  
    

class _TokenHandler():
    __metaclass__ = ABCMeta
    
    
    
    @abstractmethod   
    def __init__(self, language=None, 
                 stopword=False, more_stopwords=None, 
                 stemming=False, 
                 remove_numbers=False, deasciify=False, 
                 remove_punkt=True,
                 lowercase=True):
        self.lang = language
        self.eliminate_stopwords = stopword
        self.more_stopwords = more_stopwords
        self.apply_stemming = stemming
        self.remove_numbers = remove_numbers
        self.deasciify = deasciify
        self.remove_punkt = remove_punkt
        self.lowercase = lowercase
        return
        
        
    @abstractmethod 
    def __call__(self, doc):
        tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt) 
        
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
        
        # problem: "İ" is lowercased to "i̇"
        #i = 'i̇'
        #tokens = [token.replace(i, "i") for token in tokens]        
        
        if self.remove_numbers:
            number_pattern = "[a-zA-z]{,3}\d+"   #d{6,}  
            tokens = [re.sub(number_pattern, "", token) for token in tokens]
        
        if self.eliminate_stopwords:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopwords]  
            
            if self.more_stopwords: #len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
                tokens = [token for token in tokens if token not in self.more_stopwords]
                
        if self.apply_stemming:
            tokens = stem_words(tokens, lang=self.lang)
        
        if self.eliminate_stopwords:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopwords]  
            
            if self.more_stopwords: #len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
                tokens = [token for token in tokens if token not in self.more_stopwords]
        
        
        '''
        if self.apply_stemming:
            tokens = [tr_stemmer.stem2(token) for token in tokens]    
        
        if self.deasciify:
            tokens = [Deasciifier(token).convert_to_turkish() for token in tokens]
        '''
        
        
        tokens = eliminate_empty_strings(tokens)
        return tokens   
    



class TrTokenHandler(_TokenHandler):
    
    def __init__(self,  
                 stopword=False, more_stopwords=None,
                 stemming=False, 
                 remove_numbers=False, deasciify=False, 
                 remove_punkt=True,
                 lowercase=True):
        self.lang = "tr"
        super().__init__(self.lang, stopword, more_stopwords, stemming, remove_numbers, deasciify, remove_punkt, lowercase)   
    
    
    @abstractmethod
    def __call__(self, doc):
        
        #tokens = super.__call__(self, doc)
        tokens = _TokenHandler.__call__(self, doc) 
        
        '''
        if self.apply_stemming:
            tokens = [tr_stemmer.stem2(token) for token in tokens] 
        '''
        if self.deasciify:
            tokens = [Deasciifier(token).convert_to_turkish() for token in tokens]
        
        return tokens

  


class EnTokenHandler(_TokenHandler):
    
    spellcheck = False
    
    def __init__(self, 
                 spellcheck=False,
                 stopword=False, more_stopwords=None, 
                 stemming=False, 
                 remove_numbers=False, deasciify=False, 
                 remove_punkt=True,
                 lowercase=True):
        self.lang = "en"
        super().__init__(self.lang, stopword, more_stopwords, stemming, remove_numbers, deasciify, remove_punkt, lowercase)   
        self.spellcheck = spellcheck
    
    
    
    @abstractmethod
    def __call__(self, doc):
        #tokens = super.__call__(doc)
        tokens = _TokenHandler.__call__(self, doc) 
        
        if self.spellcheck:
            tokens = [en_spellchecker.spellcheck(token) for token in tokens]
        '''
        if self.apply_stemming:
            tokens = [en_stemmer.stem1(token) for token in tokens] 
        '''
        # ascii is english..
            
        
        return tokens
    


    
def original_to_preprocessed_map(preprocessor, text):
    
    words = text.split()
    words_prep = []
    for word in words:
        prepword = preprocessor(word)
        if prepword:
            prepword = prepword[0]
        else:
            prepword = ""
        
        words_prep.append((prepword, word))
    
    prep_word_map = {}
    for x, y in words_prep:
        prep_word_map.setdefault(x, []).append(y)
    
    return prep_word_map

    

if __name__ == "__main__":
    
    preprocessor = TrTokenHandler(stopword=True, stemming=True, 
                                                            remove_numbers=True, 
                                                            deasciify=True, 
                                                            remove_punkt=True
                                                            )
    x = preprocessor.__call__("ŞİMDİ İPTAL IPTAL ŞEHİR şehir sana verdim 46374637 hello")
    
    print(type(x))
    print(x)
    print([len(i) for i in x])
    
    p = EnTokenHandler(spellcheck=True, stopword=True, stemming=True, 
                                                            remove_numbers=True, 
                                                            deasciify=True, 
                                                            remove_punkt=True)
    
    print(p("I might be a little bit late, but this may be helpful. Here are the stop words for some languages (English, French, German, Finish, Hungarian, Turkish, Russian, Czech, Greek, Arabic, Chinese, Japanese, Korean, Catalan, Polish, Hebrew, Norwegian, Swedish, Italian, Portuguese and Spanish): "))   
    
    
    
    
    
        