'''
Created on Jan 18, 2017

@author: dicle
'''



import sys
sys.path.append("..")



# from language_tools import  stopword_lists, tr_stemmer, en_stemmer
# from language_tools.spellchecker import en_spellchecker 
'''
def simple_tokenizer(text):
    return re.split(r"\s+", text)   # r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"
'''
# checks if a word has only punctuation char.s

import string, re
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from turkish.deasciifier import Deasciifier

from modules.learning.language_tools import stopword_lists, tr_stemmer, en_stemmer, ar_stemmer
from modules.learning.language_tools.spellchecker import en_spellchecker 
import nltk.tokenize as nltktokenizer


import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.pipeline as skpipeline


from modules.learning.text_categorization.utils import tc_utils
from modules.learning.dataset import corpus_io
import modules.learning.text_categorization.prototypes.feature_extraction.text_based_transformers as tbt
import modules.learning.text_categorization.prototypes.feature_extraction.token_based_transformers as obt





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
    #tokens = nltktokenizer.word_tokenize(text, language=lang)
    tokens = nltktokenizer.wordpunct_tokenize(text)
    # t = Text(text, hint_language_code="tr")
    # tokens = list(t.tokens)
    
    if remove_punkt:
        tokens = [token for token in tokens if not is_punctuation(token)]
    
    tokens = eliminate_empty_strings(tokens)
    if lang not in ["ar"]:
        tokens = [token for token in tokens if token.isalnum()]    # this is already eliminating the punctuation!
    return tokens


def stem_words(words, lang):
    roots = []
    if lang in ["tr", "turkish"]:
        roots = tr_stemmer.stem_words(words)
    
    if lang in ["en", "english"]:
        roots = [en_stemmer.stem1(word) for word in words]
    
    if lang in ["ar", "arabic", "arab"]:
        roots = [ar_stemmer.stem(word) for word in words]
    return roots

def deasciify_words(words, lang):
    
    if lang in ["tr", "turkish"]:
        return [Deasciifier(token).convert_to_turkish() for token in words]
    else:
        return words
    '''
    if lang in ["en", "english", "ar", "arab", "arabic"]:  # not applicable for english
        return words
    '''

def spellcheck_words(words, lang):
    
    if lang in ["en", "english"]:
        return [en_spellchecker.spellcheck(token) for token in words]
    if lang in ["tr", "turkish"]:  # not yet for turkish
        return words
        


def eliminate_empty_strings(wordlist):
    l = [w.strip() for w in wordlist]
    l = [w for w in l if len(w) > 0]
    return l




def language_map(lang_shortcut):
    langmap = { "tr" : "turkish",
                "en" : "english",
                "eng" : "english",
                "ar" : "arabic",
                "arab" : "arabic"
              }
    
    return langmap[lang_shortcut]



    
# returns {"paket" : ["paketlerimiz, paketlerimizde"], "bul" : ["bulunuyor"]..}
# for a sentence like 'paketlerimiz paketlerimizde bulunuyor'
def original_to_preprocessed_map(preprocessor, text):
    
    words = text.split()
    words_prep = []
    for word in words:
        prepword = preprocessor.tokenize(word)
        if prepword:
            prepword = prepword[0]
        else:
            prepword = ""
        
        words_prep.append((prepword, word))
    
    prep_word_map = {}
    for x, y in words_prep:
        prep_word_map.setdefault(x, []).append(y)
    
    return prep_word_map




class Preprocessor2(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using tokenization and
    other normalization and filtering techniques.
    """
    lang = ""
    stopword = ""
    spellcheck = ""
    stemming = ""
    remove_numbers = ""
    deasciify = ""
    remove_punkt = ""
    lowercase = ""
    def __init__(self, 
                 lang,
                 stopword=False, 
                 spellcheck=False,
                 stemming=False,
                 remove_numbers=False,
                 deasciify=False,
                 remove_punkt=True,
                 lowercase=True):
        
        
        self.lang = lang
        self.stopword = stopword
        self.spellcheck = spellcheck
        self.stemming = stemming
        self.remove_numbers = remove_numbers
        self.deasciify = deasciify
        self.remove_punkt = remove_punkt
        self.lowercase = lowercase
        


    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        return X
    


#===============================================================================
# 
# ### stemming fixed for faster results ####
# 
# class Preprocessor(BaseEstimator, TransformerMixin):
#     """
#     Transforms input data by using tokenization and
#     other normalization and filtering techniques.
#     """
#     def __init__(self, lang,
#                  stopword=True, more_stopwords=None,
#                  spellcheck=False,
#                  stemming=False,
#                  remove_numbers=False,
#                  deasciify=False,
#                  remove_punkt=True,
#                  lowercase=True):
#         # these paramater names should be same as the class variable names (like lang -> lang). otherwise, the param. values cannot be transferred to the sk.pipeline.
#         self.lang = lang
#         self.stopword = stopword
#         self.more_stopwords = more_stopwords
#         self.spellcheck = spellcheck
#         self.stemming = stemming
#         self.remove_numbers = remove_numbers
#         self.deasciify = deasciify
#         self.remove_punkt = remove_punkt
#         self.lowercase = lowercase
# 
#         
# 
# 
#     def fit(self, X, y=None):
#         """
#         Fit simply returns self, no other information is needed.
#         """
#         
#         return self
# 
#     def inverse_transform(self, X):
#         """
#         No inverse transformation
#         """
#         return X
# 
#     def transform(self, X):
#         """
#         Actually runs the preprocessing on each document.
#         """
#         
#         
#         
#         return [self.tokenize(doc) for doc in X]
#     
#  
#        
#    
#     def tokenize(self, doc):
#         tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt) 
#                        
#         if self.lowercase and self.lang not in ["ar", "arab", "arabic"]:
#             tokens = [token.lower() for token in tokens]
#         
#         # problem: "İ" is lowercased to "i̇"
#         # i = 'i̇'
#         # tokens = [token.replace(i, "i") for token in tokens]        
#         
#         if self.deasciify:
#             tokens = deasciify_words(tokens, self.lang)
#             
#         if self.spellcheck:
#             tokens = spellcheck_words(tokens, self.lang)
#         if self.remove_numbers:
#             number_pattern = "[a-zA-z]{,3}\d+"  # d{6,}  # TODO real numbers & rational numbers
#             tokens = [re.sub(number_pattern, "", token) for token in tokens]
#         
#         if self.stopword:
#             stopwords = stopword_lists.get_stopwords(lang=self.lang)          
#             tokens = [token for token in tokens if token not in stopwords]  
#         if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
#             tokens = [token for token in tokens if token not in self.more_stopwords]
#         
#         if self.stemming:
#             tokens = stem_words(tokens, lang=self.lang)
# 
#         '''
#         if self.stopword:
#             stopwords = stopword_lists.get_stopwords(lang=self.lang)                      
#             tokens = [token for token in tokens if token not in stopwords]  
#               
#         if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
#             tokens = [token for token in tokens if token not in self.more_stopwords]
#         '''
#             
#         tokens = eliminate_empty_strings(tokens)
#       
#         #print(doc,"  -> ", tokens)
#         
#         return tokens   
#===============================================================================




class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using tokenization and
    other normalization and filtering techniques.
    """
    def __init__(self, lang,
                 stopword=True, more_stopwords=None,
                 spellcheck=False,
                 stemming=False,
                 remove_numbers=False,
                 deasciify=False,
                 remove_punkt=True,
                 lowercase=True):
        # these paramater names should be same as the class variable names (like lang -> lang). otherwise, the param. values cannot be transferred to the sk.pipeline.
        self.lang = lang
        self.stopword = stopword
        self.more_stopwords = more_stopwords
        self.spellcheck = spellcheck
        self.stemming = stemming
        self.remove_numbers = remove_numbers
        self.deasciify = deasciify
        self.remove_punkt = remove_punkt
        self.lowercase = lowercase

        self.notfound = 0
        self.surface_root_pairs = {}   # {word : root} for faster stemming, we stem all the words in all docs once - not to call tr_morph via os multiple times; then, retrieve the root form from this dict upon tokenizing - if stemming is True.
         
        

    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, X):
        """
        Actually runs the preprocessing on each document.
        """
        
        if self.stemming:
            words = []
            for doc in X:
                doc_tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt)
                words.extend(doc_tokens)
            
            if self.lowercase and self.lang not in ["ar", "arab", "arabic"]:
                words = [token.lower() for token in words]
        
            if self.deasciify:
                words = deasciify_words(words, self.lang)
                
            if self.spellcheck:
                words = spellcheck_words(words, self.lang)
            
            roots = stem_words(words, lang=self.lang)
            l = [(word, root) for word, root in zip(words, roots)]
            l.append(("", ""))
            self.surface_root_pairs = dict(l)
        
        return [self.tokenize(doc) for doc in X]
    
 
       
   
    def tokenize(self, doc):
        tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt) 
                       
        if self.lowercase and self.lang not in ["ar", "arab", "arabic"]:
            tokens = [token.lower() for token in tokens]
        
        # problem: "İ" is lowercased to "i̇"
        # i = 'i̇'
        # tokens = [token.replace(i, "i") for token in tokens]        
        
        if self.deasciify:
            tokens = deasciify_words(tokens, self.lang)
            
        if self.spellcheck:
            tokens = spellcheck_words(tokens, self.lang)
        if self.remove_numbers:
            number_pattern = "[a-zA-z]{,3}\d+"  # d{6,}  # TODO real numbers & rational numbers
            tokens = [re.sub(number_pattern, "", token) for token in tokens]
        
        if self.stopword:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopwords]  
        if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
            tokens = [token for token in tokens if token not in self.more_stopwords]
        
        if self.stemming:
            #tokens = stem_words(tokens, lang=self.lang)
            tokens_ = []
            for word in tokens:
                try:
                    root = self.surface_root_pairs[word]
                    #print(0, root, "  ", word)
                except KeyError:
                    self.notfound += 1
                    #print(self.notfound, "  ", word)
                    root = tr_stemmer.stem(word)
                    self.surface_root_pairs[word] = root
                finally:
                    tokens_.append(root)
            tokens = tokens_   
            #tokens = [self.surface_root_pairs[word] for word in tokens]

        '''
        if self.stopword:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)                      
            tokens = [token for token in tokens if token not in stopwords]  
              
        if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
            tokens = [token for token in tokens if token not in self.more_stopwords]
        '''
            
        tokens = eliminate_empty_strings(tokens)
      
        #print(doc,"  -> ", tokens)
        
        return tokens   







'''
before arabic
'''
#===============================================================================
# 
# def is_punctuation(word):
#     punkts = string.punctuation
#     tested_chars = [i for i in word if i in punkts]
#     return len(word) == len(tested_chars)
# 
# 
# def tokenizer(text, lang, remove_punkt=True):
#     '''
#     text : any string in lang
#     lang : langauge of the string (english, turkish..) 
#     remove_punkt: if true, remove the punctuation tokens
#     '''
# 
#     # @TODO if lang not found, use english
#     tokens = nltktokenizer.word_tokenize(text, language=lang)
#     # t = Text(text, hint_language_code="tr")
#     # tokens = list(t.tokens)
#     
#     if remove_punkt:
#         tokens = [token for token in tokens if not is_punctuation(token)]
#     
#     tokens = eliminate_empty_strings(tokens)
#     tokens = [token for token in tokens if token.isalnum()]
#     return tokens
# 
# 
# def stem_words(words, lang):
# 
#     if lang in ["tr", "turkish"]:
#         words = [tr_stemmer.stem2(word) for word in words]
#     
#     if lang in ["en", "english"]:
#         words = [en_stemmer.stem1(word) for word in words]
#     
#     return words
# 
# def deasciify_words(words, lang):
#     
#     if lang in ["tr", "turkish"]:
#         return [Deasciifier(token).convert_to_turkish() for token in words]
#     if lang in ["en", "english"]:  # not applicable for english
#         return words
# 
# 
# def spellcheck_words(words, lang):
#     
#     if lang in ["en", "english"]:
#         return [en_spellchecker.spellcheck(token) for token in words]
#     if lang in ["tr", "turkish"]:  # not yet for turkish
#         return words
#         
# 
# 
# def eliminate_empty_strings(wordlist):
#     l = [w.strip() for w in wordlist]
#     l = [w for w in l if len(w) > 0]
#     return l
# 
# 
# 
# 
# def language_map(lang_shortcut):
#     langmap = { "tr" : "turkish",
#                 "en" : "english",
#                 "eng" : "english"
#               }
#     
#     return langmap[lang_shortcut]
# 
# 
# class Preprocessor(BaseEstimator, TransformerMixin):
#     """
#     Transforms input data by using tokenization and
#     other normalization and filtering techniques.
#     """
#     def __init__(self, lang,
#                  stopword=True, more_stopwords=None,
#                  spellcheck=False,
#                  stemming=False,
#                  remove_numbers=False,
#                  deasciify=False,
#                  remove_punkt=True,
#                  lowercase=True):
#         
#         self.lang = lang
#         self.stopword = stopword
#         self.more_stopwords = more_stopwords
#         self.spellcheck = spellcheck
#         self.stemming = stemming
#         self.remove_numbers = remove_numbers
#         self.deasciify = deasciify
#         self.remove_punkt = remove_punkt
#         self.lowercase = lowercase
# 
# 
#     #===========================================================================
#     # def __init__(self, lang, 
#     #              params={
#     #                  stopword_key : True, more_stopwords_key : None, 
#     #                  spellcheck_key : False,
#     #                  stemming_key : False, 
#     #                  remove_numbers_key : False, 
#     #                  deasciify_key : False, 
#     #                  remove_punkt_key : True,
#     #                  lowercase_key : True}
#     #     ):
#     #     
#     #     self.lang = lang
#     #     self.stopword = params[stopword]
#     #     self.more_stopwords = params[more_stopwords]
#     #     self.spellcheck = params[spellcheck]
#     #     self.stemming = params[stemming]
#     #     self.remove_numbers = params[remove_numbers]
#     #     self.deasciify = params[deasciify]
#     #     self.remove_punkt = params[remove_punkt]
#     #     self.lowercase = params[lowercase]
#     #===========================================================================
#         
#         
#     # def __init__(self, *args, **kwargs):
#         
#         
# 
# 
#     def fit(self, X, y=None):
#         """
#         Fit simply returns self, no other information is needed.
#         """
#         
#         return self
# 
#     def inverse_transform(self, X):
#         """
#         No inverse transformation
#         """
#         return X
# 
#     def transform(self, X):
#         """
#         Actually runs the preprocessing on each document.
#         """
#         return [self.tokenize(doc) for doc in X]
#     
#  
#     
#     def tokenize(self, doc):
#         tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt) 
#         
#         if self.lowercase:
#             tokens = [token.lower() for token in tokens]
#         
#         # problem: "İ" is lowercased to "i̇"
#         # i = 'i̇'
#         # tokens = [token.replace(i, "i") for token in tokens]        
#         
#         if self.remove_numbers:
#             number_pattern = "[a-zA-z]{,3}\d+"  # d{6,}  
#             tokens = [re.sub(number_pattern, "", token) for token in tokens]
#         
#         if self.stopword:
#             stopword = stopword_lists.get_stopwords(lang=self.lang)          
#             tokens = [token for token in tokens if token not in stopword]  
#             
#         if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
#             tokens = [token for token in tokens if token not in self.more_stopwords]
#                 
#         if self.stemming:
#             tokens = stem_words(tokens, lang=self.lang)
#         
#         if self.stopword:
#             stopword = stopword_lists.get_stopwords(lang=self.lang)          
#             tokens = [token for token in tokens if token not in stopword]  
#             
#         if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
#             tokens = [token for token in tokens if token not in self.more_stopwords]
#         
#         if self.deasciify:
#             tokens = deasciify_words(tokens, self.lang)
# 
#         if self.spellcheck:
#             tokens = spellcheck_words(tokens, self.lang)
#               
#         
#         tokens = eliminate_empty_strings(tokens)
#         return tokens   
# 
# 
#     '''
#     def tokenize(self, doc):
#         tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt) 
#         
#         
#         if self.lowercase:
#             tokens = [token.lower() for token in tokens]
#         
#         # problem: "İ" is lowercased to "i̇"
#         #i = 'i̇'
#         #tokens = [token.replace(i, "i") for token in tokens]        
#         
#         if self.remove_numbers:
#             number_pattern = "[a-zA-z]{,3}\d+"   #d{6,}  
#             tokens = [re.sub(number_pattern, "", token) for token in tokens]
#         
#         if self.stopword:
#             stopword = stopword_lists.get_stopwords(lang=self.lang)          
#             tokens = [token for token in tokens if token not in stopword]  
#             
#         if self.more_stopwords: #len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
#             tokens = [token for token in tokens if token not in self.more_stopwords]
#                 
#         if self.stemming:
#             tokens = stem_words(tokens, lang=self.lang)
#         
#         if self.stopword:
#             stopword = stopword_lists.get_stopwords(lang=self.lang)          
#             tokens = [token for token in tokens if token not in stopword]  
#             
#         if self.more_stopwords: #len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
#             tokens = [token for token in tokens if token not in self.more_stopwords]
#         
#         if self.deasciify:
#             tokens = deasciify_words(tokens, self.lang)
# 
#         if self.spellcheck:
#             tokens = spellcheck_words(tokens, self.lang)
#               
#         
#         tokens = eliminate_empty_strings(tokens)
#         
#         for token in tokens:
#             yield token
#     '''
#         
#===============================================================================



def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


def run_prep():
    
    
    classifier = sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 

    lang = "tr"
    stopword_choice = True
    more_stopwords_list = None
    spellcheck_choice = False
    stemming_choice = False
    number_choice = False
    deasc_choice = True
    punct_choice = True
    case_choice = True
    
    ngramrange = (1, 2)  # tuple
    nmaxfeature = 10000  # int or None  
    norm = "l2"
    use_idf = True
                 
    preprocessor = Preprocessor(lang=lang,
                                 stopword=stopword_choice, more_stopwords=more_stopwords_list,
                                 spellcheck=spellcheck_choice,
                                 stemming=stemming_choice,
                                 remove_numbers=number_choice,
                                 deasciify=deasc_choice,
                                 remove_punkt=punct_choice,
                                 lowercase=case_choice
                                )
    tfidfvect = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False,
                                use_idf=use_idf, ngram_range=ngramrange, max_features=nmaxfeature)

    
    keyword = "arıza"
    apipe = tbt.get_keyword_pipeline(keyword)
    keyword2 = "pstn"
    pstnpipe = tbt.get_keyword_pipeline(keyword2)
    polpipe1 = tbt.get_polylglot_polarity_count_pipe(lang)
    polpipe2 = tbt.get_polylglot_polarity_value_pipe(lang)
    polpipe3 = obt.get_lexicon_count_pipeline(tokenizer=identity)
    
    tokenizedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                         ('union1',
                                          skpipeline.FeatureUnion(
                                              transformer_list=[
                                         ('vect', tfidfvect),
                                         ('polarity3', polpipe3), ])), ]
                                        )
    
    textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion([
                                         ('has_ariza', apipe),
                                         ('has_pstn', pstnpipe),
                                         ('polarity1', polpipe1),
                                         ('polarity2', polpipe2), ]),)])
    
    model = skpipeline.Pipeline([
        
        # ('preprocessor', preprocessor),
        
        ("union", skpipeline.FeatureUnion(transformer_list=[
            
            ('tfidf', tokenizedpipe),
            
            ('txtpipe', textbasedpipe),
            
            ])
         ),
            
        ('classifier', classifier),
        ])
    
    t0 = time()
    print("Read data")
    instances, labels = get_data.get_data()
    
    N = 100
    instances, labels = corpus_io.select_N_instances(N, instances, labels)
    # instances_train, instances_test, ytrain, ytest = cv.train_test_split(instances, labels, test_size=0.30, random_state=20)
    
    print("Start classification\n..")
    nfolds = 5
    ypred = cv.cross_val_predict(model, instances, labels, cv=nfolds)
    tc_utils.get_performance(labels, ypred, verbose=True)
    t1 = time()
    print("Classification took ", round(t1 - t0, 2), "sec.")




###########################3
## text prep test
def tokenize(doc, lang="tr", stopword=False, more_stopwords=None,
                 spellcheck=False,
                 stemming=True,
                 remove_numbers=False,
                 deasciify=False,
                 remove_punkt=False,
                 lowercase=False):
    #tokens = tokenizer(doc, lang=language_map(lang), remove_punkt=remove_punkt) 
    tokens = doc.split() 
                  
    if lowercase and lang not in ["ar", "arab", "arabic"]:
        tokens = [token.lower() for token in tokens]
    
    # problem: "İ" is lowercased to "i̇"
    # i = 'i̇'
    # tokens = [token.replace(i, "i") for token in tokens]        
    
    if deasciify:
        tokens = deasciify_words(tokens, lang)
        
    if spellcheck:
        tokens = spellcheck_words(tokens, lang)
    if remove_numbers:
        number_pattern = "[a-zA-z]{,3}\d+"  # d{6,}  # TODO real numbers & rational numbers
        tokens = [re.sub(number_pattern, "", token) for token in tokens]
    
    if stopword:
        stopwords = stopword_lists.get_stopwords(lang=lang)          
        tokens = [token for token in tokens if token not in stopwords]  
    if more_stopwords:  # len(more_stopwords) > 0:    # re-organize not to have this list through memory but disc
        tokens = [token for token in tokens if token not in more_stopwords]
    
    if stemming:
        #tokens = stem_words(tokens, lang=lang)
        tokens = tr_stemmer.stem_words(tokens)
    '''
    if stopword:
        stopwords = stopword_lists.get_stopwords(lang=lang)                      
        tokens = [token for token in tokens if token not in stopwords]  
          
    if more_stopwords:  # len(more_stopwords) > 0:    # re-organize not to have this list through memory but disc
        tokens = [token for token in tokens if token not in more_stopwords]
    '''
        
    tokens = eliminate_empty_strings(tokens)
  
    #print(doc,"  -> ", tokens)
    
    return tokens   
   
    
def main():    
        
    print()

    
    # tweet polarity preprocessor
    x = Preprocessor(lang="tr", stopword=True, more_stopwords=None,
                 spellcheck=False,
                 stemming=True,
                 remove_numbers=True,
                 deasciify=True,
                 remove_punkt=True,
                 lowercase=True)

    
    
    '''
    sentences = ["film guzelmis.",
                 "filmler güzelmiş.",
                      "film güzelmiş.",
                      "film güzelmiş..",
                      "film güzelmiş",
                      "hic begenmedim",
                      "hiç beğenmedim.",
                      "kapalııııı",
                      "Turkcell 97 liraya veriyor...boş verin aveayı Turkcellden alın =)  "]
    '''
    sentences = ["Turkcell 97 liraya veriyor...boş verin aveayı Turkcellden alın =)  "]
    s2 = [x.tokenize(s) for s in sentences]
    print(s2)
    for i in s2:
        print([type(j) for j in i])
    
    s3 = [" ".join(i) for i in s2]
    print(list(set(s3)))
    
    a = ['türkcell', '', 'liraya', 'veriyor', 'boş', 'verin', 'aveayı', 'türkcellden', 'alın']
    print("st1: ", stem_words(a, lang="tr"))
    print("st2: ", tr_stemmer.stem_words(a))
    print("st3: ", [tr_stemmer.stem(word) for word in a])





if __name__ == "__main__":
  
  
    # tweet polarity preprocessor
    x = Preprocessor(lang="tr", stopword=True, more_stopwords=None,
                 spellcheck=False,
                 stemming=False,
                 remove_numbers=True,
                 deasciify=True,
                 remove_punkt=True,
                 lowercase=True)
    
    from dataset import corpus_io
    import os
    folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    fname = "tr_polartweets.csv"
    csvsep="\t"
    text_col="text"
    cat_col="polarity"
    texts, _ = corpus_io.read_texts_and_category_csv(os.path.join(folder, fname), 
                                                 csvsep, text_col, cat_col, shuffle=True)
    
    Ns = [20, 50, 100, 200, 500, 1000, 2000]
    ts = []
    
    
    for N in Ns:
        stexts = texts[:N]
                    
        t0 = time()    
        #roots = [x.tokenize(stext) for stext in stexts]
        #roots = [tokenize(stext) for stext in stexts]
        #roots = [tr_stemmer.stem_words(t.split()) for t in stexts]
        for t in stexts:
            tr_stemmer.stem_words(t.split())
        t1 = time()
        ts.append((N, t1-t0))
        print(N, t1-t0)
    
    print(ts)


    '''
    for N in Ns:
        stexts = texts[:N]
        words = []
        for t in stexts:
            words.extend(t.split())
            
        t0 = time()    
        roots = stem_words(words, lang="tr")
        t1 = time()
        ts.append((N, t1-t0))
        print(N, t1-t0)
    
    print(ts)
    '''
    
    
    
    
    
    