'''
Created on Jan 18, 2017

@author: dicle
'''

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


from modules.language_tools import stopword_lists, tr_stemmer, en_stemmer
from modules.language_tools.spellchecker import en_spellchecker 
import nltk.tokenize as nltktokenizer

import modules.prototypes.text_based_transformers as tbt
import modules.prototypes.token_based_transformers as obt
#import text_based_transformers as tbt
#import token_based_transformers as obt
import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.pipeline as skpipeline
from modules.dataset import tc_utils2

from modules.dataset import corpus_io


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
        tokens = [token for token in tokens if token.isalnum()] # this is already eliminating the punctuation!
    return tokens

'''
def stem_words(words, lang):

    if lang in ["tr", "turkish"]:
        words = [tr_stemmer.stem2(word) for word in words]
    
    if lang in ["en", "english"]:
        words = [en_stemmer.stem1(word) for word in words]
    
    return words
'''

def stem_words(words, lang):

    if lang in ["tr", "turkish"]:
        words = tr_stemmer.stem_words(words)
    
    if lang in ["en", "english"]:
        words = [en_stemmer.stem1(word) for word in words]
    
    if lang in ["ar", "arabic", "arab"]:
        words = [ar_stemmer.stem(word) for word in words]
    
    return words


def deasciify_words(words, lang):
    
    if lang in ["tr", "turkish"]:
        return [Deasciifier(token).convert_to_turkish() for token in words]
    else:  # not applicable for english and arabic
        return words


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
        
        self.lang = lang
        self.eliminate_stopwords = stopword
        self.more_stopwords = more_stopwords
        self.spellcheck = spellcheck
        self.apply_stemming = stemming
        self.remove_numbers = remove_numbers
        self.deasciify = deasciify
        self.remove_punkt = remove_punkt
        self.lowercase = lowercase


    #===========================================================================
    # def __init__(self, lang, 
    #              params={
    #                  stopword_key : True, more_stopwords_key : None, 
    #                  spellcheck_key : False,
    #                  stemming_key : False, 
    #                  remove_numbers_key : False, 
    #                  deasciify_key : False, 
    #                  remove_punkt_key : True,
    #                  lowercase_key : True}
    #     ):
    #     
    #     self.lang = lang
    #     self.eliminate_stopwords = params[stopword]
    #     self.more_stopwords = params[more_stopwords]
    #     self.spellcheck = params[spellcheck]
    #     self.apply_stemming = params[stemming]
    #     self.remove_numbers = params[remove_numbers]
    #     self.deasciify = params[deasciify]
    #     self.remove_punkt = params[remove_punkt]
    #     self.lowercase = params[lowercase]
    #===========================================================================
        
        
    # def __init__(self, *args, **kwargs):
        
        


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
        return [self.tokenize(doc) for doc in X]
    
    
    def tokenize(self, doc):
        tokens = tokenizer(doc, lang=language_map(self.lang), remove_punkt=self.remove_punkt) 
        
        if self.lowercase and not self.lang in ["ar", "arab", "arabic"]:
            tokens = [token.lower() for token in tokens]
        
        # problem: "İ" is lowercased to "i̇"
        # i = 'i̇'
        # tokens = [token.replace(i, "i") for token in tokens]        
        
        if self.remove_numbers:
            number_pattern = "[a-zA-z]{,3}\d+"  # d{6,}  
            tokens = [re.sub(number_pattern, "", token) for token in tokens]
        
        if self.eliminate_stopwords:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)  
            try:        
                tokens = [token for token in tokens if token not in stopwords] 
            except:
                tokens=tokens 
            
        if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
            tokens = [token for token in tokens if token not in self.more_stopwords]
                
        if self.apply_stemming:
            tokens = stem_words(tokens, lang=self.lang)
        
        if self.eliminate_stopwords:
            stopwords = stopword_lists.get_stopwords(lang=self.lang) 
            try:         
                tokens = [token for token in tokens if token not in stopwords]  
            except:
                tokens=tokens
                
        if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
            tokens = [token for token in tokens if token not in self.more_stopwords]
        
        if self.deasciify:
            tokens = deasciify_words(tokens, self.lang)

        if self.spellcheck:
            tokens = spellcheck_words(tokens, self.lang)
              
        
        tokens = eliminate_empty_strings(tokens)
        return tokens   


    '''
    def tokenize(self, doc):
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
        
        if self.deasciify:
            tokens = deasciify_words(tokens, self.lang)

        if self.spellcheck:
            tokens = spellcheck_words(tokens, self.lang)
              
        
        tokens = eliminate_empty_strings(tokens)
        
        for token in tokens:
            yield token
    '''
        



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
    
    
if __name__ == "__main__":
    
    print()
    x = Preprocessor("tr", stemming=True)
    print(type(x))
    print(x.tokenize("merhaba merhaba nasılsın iyiyim"))
    
    '''    
    print()
    x = Preprocessor("t")
    print(type(x))
    '''
    
