'''
Created on Mar 7, 2017

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

import textblob
from textblob import TextBlob   # for translation



import modules.learning.text_categorization.prototypes.text_based_transformers as tbt
import modules.learning.text_categorization.prototypes.token_based_transformers as obt
#import text_based_transformers as tbt
#import token_based_transformers as obt
import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.pipeline as skpipeline
from modules.learning.text_categorization import tc_utils2

from modules.learning.dataset import corpus_io

from django_docker.learning.language_tools.gtranslate import *

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

    if lang in ["tr", "turkish"]:
        words = [tr_stemmer.stem2(word) for word in words]
    
    if lang in ["en", "english"]:
        words = [en_stemmer.stem1(word) for word in words]
    
    if lang in ["ar", "arabic", "arab"]:
        words = [ar_stemmer.stem(word) for word in words]
    return words

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
        self.stopword = stopword
        self.more_stopwords = more_stopwords
        self.spellcheck = spellcheck
        self.stemming = stemming
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
    #     self.stopword = params[stopword]
    #     self.more_stopwords = params[more_stopwords]
    #     self.spellcheck = params[spellcheck]
    #     self.stemming = params[stemming]
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
        
        if self.stopword:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopwords]  
            
        if self.more_stopwords:  # len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
            tokens = [token for token in tokens if token not in self.more_stopwords]
                
        if self.stemming:
            tokens = stem_words(tokens, lang=self.lang)
        
        if self.stopword:
            stopwords = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopwords]  
            
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
        
        if self.stopword:
            stopword = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopword]  
            
        if self.more_stopwords: #len(self.more_stopwords) > 0:    # re-organize not to have this list through memory but disc
            tokens = [token for token in tokens if token not in self.more_stopwords]
                
        if self.stemming:
            tokens = stem_words(tokens, lang=self.lang)
        
        if self.stopword:
            stopword = stopword_lists.get_stopwords(lang=self.lang)          
            tokens = [token for token in tokens if token not in stopword]  
            
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





if __name__ == '__main__':
    
    
    text = """ الله يعينك على ما بلاك هي مسئولية وانت قدها كان نفسي اشارك معاك في شيل المسئولية ولكن نصيحه خليك زي القطار سير في خط مستقيم وكلنا وراك """
    
    '''
    text = """ النظاملك بالقيام بتعديلات وإضافة الصفحات بحرية كاملة - أي أنك تستطيع الآنالقيام بالتعديل على اي صفحة، باستثناء عدد قليل من الصفحات المحمية.غير أنه لابد من التمعن لمعرفة مدى حياديةوموضوعيةمثل هذه الموسوعات. بدأ مشروع ويكيبيديا في 2001، ويوجد اليوم أكثر من3 مليون مقالفي الموسوعة في كافة اللغات، منها أكثر من 900.000مقال للغةالإنجليزيةوحدها . واليوم، يقوم مئات الآلاف من المتطوعينوالمهتمين حول العالم بإجراء التعديلات يوميا، إضافة إلى إنشاء العديدمن المقالات الجديدة. بدأت النسخة العربية من الموسوعة الحرة فينهايات عام 2003، ولا تزال الموسوعةال
"""
    '''
    '''
    text = """
    Son yıllarda gençler arasında hızla yayılan nargile, verdiği keyfin yanında sağlığımızı da tehdit ediyor

        Dedelerimizin keyfi nargile, son yıllarda gördüğü ilgiyle ikinci baharını yaşasa da sağlık açısından bakıldığında durum pek iç açıcı görünmüyor.
    Dokuz Eylül Üniversitesi'nden Doç. Dr. Oğuz Kılınç, nargile kullanan 397 kişinin, akciğer fonksiyonları yönünden araştırıldığını ve nargile içenlerin akciğer fonksiyonlarının tütün kullanmayanlara göre yüzde 30 azaldığını saptadıklarını söyledi. Kılınç, nargileyle sigarayı birlikte kullananların akciğer fonksiyonlarının ise yüzde 40 daha azaldığını belirtti. 
    
    """
    '''
    
    prep = Preprocessor(lang="ar",
                 stopword=True, more_stopwords=None,
                 spellcheck=False,
                 stemming=True,
                 remove_numbers=True,
                 deasciify=False,
                 remove_punkt=True,
                 lowercase=True)
    
    tokens2 = prep.tokenize(text)
    
    tokens = tokenizer(text, lang="", remove_punkt=False)
    roots = stem_words(tokens, lang="ar")
    en_text = TextBlob(text).translate(from_lang="ar", to="en")
    print(en_text)
    
    for token,ar_stem in zip(tokens, roots):
        #en_token = TextBlob(token).translate(from_lang="ar", to="en")
        en_token = translate_textblob(token)
        #en_token2 = translator("ar", "en", token)
        en_stem = translate_textblob(ar_stem)
        print(token, " : ", token.lower(), " - ", en_token)  #, " -- ", en_token2)
        print(ar_stem, " * ", en_stem)
        print("--")
    
    
    for token in tokens2:
        en_token = translate_textblob(token)
        print(token, "  +  ", en_token)
    
    print(len(tokens), len(tokens2))
    