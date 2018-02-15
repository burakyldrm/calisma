'''
Created on Oct 21, 2016

@author: dicle
'''

from collections import Counter
import pickle
import re


# import cPickle as cpickle
def _dumpWORDS(TEXTSPATH, pWORDSPATH):
    # def words(text): return re.findall(r'\w+', text.lower())
    def words(text): return re.findall(r'\w+', text)
    WORDS = Counter(words(open(TEXTSPATH).read()))
    
    pickle.dump(WORDS, open(pWORDSPATH, "wb"))

def getWORDS(pWORDSPATH):
    WORDS = pickle.load(open(pWORDSPATH, "rb"))
    return WORDS

    
