'''
Created on Jan 3, 2017

@author: dicle
'''


import nltk.stem as stemmer


def stem1(word):
    
    snowball_stemmer = stemmer.SnowballStemmer("english")
    return snowball_stemmer.stem(word)
