# -*- coding: utf-8 -*-
'''
Created on Oct 17, 2016

@author: dicle
'''

from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

import stop_words as pystopwords



def tr_stopwords():
    
    stopwords = ["a", "acaba", "altı", "ama", "ancak", "artık", "asla", "aslında", "az", "b",
                 "bana", "bazen", "bazı", "bazıları", "bazısı", "belki", "ben", "beni", "benim",
                 "beş", "bile", "bir", "birçoğu", "birçok", "birçokları", "biri", "birisi", "birkaç",
                 "birkaçı", "birşey", "birşeyi", "biz", "bize", "bizi", "bizim", "böyle", "böylece",
                 "bu", "buna", "bunda", "bundan", "bunu", "bunun", "burada", "bütün", "c", "ç", "çoğu",
                 "çoğuna", "çoğunu", "çok", "çünkü", "d", "da", "daha", "de", "değil", "demek", "diğer",
                 "diğeri", "diğerleri", "diye", "dokuz", "dolayı", "dört", "e", "elbette", "en", "f",
                 "fakat", "falan", "felan", "filan", "g", "gene", "gibi", "ğ", "h", "hâlâ", "hangi",
                 "hangisi", "hani", "hatta", "hem", "henüz", "hep", "hepsi", "hepsine", "hepsini",
                 "her", "her biri", "herkes", "herkese", "herkesi", "hiç", "hiç kimse", "hiçbiri",
                 "hiçbirine", "hiçbirini", "ı", "i", "için", "içinde", "iki", "ile", "ise", "işte", "j",
                 "k", "kaç", "kadar", "kendi", "kendine", "kendini", "ki", "kim", "kime", "kimi", "kimin",
                 "kimisi", "l", "lakin", "m", "madem", "mı", "mı", "mi", "mu", "mu", "mü", "mü", "n", "nasıl",
                 "ne", "ne kadar", "ne zaman", "neden", "nedir", "nerde", "nerede", "nereden", "nereye",
                 "nesi", "neyse", "niçin", "niye", "o", "on", "ona", "ondan", "onlar", "onlara", "onlardan",
                 "onların", "onu", "onun", "orada", "oysa", "oysaki", "ö", "öbürü", "ön", "önce",
                 "ötürü", "öyle", "p", "r", "rağmen", "s", "sana", "sekiz", "sen", "senden", "seni", "senin",
                 "siz", "sizden", "size", "sizi", "sizin", "son", "sonra", "ş", "şayet", "şey", "şeyden",
                 "şeye", "şeyi", "şeyler", "şimdi", "şöyle", "şu", "şuna", "şunda", "şundan", "şunlar", "şunu",
                 "şunun", "t", "tabi", "tamam", "tüm", "tümü", "u", "ü", "üç", "üzere", "v", "vb", "var", "ve", "veya",
                 "veyahut", "vs", "y", "ya", "ya da", "yada", "yani", "yedi", "yerine", "yine", "yoksa", "z", "zaten", "zira"]
    
    
    return stopwords


def en_stopwords():
    # return stopwords.words("english")  # nltk has ~100 words
    return list(stop_words.ENGLISH_STOP_WORDS)  # scikit has around ~300

def ar_stopwords():

    stop_words = pystopwords.get_stop_words('en')
    return stop_words

def get_stopwords(lang):
    if lang in ["en", "english"]:
        return en_stopwords()
    elif lang in ["tr", "turkish"]:
        return tr_stopwords()
    elif lang in ["ar", "arab", "arabic"]:
        return ar_stopwords()
    else:
        raise LookupError


def email_specific_stopwords():
    
    
    l1 = ["merhaba", "saygılarımla", "ilgili", "see", "attached", "file"]
    l2 = get_stopwords(lang="en")
    return l1 + l2
    
    
if __name__ == "__main__":
    
    l = get_stopwords("rt")  
    print(l)

    
