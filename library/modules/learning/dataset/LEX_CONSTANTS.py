'''
Created on Nov 22, 2016

@author: dicle
'''

import sys
sys.path.append("..")

import os

'''
BOUN_POLARITY_LEXICON_FOLDER = "/home/dicle/Documents/lexicons/tr_sentiment_boun"
POLAR_EMOTICON_FOLDER = "/home/dicle/Documents/lexicons/EmoticonSentimentLexicon"
'''

_lexicon_folder = "/home/dicle/git/serdoo-servis2/django_docker/learning/_lexicons"  # sonra proje altında daha belirgin bir yere koyup relative path yazarız.
BOUN_POLARITY_LEXICON_FOLDER = os.path.join(_lexicon_folder, "tr_sentiment_boun")
POLAR_EMOTICON_FOLDER = os.path.join(_lexicon_folder, "EmoticonSentimentLexicon")
EN_POLARITY_LEXICON_FOLDER = os.path.join(_lexicon_folder, "en_polar_words")
AR_POLARITY_LEXICON_FOLDER1 = os.path.join(_lexicon_folder, "ar_polarity_masc")
#lex_folder2 = "../../learning/_lexicons"


