'''
Created on Nov 22, 2016

@author: dicle
'''

import os
from django.conf import settings

BASE_DIR = os.path.join(settings.BASE_DIR, '../modules/')

'''
_lexicon_folder = "/code/django_docker/learning/dataset/lexicons"

BOUN_POLARITY_LEXICON_FOLDER = "/code/django_docker/learning/dataset/lexicons/tr_sentiment_boun"
POLAR_EMOTICON_FOLDER = "/code/django_docker/learning/dataset/lexicons/EmoticonSentimentLexicon"
EN_POLARITY_LEXICON_FOLDER = "/code/django_docker/learning/dataset/lexicons/en_polar_words"
AR_POLARITY_LEXICON_FOLDER1 = "/code/django_docker/learning/dataset/lexicons/ar_polarity_masc"
'''



_lexicon_folder = BASE_DIR+"dataset/lexicons"  # sonra proje altında daha belirgin bir yere koyup relative path yazarız.
BOUN_POLARITY_LEXICON_FOLDER = os.path.join(_lexicon_folder, "tr_sentiment_boun")
POLAR_EMOTICON_FOLDER = os.path.join(_lexicon_folder, "EmoticonSentimentLexicon")
EN_POLARITY_LEXICON_FOLDER = os.path.join(_lexicon_folder, "en_polar_words")
AR_POLARITY_LEXICON_FOLDER1 = os.path.join(_lexicon_folder, "ar_polarity_masc")
