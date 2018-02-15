'''
Created on Jan 31, 2017

@author: dicle
'''


import text_categorization.sentiment_analysis.sentiment_feature_extractors as sf

from language_tools import TokenHandler

from dataset import corpus_io

if __name__ == '__main__':
    
    
    x, y = sf.get_polyglot_polarity_count("merhaba", "tr")
    print(x,y)
    
    print(TokenHandler.is_punctuation(",,,"))
    
    x, y = corpus_io.get_20newsgroup()
    print(len(x), len(y))