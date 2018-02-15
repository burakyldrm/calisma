'''
Created on Jan 18, 2017

@author: dicle
'''

import sys
sys.path.append("..")


from polyglot.text import Text

from modules.learning.dataset import corpus_io


def get_polyglot_polarity_count(text, lang):
    t = Text(text, hint_language_code=lang)
    
    npos = 0
    nneg = 0
    #print(len(text))
    for w in t.tokens:
        val = w.polarity
        if val < 0:
            nneg += 1
        elif val > 0:
            npos += 1
    
    return npos, nneg


def get_polyglot_doc_polarity(text, lang):
  
    t = Text(text, hint_language_code=lang)
    polarity = 0.0

    try:
        polarity = t.polarity
    except:
        pass
    finally:
        return polarity




# label is pos or neg
'''
lexicon is from
Cuneyd Murad Ozsert and Arzucan Ozgur. Word Polarity Detection using a Multilingual Approach. 
(CICLing), 2013.
''' 
def get_boun_polarity_count(words, label):

    boun_polars = corpus_io.get_boun_polarity_lexicon(label)
    
    npolar = len([w for w in words if w in boun_polars])
    # print(len(boun_polars), npolar)
    return npolar




'''
lexicon is from
Minqing Hu and Bing Liu. Mining and summarizing customer reviews. 
KDD 2004.
'''
def get_english_polarity_count(words, label):
    polar_terms = corpus_io.get_english_polarity_lexicon(label)
    npolar = len([w for w in words if w in polar_terms])
    return npolar

def get_emoticon_polarity_count(words, label):

    polar_emoticons = corpus_io.get_emoticon_lexicon(label)
    npolar = len([w for w in words if w in polar_emoticons])
    return npolar




'''
lexicon is from
Al-Moslmi, Tareq, et al. 
"Arabic senti-lexicon: Constructing publicly available language resources for Arabic sentiment analysis." 
Journal of Information Science (2017): 
'''
def get_arabic_polarity_count1(tokens):
    poswords, negwords = corpus_io.get_arabic_polarity_lexicon1()
    
    npos = len([w for w in tokens if w in poswords])
    nneg = len([w for w in tokens if w in negwords])
    
    return npos, nneg
    


if __name__ == '__main__':
    
    print()
