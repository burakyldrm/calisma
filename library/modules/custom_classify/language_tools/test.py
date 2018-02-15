'''
Created on Sep 27, 2016

@author: dicle
'''

import nltk
import nltk.stem as stem
from nltk.corpus import stopwords
#import stop_words
import re
from dataset import corpus_utils

def split_sentences1(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
    sentences = sentence_delimiters.split(text)
    return sentences

def split_sentences2(text):
    return nltk.sent_tokenize(text)

if __name__ == '__main__':
    
    w = "w"
    a = "a"
    print(w, end="")
    print()
    print(a)
    
    stemmer = stem.PorterStemmer()
    stemmer2 = stem.SnowballStemmer(language="english")
    print(stemmer.stem("colonization"))
    print(stemmer2.stem("colonization"))
    print(len(stopwords.words("english")))
    
    lang = "tr"
    folderpath = "/home/dicle/Documents/data/tr/radikal_5class_newstexts/ekonomi"
    #instances = corpus_utils.read_n_files(folderpath, N=2)
    instances, labels = corpus_utils.get_20newsgroup()
    instances = instances[:2]
    for i,text in enumerate(instances):
        print(i," Sentences:")
        print(split_sentences1(text))
        print("####")
        print(split_sentences2(text))
        print(nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in split_sentences2(text)))
    