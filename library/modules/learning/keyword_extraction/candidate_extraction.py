'''
Created on Oct 12, 2016

@author: dicle
'''


import itertools, nltk, string

from modules.learning.language_tools import stopword_lists

'''
Adapted from http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/

'''

def extract_candidate_words(text, lang, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    
    stop_words = set(stopword_lists.get_stopwords(lang))   #set(nltk.corpus.stopwords.words('english'))
    
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def extract_candidate_chunks(text, lang, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(stopword_lists.get_stopwords(lang)) #set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    #print("nsents: %d" % len(tagged_sents))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    #print("nchunks: %d" % len(all_chunks))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(all_chunks, lambda x : x[2] != 'O') if key]

    #print("ncandidates: %d" % len(candidates))

    candidates = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    #print("ncandidates: %d" % len(candidates))
    return candidates



