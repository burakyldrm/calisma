'''
Created on Oct 12, 2016

@author: dicle
'''

import itertools, nltk, string
import gensim
import networkx
from itertools import takewhile, tee

import candidate_extraction
    

def score_keyphrases_by_tfidf(texts, candidate_type='chunks'):
    
    # extract candidates from each text in texts, either chunks or words
    if candidate_type == 'chunks':
        boc_texts = [candidate_extraction.extract_candidate_chunks(text) for text in texts]
    elif candidate_type == 'words':
        boc_texts = [candidate_extraction.extract_candidate_words(text) for text in texts]
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf, dictionary


def score_keyphrases_by_textrank(text, n_keywords=50):
    
    
    # tokenize for all words, and extract *candidate* words

    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = candidate_extraction.extract_candidate_words(text)
 
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    
    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)
 
if __name__ == '__main__':
    
    text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility " \
       "of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. " \
       "Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating"\
       " sets of solutions for all types of systems are given. These criteria and the corresponding algorithms " \
       "for constructing a minimal supporting set of solutions can be used in solving all the considered types of " \
       "systems and systems of mixed types."

    '''
    print("Chunks:")
    print(extract_candidate_chunks(text))
    print("\nWords:")
    print(extract_candidate_words(text))
    '''
    
    # @todo
    print("Phrases scored by tfidf")
    corp, dict = score_keyphrases_by_tfidf([text])
    print(corp)
    print(dict)
    
    print("Phrases scored by textrank")
    phrases = score_keyphrases_by_textrank(text)
    print(phrases)
    
    