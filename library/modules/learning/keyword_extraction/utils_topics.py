'''
Created on May 15, 2017

@author: dicle
'''

import sys
sys.path.append("..")

def print_topic_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics_keywords = _get_top_words(model, feature_names, n_top_words)
    for topic, keywords in topics_keywords.items():
        print(topic, " : ", ", ".join(keywords))


def get_topic_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics_keywords = _get_top_words(model, feature_names, n_top_words)
    return topics_keywords


def _get_top_words(model, feature_names, n_top_words):
    topics_keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        topical_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics_keywords[topic_idx] = topical_terms
    return topics_keywords



