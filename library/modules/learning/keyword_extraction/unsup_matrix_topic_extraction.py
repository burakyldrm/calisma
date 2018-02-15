'''
Created on Oct 11, 2016

@author: dicle
'''

from __future__ import print_function
import language_tools._ENSimpleTokenHandler as tokenhandler


'''
Modified the code at http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html


# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

'''

from time import time

import sklearn.feature_extraction.text as txtfeatext
import sklearn.decomposition as decomposer
import sklearn.datasets as datasets




n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

language = "en"
apply_stemming = False
remove_stopwords = True
ngram_range = (2, 2) # include both unigrams and bigrams




def print_topic_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics_keywords = get_top_words(model, feature_names, n_top_words)
    for topic, keywords in topics_keywords:
        print(topic, " : ", ", ".join(keywords))


def get_top_words(model, feature_names, n_top_words):
    topics_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        topical_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics_keywords.append(("Topic #"+str(topic_idx), topical_terms))
    return topics_keywords

'''
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
'''





# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = datasets.fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))



preprocessor = tokenhandler._ENSimpleTokenHandler(stem=apply_stemming, stopword=remove_stopwords)


tf_vectorizer = txtfeatext.CountVectorizer(tokenizer=preprocessor, 
                                      ngram_range=ngram_range,
                                      max_features=n_features)  # @TODO encoding?? (default utf8 but may depend on the user per application needs)
tf_matrix = tf_vectorizer.fit_transform(data_samples)





tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                      ngram_range=ngram_range,
                                      max_features=n_features)
tfidf_matrix = tfidf_vectorizer.fit_transform(data_samples)



# apply NMF
print("Applying NMF on tf*idf weighted terms, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = decomposer.NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf_matrix)


print("Applying LDA on tf weighted terms, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = decomposer.LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf_matrix)


print("\nTopics in NMF model:")
print_topic_words(nmf, tfidf_vectorizer, n_top_words)


print("\nTopics in LDA model:")
print_topic_words(lda, tf_vectorizer, n_top_words)



    
    

