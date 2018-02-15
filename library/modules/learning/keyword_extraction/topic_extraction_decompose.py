'''
Created on Sep 7, 2016

'''

from __future__ import print_function

import os
from django_docker.learning.dataset import io_utils

'''
Modified the code at http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
-dicle

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

'''

from time import time

import sklearn.feature_extraction.text as txtfeatext
import sklearn.decomposition as decomposer
import sklearn.cluster as skcluster
import sklearn.datasets as skdatasets

from modules.learning.language_tools import TokenHandler, stopword_lists
from modules.learning.dataset import corpus_io


def print_topic_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics_keywords = _get_top_words(model, feature_names, n_top_words)
    for topic, keywords in topics_keywords:
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


'''
def _get_top_words(model, feature_names, n_top_words):
    topics_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        topical_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics_keywords.append(("Topic #"+str(topic_idx), topical_terms))
    return topics_keywords
'''

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.
def fetch_dataset(n_samples):
    print("Loading dataset...")
    t0 = time()
    dataset = skdatasets.fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    data_samples = dataset.data[:n_samples]
    print("done in %0.3fs." % (time() - t0))
    
    return data_samples




def _extract_topics_decomposer(data_samples, preprocessor, 
                              decomposer_model,
                              n_features, n_topics, n_top_words, n_gram_range=(1,1), more_stopwords=None):
    
    
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_samples)
    decomposer_model.fit(tfidf_matrix)  
    return get_topic_words(decomposer_model, tfidf_vectorizer, n_top_words)
    

'''
def extract_topics_lda(data_samples, preprocessor, n_features, n_topics, n_top_words, n_gram_range=(1,1), more_stopwords=None):
    
      
    t0 = time()

    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_samples)
    
    t1 = time()
    print("Vectorizing took ", (t1-t0), "sec.")
    
    
    
    #print("Applying LDA on tf weighted terms, n_samples=%d and n_features=%d..." % (n_samples, n_features))
    lda = decomposer.LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    lda.fit(tfidf_matrix)  
      
    t2 = time()
    print("LDA took ", t2 - t1, "sec.")
    return get_topic_words(lda, tfidf_vectorizer, n_top_words)
'''

'''
def extract_topics_nmf(data_samples, preprocessor, n_features, n_topics, n_top_words, n_gram_range=(1,1), more_stopwords=None):
    
      
    t0 = time()

    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_samples)
    
    t1 = time()
    print("Vectorizing took ", (t1-t0), "sec.")
    
    # apply NMF
  
    nmf = decomposer.NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf_matrix)
    #nmf_topics = get_topic_words(model, vectorizer, n_top_words)
    
    t2 = time()
    print("NMF took ", t2 - t1, "sec.")
    return get_topic_words(nmf, tfidf_vectorizer, n_top_words)

'''

def extract_topics_lda(data_samples, preprocessor, n_features, n_topics, n_top_words, n_gram_range=(1,1), more_stopwords=None):

    lda = decomposer.LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    
    topics_words = _extract_topics_decomposer(data_samples, preprocessor, lda, n_features, n_topics, n_top_words, n_gram_range, more_stopwords)
    
    return topics_words





def extract_topics_nmf(data_samples, preprocessor, n_features, n_topics, n_top_words, n_gram_range=(1,1), more_stopwords=None):
     
    nmf = decomposer.NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
    
    topics_words = _extract_topics_decomposer(data_samples, preprocessor, nmf, n_features, n_topics, n_top_words, n_gram_range, more_stopwords)

    return topics_words
    
    
    

def extract_topics(data_samples, lang, n_features, n_topics, n_top_words, more_stopwords=None):
    
    n_gram_range = (2, 2)
    
    t0 = time()

    preprocessor = None
    if lang in ["en", "english"]:
        preprocessor = TokenHandler.EnTokenHandler(stemming=True, stopword=True)
    if lang in ["tr", "turkish"]:
        preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=more_stopwords, 
                                                   stemming=False, 
                                                   remove_numbers=True,
                                                   deasciify=True, remove_punkt=True)

    '''    
    tf_vectorizer = txtfeatext.CountVectorizer(tokenizer=preprocessor, 
                                      ngram_range=(1, 2),
                                      max_features=n_features)  
    tf_matrix = tf_vectorizer.fit_transform(data_samples)
    ''' 
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_samples)
    
    t1 = time()
    print("1- Vectorizing took ", (t1-t0), "sec.")
    
    # apply NMF
    '''
    print("Applying NMF on tf*idf weighted terms, n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    '''
    nmf = decomposer.NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf_matrix)
    print("\nTopics in NMF model:")
    print_topic_words(nmf, tfidf_vectorizer, n_top_words)
    #nmf_topics = get_topic_words(model, vectorizer, n_top_words)
    
    t2 = time()
    print("NMF took ", t2 - t1, "sec.")
    
    #print("Applying LDA on tf weighted terms, n_samples=%d and n_features=%d..." % (n_samples, n_features))
    lda = decomposer.LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    lda.fit(tfidf_matrix)  
    print("\nTopics in LDA model:")
    print_topic_words(lda, tfidf_vectorizer, n_top_words)
    
    
    t3 = time()
    print("LDA took ", t3 - t2, "sec.")
    
    '''
    kmeans = skcluster.KMeans(n_clusters=n_topics, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(tfidf_matrix)
    print("\nTopics in k-means model:")
    print_topic_words(kmeans, tfidf_vectorizer, n_top_words)
    
    mbkmeans = skcluster.MiniBatchKMeans(n_clusters=n_topics, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
    mbkmeans.fit(tfidf_matrix)
    print("\nTopics in mini batch k-means model:")
    print_topic_words(mbkmeans, tfidf_vectorizer, n_top_words)
    '''
    #lda_topics = get_topic_words(model, vectorizer, n_top_words)
    
    #return topic_keywords

if __name__ == "__main__":
    

    n_features = 1000
    n_topics = 3
    n_top_words = 30
    
    #data = fetch_dataset(n_samples)
    #lang = "en"
    #extract_topics(data, lang, n_features, n_topics, n_top_words)
    
    
    lang = "tr"
    # tcell
    folder = "/home/dicle/Documents/experiments/tcell_topics/docs_less"
    fnames = io_utils.getfilenames_of_dir(folder, removeextension=False)
    docs = []
    for fname in fnames: 
        path = os.path.join(folder, fname)
        text = ""
        with open(path, "r") as f:
            text = f.read().strip()
        docs.append((fname, text))
    
    instances = [t for _, t in docs]
    extract_topics(instances, lang, n_features, n_topics, n_top_words)
    
    #===========================================================================
    # 
    # '''path = "/home/dicle/Documents/data/data1.csv"  # CHANGE
    # delimiter = ";"'''
    # '''
    # instances, _ = corpus_io.get_csv_data(path="/home/dicle/Documents/data/ttnet_email/PROBSUMMARYM1.csv", 
    #                                               delimiter=",")
    # '''
    # # 5K email 
    # folderpath = "/home/dicle/Documents/data/emailset2"
    # fname = "has_pstn2.csv"    #_nodupl_less.csv"   #"Raw_Email_Data-OriginalSender.csv"
    # more_stopwords = stopword_lists.email_specific_stopwords()
    # 
    # '''
    # folderpath = "/home/dicle/Documents/data/emailset2/set2_eray"
    # fname = "joint_5K_37K.csv"
    # '''
    # cat_col = "TIP"   #"TIP"
    # instances, labels = corpus_io.read_labelled_texts_csv(os.path.join(folderpath, fname), 
    #                                          sep=";", textcol="MAIL", catcol=cat_col)
    # '''
    # N = 100
    # instances, _ = corpus_io.select_N_instances(N, instances, labels)
    # '''
    # 
    # extract_topics(instances, lang, n_features, n_topics, n_top_words, more_stopwords)
    # 
    # 
    #===========================================================================
    
