'''
Created on Jan 4, 2017

@author: dicle
'''

import os

from time import time

import sklearn.feature_extraction.text as txtfeatext
import sklearn.cluster as skcluster
import sklearn.decomposition as decomposer
import sklearn.pipeline as skpipeline
import sklearn.preprocessing as skprep


from modules.learning.language_tools import TokenHandler, stopword_lists
from modules.learning.dataset import corpus_io


'''
Adapted from: http://scikit-learn.org/stable/auto_examples/text/document_clustering.html

'''



def print_top_N_words(clusterer, vectorizer, nclusters, top_N_words):
    

    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(nclusters):
        print("Cluster %d:" % i)
        
        print(", ".join([terms[ind] for ind in order_centroids[i, : top_N_words]]))


def get_top_N_words(clusterer, vectorizer, nclusters, top_N_words):
    
    clwords = dict.fromkeys(list(range(nclusters)), [])
    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    print(clusterer.cluster_centers_)
    print(clusterer.cluster_centers_.shape)
    for i in range(nclusters):
        
        words = [terms[ind] for ind in order_centroids[i, : top_N_words]]
        clwords[i] = words
    
    return clwords

def get_top_N_words_with_weights(clusterer, vectorizer, nclusters, top_N_words):
    
    clwords = dict.fromkeys(list(range(nclusters)), [])
    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    print(clusterer.cluster_centers_)
    print(clusterer.cluster_centers_.shape)
    for i in range(nclusters):
        
        words = [(terms[ind], clusterer.cluster_centers_[i, ind]) for ind in order_centroids[i, : top_N_words]]
        clwords[i] = words
    
    return clwords

def get_cluster_member_counts(clusterer, true_labels):

       
    cluster_labels = clusterer.labels_  # cluster labels per instance
    cats = list(set(true_labels))
    
    catdict = {}  # { cat_i : { cluster_i : n_members, }, }
    
    
    for cat in cats:
        clusterdict = {}
        for cluster in cluster_labels:
            clusterdict[cluster] = 0
        catdict[cat] = clusterdict
    
    for i in range(len(true_labels)):
        cat = true_labels[i]
        cluster = cluster_labels[i]
        
        catdict[cat][cluster] = catdict[cat][cluster] + 1
    
    return catdict



def get_cluster_members(clusterer, true_labels):

       
    cluster_labels = clusterer.labels_  # cluster labels per instance
    cats = list(set(true_labels))
    clusters = list(set(cluster_labels))
    
    cldict = {}
    for clabel in clusters:
        cldict[clabel] = []
          
    
    for truelabel, clusterlabel in zip(true_labels, cluster_labels):
        
        cldict[clusterlabel] = cldict[clusterlabel] + [truelabel]
        
    
    return cldict

def extract_topics(n_clusters, instances, lang, more_stopwords, reduce_dim=True):
    
    
    n_gram_range = (1, 2)
    n_max_features = None
    top_N_words = 30
    
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
                                          max_features=n_max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(instances)
    
    t1 = time()
    print("1- Vectorizing took ", (t1-t0), "sec.")
    
    
    #  @TODO: dim.reduce.
    if reduce_dim:
        
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = decomposer.TruncatedSVD(2000)
        normalizer = skprep.Normalizer(copy=False)
        lsa = skpipeline.make_pipeline(svd, normalizer)
    
        tfidf_matrix = lsa.fit_transform(tfidf_matrix)
    
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(tfidf_matrix)
    
    t2 = time()
    print("kmeans clustering took ", t2-t1, "sec.")
    print("\nTopics in k-means model:")
    print_top_N_words(kmeans, tfidf_vectorizer, n_clusters, top_N_words)
    
    
    '''
    spectral = skcluster.SpectralClustering(n_clusters=n_clusters)
    spectral.fit(tfidf_matrix)
    t3 = time()
    print("spectral clustering took", t3-t2, "sec.")
    print("\nTopics in the spectral model:")
    print_top_N_words(spectral, tfidf_vectorizer, n_clusters, top_N_words)
    '''
    
    '''
    mbkmeans = skcluster.MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
    mbkmeans.fit(tfidf_matrix)
    print("\nTopics in mini batch k-means model:")
    print_top_N_words(mbkmeans, tfidf_vectorizer, n_clusters, top_N_words)
    '''
    
    return kmeans



def extract_topics_kmeans(n_clusters, instances, preprocessor, 
                          n_max_features = None, 
                          top_N_words = 30,
                          n_gram_range=(1, 1), 
                          more_stopwords=None, reduce_dim=True):
    
    
    
    
    t0 = time()


    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)

    tfidf_matrix = tfidf_vectorizer.fit_transform(instances)
    
    t1 = time()
    
    
    #  @TODO: dim.reduce.
    if reduce_dim:
        
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        _, N = tfidf_matrix.shape
        N = (N / 10) + 1
        svd = decomposer.TruncatedSVD(N)
        normalizer = skprep.Normalizer(copy=False)
        lsa = skpipeline.make_pipeline(svd, normalizer)
    
        tfidf_matrix = lsa.fit_transform(tfidf_matrix)
    
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='random', max_iter=100, n_init=1)
    kmeans.fit(tfidf_matrix)
    
    t2 = time()
    #print("kmeans clustering took ", t2-t1, "sec.")
    #print("\nTopics in k-means model:")
    #print_top_N_words(kmeans, tfidf_vectorizer, n_clusters, top_N_words)
    
    #print(tfidf_matrix.shape)
    #print(kmeans.cluster_centers_.shape)
    
    #return kmeans, tfidf_vectorizer
    topics_words = get_top_N_words(kmeans, tfidf_vectorizer, n_clusters, top_N_words)
    return topics_words


def cluster_emails():
    

    lang = "tr"
    '''path = "/home/dicle/Documents/data/data1.csv"  # CHANGE
    delimiter = ";"'''
    '''
    instances, _ = corpus_io.get_csv_data(path="/home/dicle/Documents/data/ttnet_email/PROBSUMMARYM1.csv", 
                                                  delimiter=",")
    '''
    # 5K email 
    folderpath = "/home/dicle/Documents/data/emailset2"
    fname = "has_pstn2.csv"    #_nodupl_less.csv"   #"Raw_Email_Data-OriginalSender.csv"
    more_stopwords = stopword_lists.email_specific_stopwords()
    
    '''
    folderpath = "/home/dicle/Documents/data/emailset2/set2_eray"
    fname = "joint_5K_37K.csv"
    '''
    cat_col = "TIP"   #"TIP"
    instances, labels = corpus_io.read_labelled_texts_csv(os.path.join(folderpath, fname), 
                                             sep=";", textcol="MAIL", catcol=cat_col)
    
    
    N = 200
    instances, labels = corpus_io.select_N_instances(N, instances, labels)
    
    
    n_clusters = 3
    
    clusterer = extract_topics(n_clusters, instances, lang, more_stopwords, reduce_dim=False)
    cat_cluster_dict = get_cluster_member_counts(clusterer, labels)
    print(cat_cluster_dict)
    



if __name__ == '__main__':

    cluster_emails()
    print()

  





