'''
Created on Jan 4, 2017

@author: dicle
'''

from time import time
import os
import pickle

import numpy as np

import sklearn.cluster as skcluster

from modules.learning.language_tools import TokenHandler
from modules.learning.dataset import corpus_io

def load_bin_vec(fpath, vocab):
    """
    Loads 200x1 word vecs from boun_twitter_embeddings produced via word2vec
    """
    word_vecs = {}
    

    with open(fpath, "rb") as f:
        for line in f:
            
            items = line.split()
            word = items[0]
            vect = items[1:]
            vect = np.array(vect, dtype='float32')
            if word in vocab:
                word_vecs[word] = vect
    
    print("loaded word vectors.")
    return word_vecs




def add_unknown_words(word_vecs, vocab, k): #, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        #if word not in word_vecs and vocab[word] >= min_df:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 
    
    print("added vects for the unknown words.")       
            

def get_vocabulary(texts):
    """
    Loads data 
    """
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=False,  
                                                   stemming=False, 
                                                   remove_numbers=True,
                                                   deasciify=True, remove_punkt=True)
    tokens = []
    for text in texts:
        currentokens = preprocessor(text)
        currenttokens = list(set(currentokens))
        tokens.extend(currenttokens)
    
    tokens = list(set(tokens))
  
    print("loaded the dataset.  ntokens: ", str(len(tokens)))
    return tokens
            

def save_vectors(word_vec, vocab, k, dumppath):
    
    pickle.dump([word_vec, vocab, k], open(dumppath, "wb"))
    print("recorded the word vectors.")


def record_dataset_vects():
    
    # load the dataset
    # 5K email 
    folderpath = "/home/dicle/Documents/data/emailset2"
    fname = "has_pstn2.csv"    #_nodupl_less.csv"   #"Raw_Email_Data-OriginalSender.csv"
    
    cat_col = "TIP"   #"TIP"
    instances, labels = corpus_io.read_labelled_texts_csv(os.path.join(folderpath, fname), 
                                             sep=";", textcol="MAIL", catcol=cat_col)
    
    N = 200
    instances, labels = corpus_io.select_N_instances(N, instances, labels)
    
    vocabulary = get_vocabulary(instances)
    
    
    
    # load embeddings of the vocabulary of this dataset
    
    '''
    1- boun embeddings
    2- sabancı
    3- polyglot
    '''
    
    ''' # boun '''
    embeddings_path = "/home/dicle/Documents/tools/tr_embeddings/bounweb+tweetscorpus_twitterprocessed_vectors_lowercase_w5_dim200_fixed.txt"
    k = 200
   
    '''  # sabanci 
    embeddings_path = "/home/dicle/Documents/tools/tr_embeddings/vectors.8.txt"
    k = 100
    '''
    
    word_vecs = load_bin_vec(embeddings_path, vocabulary)

    add_unknown_words(word_vecs, vocabulary, k)

    dumppath = "/home/dicle/Documents/data/emailset2/clustering/embeddings/dump200-200.b"
    save_vectors(word_vecs, vocabulary, k, dumppath)

##############################################################


def get_cluster_members(clusterer, words):

    cluster_labels = clusterer.labels_
    clusters = list(set(cluster_labels))
    
    cldict = {}
    for c  in clusters:
        cldict[c] = []
        
    for i in range(len(cluster_labels)):
        cldict[cluster_labels[i]].append(words[i])
    
    return cldict

def print_cluster_members(cldict):
    
    for cl, words in cldict.items():
        
        print("Cluster ", cl)
        print(", ".join(words))
        
        
         
# word_vec : { word : []}
def clustering(word_vecs, n_clusters):
    
    words = list(word_vecs.keys())
    matrix = []
    for _, vect in word_vecs.items():
        matrix.append(vect)
        
    
    t1 = time()
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(matrix)
    
    t2 = time()
    print("kmeans clustering took ", t2-t1, "sec.")
    print("\nTopics by k-means model:")
    cluster_words = get_cluster_members(kmeans, words) 
    print_cluster_members(cluster_words)



def load_dataset_vect():
    
    dpath = "/home/dicle/Documents/data/emailset2/clustering/embeddings/dump200-200.b"
    f = open(dpath, "rb")  #.read()
    word_vecs, vocabulary, k = pickle.load(f)
    
    n_clusters = 3
    clustering(word_vecs, n_clusters)
    

if __name__ == '__main__':
    
    load_dataset_vect()
    
    
    

