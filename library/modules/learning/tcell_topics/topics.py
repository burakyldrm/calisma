'''
Created on Feb 15, 2017

@author: dicle
'''
import os

import numpy as np
import pickle
from time import time

import sklearn.preprocessing as skprep
import sklearn.pipeline as skpipe
import sklearn.feature_extraction.text as txtfeatext
import sklearn.cluster as skcluster
import sklearn.decomposition as decomposer
import sklearn.pipeline as skpipeline

from language_tools import TokenHandler
import keyword_extraction.topic_extraction_decompose as clu1
import keyword_extraction.topic_extraction_cluster as clu2
import keyword_extraction.comparison_topic_extraction as kext
from dataset import io_utils


def get_topics_decomposed(instances):
    
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    n_features = 1000
    n_topics = 3
    n_top_words = 30
    
    clu1.extract_topics_decomposed(instances, preprocessor, n_features, n_topics, n_top_words, 
                                   n_gram_range=(1,1), more_stopwords=None)
    



def get_topics_kmeans(instances, labels, sentence):
    
    output = {"word" : "", "nearest_docs" : [], "nearest_terms" : []}
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    n_clusters = 3
    n_max_features = None
    top_N_words = 30
    n_gram_range=(1, 1)
    more_stopwords=None
    reduce_dim=False
    
        
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='random', max_iter=1000, n_init=10,
                              random_state=42)
    
    pipeline = skpipe.Pipeline([('vect', tfidf_vectorizer),
                                ('normalizer', skprep.Normalizer()),
                                ('clusterer', kmeans)])
    
        
    
    ''' Prediction on input words 
         - assign each word to one of the clusters formed by the database
           - print the closest terms and docs for each input word
         - find the word closest to its own cluster center, assign it to be highlighted
           - if the total distance of the input words exceeds SOME THRESHOLD, then that input is irrelevant and will not be highlighted
           
        + map the preprocessed word to its original
        + handle empty or after-preprocessing empty input
    '''
    data_distances = pipeline.fit_transform(instances)
    data_clusters = kmeans.labels_
    clnames = list(set(data_clusters))
    
    words = preprocessor.__call__(sentence)
    words = list(set(words))   # remove repeating words
    if(len(words) < 1 or len("".join(words)) < 1):
        return None
    
    #words = [sentence]
    input_clusters = pipeline.predict(words)
    input_distances = pipeline.transform(words)
    
    
    '''
     store the ids and distances of the members for each cluster
      - {clNO : [(member_id, member_distance)]}
    '''
    cluster_members = dict.fromkeys(clnames, [])
    for memberid, memberclNo in enumerate(data_clusters):
        distance = data_distances[memberid, memberclNo]
        cluster_members[memberclNo] = cluster_members[memberclNo] + [(memberid, distance)]
    
    
    input_cluster_members = []   # store [(input_id, input_clNO, input_distance)]
    for inputid, inputclNo in enumerate(input_clusters):
        distance = input_distances[inputid, inputclNo]
        input_cluster_members.append((inputid, inputclNo, distance))
    
    print(words)
    print(input_distances)
    '''
     the most important word -> the closest to its cluster center
    '''
    l = sorted(input_cluster_members, key=lambda x : x[2])    
    hwordid, hclusterNo, hdist = l[0]
    highlight_word = words[hwordid]
    
    
    #the validity of the most important word. check its distance with the doc farthest in the cluster
    # the farthest instance
    highlight_cluster_members = cluster_members[hclusterNo] 
    l = sorted(highlight_cluster_members, key=lambda x : x[1], reverse=True)  
    _, farthestDocDist = l[0]
    print(highlight_word, " dist: ", hdist, " farthest dist: ", farthestDocDist)
    # distance to the closest words
    word_distances = kmeans.cluster_centers_   # n_clusters X n_database_words
    cl_word_distances = word_distances[hclusterNo, :]
    print(cl_word_distances.shape)
    word_max_dist = cl_word_distances[0]
    print("word_max_dist: ", word_max_dist)
    print("word_min_dist: ", cl_word_distances[-1])
    cl_word_distances = cl_word_distances[::-1]
    word_avg_dist = cl_word_distances.mean()
    print("word_avg_dist: ", word_avg_dist)
  
    cluster_terms = clu2.get_top_N_words(kmeans, tfidf_vectorizer, nclusters=n_clusters, top_N_words=top_N_words)
    hclusterterms = cluster_terms[hclusterNo]
    
    if highlight_word not in hclusterterms:    # @TODO find a real threshold!
        return None
    
    
    cluster_membernames = clu2.get_cluster_members(kmeans, labels)
    hclustermembers = cluster_membernames[hclusterNo]
    
    # find the original word
    preprocessed_word_map = TokenHandler.original_to_preprocessed_map(preprocessor, sentence)
    original_words = preprocessed_word_map[highlight_word]
    
    
    output["word"] = original_words
    output["nearest_terms"] = hclusterterms
    output["nearest_docs"] = hclustermembers
    return output
      

#===============================================================================
# 
# # not working!!
# def get_topics_affinity_prop(instances, labels, sentence):
#     
#     output = {"word" : "", "nearest_docs" : [], "nearest_terms" : []}
#     
#     preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
#                                                    stemming=True, 
#                                                    remove_numbers=True,
#                                                    deasciify=False, remove_punkt=True)
#     
#     n_max_features = None
#     top_N_words = 30
#     n_gram_range=(1, 1)
#     more_stopwords=None
#     reduce_dim=False
#     
#         
#     tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
#                                           ngram_range=n_gram_range,
#                                           max_features=n_max_features)
#     model = skcluster.AffinityPropagation()
#     
#     pipeline = skpipe.Pipeline([('vect', tfidf_vectorizer),
#                                 ('normalizer', skprep.Normalizer()),
#                                 ('clusterer', model)])
#     
#         
#     
#     ''' Prediction on input words 
#          - assign each word to one of the clusters formed by the database
#            - print the closest terms and docs for each input word
#          - find the word closest to its own cluster center, assign it to be highlighted
#            - if the total distance of the input words exceeds SOME THRESHOLD, then that input is irrelevant and will not be highlighted
#            
#         + map the preprocessed word to its original
#         + handle empty or after-preprocessing empty input
#     '''
#     pipeline.fit(instances)
#     affinity_matrix = model.affinity_matrix_
#     data_clusters = model.labels_
#     clnames = list(set(data_clusters))
#     
#     print("# clusters: ", len(clnames))
#     
#     words = preprocessor.__call__(sentence)
#     if(len(words) < 1 or len("".join(words)) < 1):
#         return None
#     
#     #words = [sentence]
#     input_distances = pipeline.predict(words)
#     
#     
#     '''
#      store the ids and distances of the members for each cluster
#       - {clNO : [(member_id, member_distance)]}
#     '''
#     '''
#     cluster_members = dict.fromkeys(clnames, [])
#     for memberid, memberclNo in enumerate(data_clusters):
#         distance = data_distances[memberid, memberclNo]
#         cluster_members[memberclNo] = cluster_members[memberclNo] + [(memberid, distance)]
#     
#     
#     input_cluster_members = []   # store [(input_id, input_clNO, input_distance)]
#     for inputid, inputclNo in enumerate(input_clusters):
#         distance = input_distances[inputid, inputclNo]
#         input_cluster_members.append((inputid, inputclNo, distance))
#     '''
#     
#     '''
#      the most important word -> the closest to its cluster center
#     '''
#     l = sorted(input_cluster_members, key=lambda x : x[2])    
#     hwordid, hclusterNo, hdist = l[0]
#     highlight_word = words[hwordid]
#     
#     
#     #the validity of the most important word. check its distance with the doc farthest in the cluster
#     # the farthest instance
#     highlight_cluster_members = cluster_members[hclusterNo] 
#     l = sorted(highlight_cluster_members, key=lambda x : x[1], reverse=True)  
#     _, farthestDocDist = l[0]
#     print(highlight_word, " dist: ", hdist, " farthest dist: ", farthestDocDist)
#     # distance to the closest words
#     word_distances = model.cluster_centers_   # n_clusters X n_database_words
#     cl_word_distances = word_distances[hclusterNo, :]
#     print(cl_word_distances.shape)
#     word_max_dist = cl_word_distances[0]
#     print("word_max_dist: ", word_max_dist)
#     print("word_min_dist: ", cl_word_distances[-1])
#     cl_word_distances = cl_word_distances[::-1]
#     word_avg_dist = cl_word_distances.mean()
#     print("word_avg_dist: ", word_avg_dist)
#   
#     cluster_terms = clu2.get_top_N_words(model, tfidf_vectorizer, nclusters=len(clnames), top_N_words=top_N_words)
#     hclusterterms = cluster_terms[hclusterNo]
#     
#     if highlight_word not in hclusterterms:    # @TODO find a real threshold!
#         return None
#     
#     
#     cluster_membernames = clu2.get_cluster_members(model, labels)
#     hclustermembers = cluster_membernames[hclusterNo]
#     
#     # find the original word
#     preprocessed_word_map = TokenHandler.original_to_preprocessed_map(preprocessor, sentence)
#     original_words = preprocessed_word_map[highlight_word]
#     
#     
#     output["word"] = original_words
#     output["nearest_terms"] = hclusterterms
#     output["nearest_docs"] = hclustermembers
#     return output
#       
#===============================================================================


def build_kmeans_model(instances, labels, path_to_model_dump):
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    n_clusters = 3
    n_max_features = None
    
    n_gram_range=(1, 1)
    more_stopwords=None
    reduce_dim=False
    
        
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)
    normalizer = skprep.Normalizer()
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='random', max_iter=1000, n_init=10,
                              random_state=42)
    
    pipeline = skpipe.Pipeline([('vect', tfidf_vectorizer),
                                #('normalizer', normalizer),
                                ('scaler', skprep.StandardScaler()),
                                ('clusterer', kmeans)])
    
    if path_to_model_dump:
        pickle.dump([pipeline, preprocessor, kmeans, tfidf_vectorizer], open(path_to_model_dump, "wb"))
    
    return pipeline, preprocessor, kmeans, tfidf_vectorizer




def get_topics_kmeans2(path_to_model_dump, sentence):
    
    pipeline, preprocessor, kmeans, tfidf_vectorizer = pickle.load(open(path_to_model_dump, "rb"))
    top_N_words = 30
    
    output = {"word" : "", "nearest_docs" : [], "nearest_terms" : []}
    
    
    
        
    
    ''' Prediction on input words 
         - assign each word to one of the clusters formed by the database
           - print the closest terms and docs for each input word
         - find the word closest to its own cluster center, assign it to be highlighted
           - if the total distance of the input words exceeds SOME THRESHOLD, then that input is irrelevant and will not be highlighted
           
        + map the preprocessed word to its original
        + handle empty or after-preprocessing empty input
    '''
    data_distances = pipeline.fit_transform(instances)
    data_clusters = kmeans.labels_
    clnames = list(set(data_clusters))
    
    words = preprocessor.__call__(sentence)
    if(len(words) < 1 or len("".join(words)) < 1):
        return None
    
    #words = [sentence]
    input_clusters = pipeline.predict(words)
    input_distances = pipeline.transform(words)
    
    
    '''
     store the ids and distances of the members for each cluster
      - {clNO : [(member_id, member_distance)]}
    '''
    cluster_members = dict.fromkeys(clnames, [])
    for memberid, memberclNo in enumerate(data_clusters):
        distance = data_distances[memberid, memberclNo]
        cluster_members[memberclNo] = cluster_members[memberclNo] + [(memberid, distance)]
    
    
    input_cluster_members = []   # store [(input_id, input_clNO, input_distance)]
    for inputid, inputclNo in enumerate(input_clusters):
        distance = input_distances[inputid, inputclNo]
        input_cluster_members.append((inputid, inputclNo, distance))
    
    
    '''
     the most important word -> the closest to its cluster center
    '''
    l = sorted(input_cluster_members, key=lambda x : x[2])    
    hwordid, hclusterNo, hdist = l[0]
    highlight_word = words[hwordid]
    
    '''
     #the validity of the most important word. check its distance with the doc farthest in the cluster
    
    highlight_cluster_members = cluster_members[hclusterNo] 
    l = sorted(highlight_cluster_members, key=lambda x : x[1], reverse=True)  
    _, farthestDocDist = l[0]
    '''
  
    cluster_terms = clu2.get_top_N_words(kmeans, tfidf_vectorizer, nclusters=len(clnames), top_N_words=top_N_words)
    hclusterterms = cluster_terms[hclusterNo]
    
    if highlight_word not in hclusterterms:    # @TODO find a real threshold!
        return None
    
    
    cluster_membernames = clu2.get_cluster_members(kmeans, labels)
    hclustermembers = cluster_membernames[hclusterNo]
    
    # find the original word
    preprocessed_word_map = TokenHandler.original_to_preprocessed_map(preprocessor, sentence)
    original_words = preprocessed_word_map[highlight_word]
    
    
    output["word"] = original_words
    output["nearest_terms"] = hclusterterms
    output["nearest_docs"] = hclustermembers
    return output
      


def _get_topics_kmeans(instances, labels, sentence):
    
    output = {"word" : "", "nearest_docs" : [], "nearest_terms" : []}
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    n_clusters = 3
    n_max_features = None
    top_N_words = 30
    n_gram_range=(1, 1)
    more_stopwords=None
    reduce_dim=False
    
        
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='random', max_iter=1000, n_init=10,
                              random_state=42)
    
    pipeline = skpipe.Pipeline([('vect', tfidf_vectorizer),
                                ('normalizer', skprep.Normalizer()),
                                ('clusterer', kmeans)])
    
        
    
    ''' Prediction on input words 
         - assign each word to one of the clusters formed by the database
           - print the closest terms and docs for each input word
         - find the word closest to its own cluster center, assign it to be highlighted
           - if the total distance of the input words exceeds SOME THRESHOLD, then that input is irrelevant and will not be highlighted
           
        + map the preprocessed word to its original
        + handle empty or after-preprocessing empty input
    '''
    data_distances = pipeline.fit_transform(instances)
    data_clusters = kmeans.labels_
    clnames = list(set(data_clusters))
    
    words = preprocessor.__call__(sentence)
    if(len(words) < 1 or len("".join(words)) < 1):
        return None
    
    #words = [sentence]
    input_clusters = pipeline.predict(words)
    input_distances = pipeline.transform(words)
    
    
    '''
     store the ids and distances of the members for each cluster
      - {clNO : [(member_id, member_distance)]}
    '''
    print(clnames, data_clusters)
    cluster_members = dict.fromkeys(clnames, [])
    for memberid, memberclNo in enumerate(data_clusters):
        distance = data_distances[memberid, memberclNo]
        cluster_members[memberclNo] = cluster_members[memberclNo] + [(memberid, distance)]
    
    
    input_cluster_members = []   # store [(input_id, input_clNO, input_distance)]
    for inputid, inputclNo in enumerate(input_clusters):
        distance = input_distances[inputid, inputclNo]
        input_cluster_members.append((inputid, inputclNo, distance))
    
    
    '''
     the most important word -> the closest to its cluster center
    '''
    l = sorted(input_cluster_members, key=lambda x : x[2])    
    wordid, clNo, dist = l[0]
    highlight_word = words[wordid]
    print("highlight_item ", highlight_word, dist)
    '''
     the validity of the most important word. check its distance with the doc farthest in the cluster
    '''
    highlight_cluster_members = cluster_members[clNo] 
    l = sorted(highlight_cluster_members, key=lambda x : x[1], reverse=True)  
    _, farthestDocDist = l[0]
    print("word: ", dist, " farthest's dist: ", farthestDocDist)
    dists = [d for _,d in l]
    avg_dist = sum(dists) / len(dists)
    print("avg dist: ", avg_dist)
    
    print("matrix ", data_distances.shape)
    print(data_distances)
    print("sentence vect ", input_distances.shape)
    print(input_distances)
    print("centers ", kmeans.cluster_centers_.shape)
    print("threshold ", kmeans.inertia_ / len(instances))
    print(kmeans.inertia_, "  ", np.sum(data_distances, axis=0).sum())
    print(kmeans.labels_)
    print(cluster_members)
   
    cluster_terms = clu2.get_top_N_words(kmeans, tfidf_vectorizer, nclusters=n_clusters, top_N_words=top_N_words)
    for cl, terms in cluster_terms.items():
        print(cl, ":", ", ".join(terms) )
    cluster_members = clu2.get_cluster_members(kmeans, labels)
    for cl, docs in cluster_members.items():
        print(cl, ": ", docs)
    
    # get important term of the input
    results = []  # will store (input_word, clusterNo, num of occurrences in the words of its cluster)
    for word, cl in zip(words, input_clusters):
        neighbour_terms = cluster_terms[cl]
        occ = neighbour_terms.count(word)   # try PMI
        print(word, " -> in cluster ", cl, ", ", occ, " times.")
        print(" related docs: ", ", ".join(cluster_members[cl]))
        print(" related words: ", ", ".join(neighbour_terms))
        results.append((word, cl, occ))
    
        
    
    '''
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    print(order_centroids)
    '''
    
    
if __name__ == "__main__":
    
    # build clustering model 
    lang = "tr"
    # tcell
    folder = "/home/dicle/Documents/experiments/tcell_topics/docs_less"
    fnames = io_utils.getfilenames_of_dir(folder, removeextension=False)
    instances = []
    labels = []
    for fname in fnames: 
        path = os.path.join(folder, fname)
        text = ""
        with open(path, "r") as f:
            text = f.read().strip()
        instances.append(text)
        labels.append(fname)


    sentence = "hattımdaki paket değişikliklerini ve paketleri nasıl öğrenebilirim?"
    #sentence = "bugün hava ne güzel"
    t0 = time()
    output = get_topics_kmeans(instances, labels, sentence)
    '''
    path_to_model = "/home/dicle/Documents/experiments/tcell_topics/models/tcell_kmeans_model.b"
    #build_kmeans_model(instances, labels, path_to_model)
    output = get_topics_kmeans2(path_to_model, sentence)
    '''
    
    print("Sentence: ", sentence)
    if output:
        print(output)
    
    t1 = time()
    duration = round(t1 - t0, 3)
    print("Duration: ", duration, "sn.")

    '''
    sentence = "bugün hava ne güzel"
    
    
    t0 = time()
    
    output = get_topics_kmeans(instances, labels, sentence)
    print("Sentence: ", sentence)
    if output:
        print(output)

    t1 = time()
    duration = round(t1 - t0, 3)
    print("Duration: ", duration, "sn.")
    '''
    
    '''
    # rake
    doc_topphrases_rake = [kext.score_keyphrases_by_rake(text, lang) for text in instances]
    kext.print_top_phrases(doc_topphrases_rake, print_weight=True)
    '''
    
    
    
    
