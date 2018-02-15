'''
Created on Apr 6, 2017

@author: dicle
'''

import os, json
import re
import numpy as np

import sklearn.feature_extraction.text as txtfeatext
import sklearn.preprocessing as skprep
from sklearn.decomposition import TruncatedSVD
import sklearn.pipeline as skpipeline
import sklearn.cluster as skcluster

import keyword_extraction.topic_extraction_cluster as clustering
from misc import list_utils
import text_categorization.prototypes.text_preprocessor as prep
from dataset import io_utils
from sklearn.externals import joblib




def train_kmeans_model(instances, labels, n_clusters=3):

    n_max_features = None
    n_gram_range=(1, 1)
    more_stopwords=None
    reduce_dim=False

    preprocessor = prep.Preprocessor(lang="tr", stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)

    
        
  
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='random', max_iter=1000, n_init=10,
                              random_state=42)
    
    pipeline = skpipeline.Pipeline([('preprocessor', preprocessor),
                                    ('vect', tfidf_vectorizer),
                                    ('normalizer', skprep.Normalizer()),
                                    ('clusterer', kmeans)])
    
    data_distances = pipeline.fit_transform(instances)
    
    return pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, instances, labels



def train_and_dump_kmeans_model(instances, labels, n_clusters,
                                picklepath):

    pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, _, _ = train_kmeans_model(instances, labels, n_clusters)

    joblib.dump([pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, instances, labels],
                picklepath)


def load_kmeans_model(picklepath):

    pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, instances, labels = joblib.load(picklepath)
    '''
    pipeline = model_items[0]
    data_distances = model_items[1]
    preprocessor = model_items[2]
    tfidf_vectorizer = model_items[3]
    kmeans = model_items[]
    instances = model_items[]
    labels = model_items[] 
    '''
    return pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, instances, labels

def online_kmeans_topics(instances, labels, sentence, n_clusters, top_N_words):
    
    pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, _, _ = train_kmeans_model(instances, labels, n_clusters)
    
    result = _topics_kmeans(pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, 
                            instances, labels, 
                            sentence,
                            n_clusters, top_N_words)
    
    return result

def offline_kmeans_topics(picklepath, sentence, n_clusters, top_N_words):
    
    pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, instances, labels = load_kmeans_model(picklepath)
    
    result = _topics_kmeans(pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans, 
                            instances, labels, 
                            sentence,
                            n_clusters, top_N_words) 

    return result

def _topics_kmeans(pipeline, data_distances, preprocessor, tfidf_vectorizer, kmeans,
                   instances, labels,
                   sentence,
                   n_clusters, top_N_words):
    
    output = {"word" : "", "nearest_docs" : [], "nearest_terms" : []}
    
     
        
    
    ''' Prediction on input words 
         - assign each word to one of the clusters formed by the database
           - print the closest terms and docs for each input word
         - find the word closest to its own cluster center, assign it to be highlighted
           - if the total distance of the input words exceeds SOME THRESHOLD, then that input is irrelevant and will not be highlighted
           
        + map the preprocessed word to its original
        + handle empty or after-preprocessing empty input
    '''
    
    data_clusters = kmeans.labels_
    clnames = list(set(data_clusters))
    
    
    words = preprocessor.tokenize(sentence)
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
    
    #print(words)
    #print(input_distances)
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
    #print(highlight_word, " dist: ", hdist, " farthest dist: ", farthestDocDist)
    # distance to the closest words
    word_distances = kmeans.cluster_centers_   # n_clusters X n_database_words
    cl_word_distances = word_distances[hclusterNo, :]
    #print(cl_word_distances.shape)
    word_max_dist = cl_word_distances[0]
    #print("word_max_dist: ", word_max_dist)
    #print("word_min_dist: ", cl_word_distances[-1])
    cl_word_distances = cl_word_distances[::-1]
    word_avg_dist = cl_word_distances.mean()
    #print("word_avg_dist: ", word_avg_dist)
  
    cluster_terms = clustering.get_top_N_words(kmeans, tfidf_vectorizer, nclusters=n_clusters, top_N_words=top_N_words)
    cluster_terms_w = clustering.get_top_N_words_with_weights(kmeans, tfidf_vectorizer, nclusters=n_clusters, top_N_words=top_N_words)
    print(cluster_terms_w)
    hclusterterms = cluster_terms[hclusterNo]
    
    if highlight_word not in hclusterterms:    # @TODO find a real threshold!
        return None
    
    
    cluster_membernames = clustering.get_cluster_members(kmeans, labels)
    hclustermembers = cluster_membernames[hclusterNo]
    
    # find the original word
    preprocessed_word_map = prep.original_to_preprocessed_map(preprocessor, sentence)
    original_words = preprocessed_word_map[highlight_word]
    
    
    output["word"] = original_words
    output["nearest_terms"] = hclusterterms
    output["nearest_docs"] = hclustermembers
    return output


def topics_kmeans(instances, labels, sentence):
    
    output = {"word" : "", "nearest_docs" : [], "nearest_terms" : []}
    
    
    n_clusters = 3
    n_max_features = None
    top_N_words = 30
    n_gram_range=(1, 1)
    more_stopwords=None
    reduce_dim=False
     
    preprocessor = prep.Preprocessor(lang="tr", stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)

    
        
  
    kmeans = skcluster.KMeans(n_clusters=n_clusters, init='random', max_iter=1000, n_init=10,
                              random_state=42)
    
    pipeline = skpipeline.Pipeline([('preprocessor', preprocessor),
                                    ('vect', tfidf_vectorizer),
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
    
    words = preprocessor.tokenize(sentence)
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
  
    cluster_terms = clustering.get_top_N_words(kmeans, tfidf_vectorizer, nclusters=n_clusters, top_N_words=top_N_words)
    hclusterterms = cluster_terms[hclusterNo]
    
    if highlight_word not in hclusterterms:    # @TODO find a real threshold!
        return None
    
    
    cluster_membernames = clustering.get_cluster_members(kmeans, labels)
    hclustermembers = cluster_membernames[hclusterNo]
    
    # find the original word
    preprocessed_word_map = prep.original_to_preprocessed_map(preprocessor, sentence)
    original_words = preprocessed_word_map[highlight_word]
    
    
    output["word"] = original_words
    output["nearest_terms"] = hclusterterms
    output["nearest_docs"] = hclustermembers
    return output
      

# database: instances, labels
# topics extracted, then the important words and close documents of the input sentence is detected based on cosine sim..
def topics_lsi(instances, labels, sentence, 
                 ndim=5,
                 n_gram_range=(1,1),
                 n_max_features=None):
    
    print(instances[:5])
    print(labels[:5])
    
    
    highlight_word = ""
    
    
    
    
    preprocessor = prep.Preprocessor(lang="tr", stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)
    
    svd_model = TruncatedSVD(n_components=ndim,
                         algorithm='randomized',
                         n_iter=10, random_state=42)
    
    svd_transformer = skpipeline.Pipeline([('preprocessor', preprocessor),        
                                           ('vectorizer', tfidf_vectorizer),
                                           #('normalizer', skprep.Normalizer()),
                                           ('scaler', skprep.StandardScaler(with_mean=False)),
                                           ('svd', svd_model)])

    docmatrix = svd_transformer.fit_transform(instances)
    
    


    input_ = preprocessor.tokenize(sentence)
    if(len(input_) < 1 or len("".join(input_)) < 1):
        highlight_word = ""
        return highlight_word
    
    
    inputmatrix = svd_transformer.transform(input_)    
     
    termmatrix = svd_model.components_.T
    print(termmatrix.shape)
    print(inputmatrix.shape)
    print(docmatrix.shape)


    # closest docs
    # @TODO different similarity metrics
    docsim, docindices = list_utils.matrix_similarity(inputmatrix, docmatrix, top_N=10)
    for i,w in enumerate(input_):
        print(w)
        sim_docs = [labels[j] for j in docindices[i]]
        #print("most similar docs: ", ", ".join(sim_docs))
        print("most similar docs: ", sim_docs)
        sim_vals = docsim[i]
        print(sim_vals)
        print()
    
    # closest terms -> the input word which has the largest similarity value
    termsim, termindices = list_utils.matrix_similarity(inputmatrix, termmatrix, top_N=10)
    allterms = tfidf_vectorizer.get_feature_names()
    print(len(allterms))
    #open("/home/dicle/Documents/experiments/thy_topics/output/all_terms.txt", "w").write("\n".join(allterms))
    
    for i,w in enumerate(input_):
        print(w)
        sim_terms = [allterms[j] for j in termindices[i]]
        print("most similar terms: ", ", ".join(sim_terms))
        sim_vals = termsim[i]
        print(sim_vals)
        print(sum(sim_vals))

    # the heaviest term
    similarity_threshold = 0.0  # @TODO should be inferred from the data_matrix
    
    total_termsim_per_instance = np.sum(termsim, axis=1)
    max_sim = total_termsim_per_instance.max()
    max_index = total_termsim_per_instance.argmax()
    #print("max -> ", input_[max_index], " : ",max_sim)
    
    if max_sim <= similarity_threshold:
        highlight_word = ""
        return highlight_word
    
    highlight_word = input_[max_index]
    return highlight_word    



def test():
    
    
    # lsi
    # kmeans
    
    folder = "/home/dicle/Documents/experiments/thy_topics/data"
    fname = "milesNsmiles.txt"
    p = os.path.join(folder, fname)
    t = open(p).read()
    #t = "*** abc *** asd ***"
    #x = re.findall("\*{3}(.*?)\*{3}", t, re.DOTALL)  #, re.MULTILINE)
    #print(len(x))
    
    
    #t = "$$$\n abc\n *** $$$ asd ***"
    items = re.findall("\${3}\n*(.*?)\n*\*{3}\n*", t, re.DOTALL)
    print(len(items))
    y = [i.strip()[:2] for i in items]
    print(y)
    
    # questions
    '''
    for i,item in enumerate(items):
        print(i+1, item[:10])
        x = re.findall("\d{1,2}\.(.*?)\?\n", item)[0]
        print(x)
    '''
    questions = [re.findall("\d{1,2}\.(.*?)\?\n", item)[0].strip()+"?" for item in items]
    print(len(questions))
    
    
    answers = [re.findall("\?\n(.*?)$", item, re.DOTALL)[0].strip() for item in items]
    print(len(answers))
    
    pairs = []
    for q,a in zip(questions, answers):
        print("\n", q,"\n--->",a,"\n")
        pairs.append({"question" : q, "answer" : a})
    
    outfolder = "/home/dicle/Documents/experiments/thy_topics/jsons"
    fname = "miles.json"
    with open(os.path.join(outfolder, fname), "w") as f:
        json.dump(pairs, f, ensure_ascii=False)
        


def get_thy_data(folderpath):

    fnames = io_utils.getfilenames_of_dir(folderpath, removeextension=False)
    fnames.sort()
    
    instances = []
    labels = []
    for fname in fnames:

        content = open(os.path.join(folderpath, fname), "r").read().strip()
        instances.append(content)
        labels.append(fname)
    
    return instances, labels

if __name__ == "__main__":
    
    
    #  EĞER MODEL OLUŞTURULACAKSA THY-FAQ METİNLERİ OKUYALIM
    #jsonfolder = "/home/dicle/Documents/experiments/thy_topics/jsons"
    txtfolder = "/home/dicle/Documents/experiments/thy_topics/txts"  
    instances, labels = get_thy_data(txtfolder)
    
    
    
    # INPUT CÜMLE    
    sentence = "Biletimi nasıl iade edebilirim?"
    #sentence = "Bugün hava güzelmiş"
    #topics_lsi(instances, labels, sentence)
    
    n_clusters = 3
    top_N_words = 30
    
    
    # MODELİ TRAIN EDİP SONUÇ ALMAK İÇİN online_kmeans_topics() METODUNU KULLANALIM
    #result = online_kmeans_topics(instances, labels, sentence, n_clusters, top_N_words)
    
    
    # MODELİ YALNIZCA TRAIN EDİP DİSKE YAZMAK İÇİN train_and_dump_kmeans_model() METODUNU ÇAĞIRALIM (yalnız bir kere)
    picklepath = "/home/dicle/Documents/experiments/thy_topics/models/thy_kmeans.b"
    #train_and_dump_kmeans_model(instances, labels, n_clusters, picklepath)
    
    
    # TRAIN EDİLMİŞ HAZIR MODELİ DİSKTEN OKUYARAK SONUÇ ALMAK İÇİN YALNIZCA offline_kmeans_topics() METODUNU KULLANALIM 
    result = offline_kmeans_topics(picklepath, sentence, n_clusters, top_N_words)
    
    # result : {"word" : "", "nearest_docs" : [], "nearest_terms" : []} değerlerini içeren bir dictionary'dir.
    print(result)
    
    
    
    
    
    
    '''
    results = topics_kmeans(instances, labels, sentence)
    print(results)
    '''
    
    
    
    
    
    
    