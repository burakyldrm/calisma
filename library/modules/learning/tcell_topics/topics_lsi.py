'''
Created on Feb 16, 2017

@author: dicle
'''

import warnings
import os

import numpy as np
import sklearn.feature_extraction.text as txtfeatext
import sklearn.preprocessing as skprep
from sklearn.decomposition import TruncatedSVD
import sklearn.pipeline as skpipeline


from dataset import io_utils
from language_tools import TokenHandler
from misc import list_utils


with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    
    
# database: instances, labels
# topics extracted, then the important words and close documents of the input sentence is detected based on cosine sim..
def detect_topic(instances, labels, sentence, 
                 ndim=5,
                 n_gram_range=(1,1),
                 n_max_features=None):
    
    highlight_word = ""
    
    svd_model = TruncatedSVD(n_components=ndim,
                         algorithm='randomized',
                         n_iter=10, random_state=42)
    
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
                                                   stemming=True, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    tfidf_vectorizer = txtfeatext.TfidfVectorizer(tokenizer=preprocessor, 
                                          ngram_range=n_gram_range,
                                          max_features=n_max_features)
    svd_transformer = skpipeline.Pipeline([('vectorizer', tfidf_vectorizer),
                                           #('normalizer', skprep.Normalizer()),
                                           ('scaler', skprep.StandardScaler(with_mean=False)),
                                           ('svd', svd_model)])

    docmatrix = svd_transformer.fit_transform(instances)
    
    
    

    input_ = preprocessor(sentence)
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
        print("most similar docs: ", ", ".join(sim_docs))
        sim_vals = docsim[i]
        print(sim_vals)
        print()
    
    # closest terms -> the input word which has the largest similarity value
    termsim, termindices = list_utils.matrix_similarity(inputmatrix, termmatrix, top_N=10)
    allterms = tfidf_vectorizer.get_feature_names()
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
    


    


if __name__ == '__main__':
    
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
    
    sentence = "fatura değişikliklerini nasıl öğrenebilirim?"
    sentence = "hava ne güzel"
    #sentence = "merhaba"
    max_term = detect_topic(instances, labels, sentence)
    print("max term ", max_term)
    
    
    
    