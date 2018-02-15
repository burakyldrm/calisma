'''
Created on May 15, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import sklearn.feature_extraction.text as sktext
import sklearn.pipeline as skpipeline
import sklearn.decomposition as decomposer
import sklearn.preprocessing as skprep


import modules.learning.text_categorization.prototypes.feature_extraction.text_preprocessor as prep
import modules.learning.keyword_extraction.utils_topics as utilstopics



def detect_topic_svd(instances, labels, 
                 lang, N=20,
                 stopword=True, stemming=False,
                 remove_numbers=True, deasciify=False,
                 remove_punkt=True, lowercase=True,
                 wordngramrange=(1,1), nmaxfeature=10000):
    
    ndim = 1
      
    preprocessor = prep.Preprocessor(lang=lang,
                                     stopword=stopword, more_stopwords=None,
                                     spellcheck=False,
                                     stemming=stemming,
                                     remove_numbers=remove_numbers,
                                     deasciify=deasciify,
                                     remove_punkt=remove_punkt,
                                     lowercase=lowercase
                                    )
    
    tfidfvect = sktext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=True,
                                    ngram_range=wordngramrange,
                                    max_features=nmaxfeature)
    
    
    svd_model = decomposer.TruncatedSVD(n_components=ndim,
                         algorithm='randomized',
                         n_iter=10, random_state=42)
    
    

    
    svd_transformer = skpipeline.Pipeline([('txtprep', preprocessor),
                                           ('tfidf_vect', tfidfvect),
                                           #('normalizer', skprep.Normalizer()),
                                           ('scaler', skprep.StandardScaler(with_mean=False)),
                                           ('svd', svd_model)])

    docmatrix = svd_transformer.fit_transform(instances)  
    
    
    termmatrix = svd_model.components_.T
    
    import keyword_extraction.topic_extraction_decompose as topics_decomposed             
    topics_decomposed.print_topic_words(svd_model, tfidfvect, n_top_words=N)




def detect_topics_nmf(instances, 
                 lang, N=20, ndim=1,
                 stopword=True, stemming=False,
                 remove_numbers=True, deasciify=True,
                 remove_punkt=True, lowercase=True,
                 wordngramrange=(1,1)):
    
    
    nmf_model = decomposer.NMF(n_components=ndim, random_state=1, alpha=.1, l1_ratio=.5)
    topical_words = _detect_topics(nmf_model, instances, lang, N, stopword, stemming, remove_numbers, deasciify, remove_punkt, lowercase, wordngramrange)
    return topical_words
    
    

'''
def detect_topic_nmf(instances, labels, 
                 lang, N=20,
                 stopword=True, stemming=True,
                 remove_numbers=True, deasciify=True,
                 remove_punkt=True, lowercase=True,
                 wordngramrange=(1,1)):
    
    ndim = 1
    nmaxfeature=200
    
    preprocessor = prep.Preprocessor(lang=lang,
                                     stopword=stopword, more_stopwords=None,
                                     spellcheck=False,
                                     stemming=stemming,
                                     remove_numbers=remove_numbers,
                                     deasciify=deasciify,
                                     remove_punkt=remove_punkt,
                                     lowercase=lowercase
                                    )
    
    tfidfvect = sktext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=True,
                                    ngram_range=wordngramrange,
                                    max_features=nmaxfeature)
    
    nmf_model = decomposer.NMF(n_components=ndim, random_state=1, alpha=.1, l1_ratio=.5)
    
    nmf_transformer = skpipeline.Pipeline([('txtprep', preprocessor),
                                           ('tfidf_vect', tfidfvect),
                                           #('normalizer', skprep.Normalizer()),
                                           ('scaler', skprep.StandardScaler(with_mean=False)),
                                           ('nmf', nmf_model)])

    nmf_transformer.fit(instances)
    
    
    import keyword_extraction.topic_extraction_decompose as topics_decomposed             
    topics_decomposed.print_topic_words(nmf_model, tfidfvect, n_top_words=N)
'''   
    
    


def _detect_topics(model,
                 instances,  
                 lang, N=20,
                 stopword=True, stemming=True,
                 remove_numbers=True, deasciify=True,
                 remove_punkt=True, lowercase=True,
                 wordngramrange=(1,1)):
    
    ndim = 1
    nmaxfeature=200
    
    preprocessor = prep.Preprocessor(lang=lang,
                                     stopword=stopword, more_stopwords=None,
                                     spellcheck=False,
                                     stemming=stemming,
                                     remove_numbers=remove_numbers,
                                     deasciify=deasciify,
                                     remove_punkt=remove_punkt,
                                     lowercase=lowercase
                                    )
    
    tfidfvect = sktext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=True,
                                    ngram_range=wordngramrange,
                                    max_features=nmaxfeature)
    
    
    topical_transformer = skpipeline.Pipeline([('txtprep', preprocessor),
                                           ('tfidf_vect', tfidfvect),
                                           #('normalizer', skprep.Normalizer()),
                                           ('scaler', skprep.StandardScaler(with_mean=False)),
                                           ('nmf', model)])

    topical_transformer.fit(instances)
    
    return utilstopics.get_topic_words(model, tfidfvect, N)
    
    
    



