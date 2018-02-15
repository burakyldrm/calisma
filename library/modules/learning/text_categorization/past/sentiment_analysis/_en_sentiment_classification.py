'''
Created on Feb 28, 2017

@author: dicle
'''


import sys
sys.path.append("..")

import os
from time import time

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline





from text_categorization.prototypes.classification import TextClassifier
import text_categorization.prototypes.text_preprocessor as prep
import text_categorization.prototypes.text_based_transformers as tbt
#import text_categorization.prototypes.token_based_transformers as obt
import text_categorization.prototypes.token_based_multilang as obt

import SENTIMENT_CONF as conf


from dataset import corpus_io, io_utils


def _en_sentiment_features_pipeline2(feature_params_config_dict  #  {feature_params: {lang: .., weights : .., prep : {}, keywords : []}} see EMAIL_CONF for an example.
                                ):
        

    lang = feature_params_config_dict[conf.lang_key]
    feature_weights = feature_params_config_dict[conf.weights_key]
    prep_params = feature_params_config_dict[conf.prep_key]

    #print(feature_weights)
               
        # features found in the processed tokens


    preprocessor = prep.Preprocessor(lang=lang,
                                     stopword=prep_params[conf.stopword_key], more_stopwords=prep_params[conf.more_stopwords_key],
                                     spellcheck=prep_params[conf.spellcheck_key],
                                     stemming=prep_params[conf.stemming_key],
                                     remove_numbers=prep_params[conf.remove_numbers_key],
                                     deasciify=prep_params[conf.deasciify_key],
                                     remove_punkt=prep_params[conf.remove_punkt_key],
                                     lowercase=prep_params[conf.lowercase_key]
                                )
    
    tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                use_idf=prep_params[conf.use_idf_key],
                                ngram_range=prep_params[conf.wordngramrange_key],
                                max_features=prep_params[conf.nmaxfeature_key])

    
    
    #polpipe3 = obt.get_lexicon_count_pipeline(tokenizer=prep.identity)
    emoticon_count = obt.get_EMOT_lexicon_count_pipeline(tokenizer=prep.identity)
    polarity_lexicon_count = obt.get_EN_lexicon_count_pipeline(tokenizer=prep.identity)
    
    token_weights = dict(tfidfvect=feature_weights["word_tfidf"],
                         polarity_lexicon_count=feature_weights["polarity_lexicon_count"],
                         emoticon_count=feature_weights["emoticon_count"])
    token_transformers_dict = dict(tfidfvect=tfidfvect,  # not to lose above integrity if we change variable names
                                   emoticon_count=emoticon_count,
                                   polarity_lexicon_count=polarity_lexicon_count
                                   )
    token_transformers = [(k, v) for k, v in token_transformers_dict.items()]
    
    tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                          # ('nadropper', tbt.DropNATransformer()),                                       
                                          ('union1', skpipeline.FeatureUnion(
                                                transformer_list=token_transformers,
                                                transformer_weights=token_weights                                            
                                        )), ]
                                        )
    
    
    
    
    charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=prep_params[conf.charngramrange_key], lowercase=False)
    
    # polyglot polarity
    polpipe1 = tbt.get_polylglot_polarity_count_pipe(lang)
    polpipe2 = tbt.get_polylglot_polarity_value_pipe(lang)
    
    print("--", lang)
    # stylistic
    
    '''
    # BUG
    named_entity_pipe = tbt.get_named_entity_weight_pipeline(lang)
    
    text_weights = dict(charngramvect=feature_weights["char_tfidf"],   # @TODO hardcoded
                             polpipe1=feature_weights["polyglot_count"],
                             polpipe2=feature_weights["polyglot_value"],
                             named_entity_pipe=feature_weights["named_entity_rate"])
                             
    text_transformers_dict = dict(charngramvect=charngramvect,
                             polpipe1=polpipe1,
                             polpipe2=polpipe2,
                             named_entity_pipe=named_entity_pipe)
    '''
    
    text_weights = dict(charngramvect=feature_weights["char_tfidf"],   # @TODO hardcoded
                             polpipe1=feature_weights["polyglot_count"],
                             polpipe2=feature_weights["polyglot_value"])
                         
                             
    text_transformers_dict = dict(charngramvect=charngramvect,
                             polpipe1=polpipe1,
                             polpipe2=polpipe2)
                            
    
    text_transformers = [(k, v) for k, v in text_transformers_dict.items()]
    '''
    textpipes = [('charngramvect', charngramvect),]
    textpweights = {'charngramvect' : 1.5}
    textpweights = dict(charngramvect = 1 if charngramvect else 0)
    '''
    textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(transformer_list=text_transformers,
                                                                            transformer_weights=text_weights),)])
    
    
    final_transformers_dict = dict(tokenbasedpipe=tokenbasedpipe,
                                   textbasedpipe=textbasedpipe)
    final_transformers = [(k, v) for k, v in final_transformers_dict.items()]
    
    #print(textbasedpipe.named_steps)
    '''        
    #tweights = {k : 1 if v else 0 for k,v in final_transformers.items()}
    check_zero = lambda x : 1 if sum(x) > 0 else 0
    x = list(tokenbasedpipe.get_params(False).values())
    print(len(x), x[0])
    print(x[0][1])   # convert x[0] tuple to dict, then get transformer weights
    print("**")
    print(x,"\n--")
    print(list(textbasedpipe.get_params(False).values()))
    tweights = {k : check_zero(list(k.get_params(False).values())[0][0][1].get_params(False)["transformer_weights"].values())
                      for _, k in final_transformers_dict.items()}
    '''

    features = skpipeline.FeatureUnion(transformer_list=final_transformers,
                                       # transformer_weights=tweights   # weight assignment is not necessary as the number of features is small
                                       )

    return features




def get_en_sentiment_data(folder="/home/dicle/Documents/data/en_sentiment",
                            fname="en_sentiment_2K_movie-reviews.csv",
                            sep=",",
                            text_col="text",
                            cat_col="category"):
    
    path = os.path.join(folder, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(path, sep, text_col, cat_col)
    
    return instances, labels


def run_en_sentiment_analyser2(instances,
                              labels,
                              config_dict=conf.en_sentiment_params,
                              picklefolder="/home/dicle/Documents/karalama",
                              modelname="en_sentiment1002"):
    
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _en_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    analyser = TextClassifier(features_pipeline, classifier)
    
    analyser.cross_validated_classify(instances, labels)
    
    '''
    model, _ = analyser.train_and_save_model(instances, labels, picklefolder, modelname)
    
    test_instances = ["the movie was nice",
                      "it was a very nice movie",
                      "it was a beautiful movie",
                      "i didn't like it..",
                      "i didn't like it at all.."]
    
    ypred = analyser.predict(model, test_instances)
    print(ypred)
    '''


if __name__ == '__main__':
    
    instances_reviews, labels_reviews = get_en_sentiment_data(fname="en_polar_10Kreviews.csv",
                                                              sep="\t")
    instances_tweets, labels_tweets = get_en_sentiment_data(fname="en_sentiment_7K_tweets.csv", 
                                                            sep="\t")
    
    instances = instances_reviews
    labels = labels_reviews
    
    '''
    instances = []
    labels = []
    
    x = []
    x = [(i,j) for i,j in zip(instances_reviews, labels_reviews)]
    for i,j in zip(instances_tweets, labels_tweets):
        x.append((i,j))
    import random
    random.shuffle(x)
    instances = [i for i,_ in x]
    labels = [i for _,i in x]
    '''
    ypred = run_en_sentiment_analyser2(instances, labels, modelname="en_sentiment_reviews_NE")
    print(ypred)
    
    
    
    
    