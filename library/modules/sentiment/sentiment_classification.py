'''
Created on Jan 18, 2017

@author: dicle
'''

import os
from time import time

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline


from modules.dataset import corpus_io, io_utils

from modules.prototypes.classification import TextClassifier
import modules.prototypes.text_preprocessor as prep
import modules.prototypes.text_based_transformers as tbt
import modules.prototypes.token_based_transformers as obt
from modules.sentiment import sentiment_conf



def _tr_sentiment_features_pipeline(
                        lang="tr",
                        feature_weights={"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "lexicon_count" : 0,
                           "char_tfidf" : 1},
    
                        stopword_choice=True,
                        more_stopwords_list=None,
                        spellcheck_choice=False,
                        stemming_choice=False,
                        number_choice=False,
                        deasc_choice=True,
                        punct_choice=True,
                        case_choice=True,
                        
                        word_ngramrange=(1, 2),  # tuple
                        char_ngramrange=(2, 2),
                        nmaxfeature=10000,  # int or None  
                        norm="l2",
                        use_idf=True):
    
    
                 
    preprocessor = prep.Preprocessor(lang=lang,
                                 stopword=stopword_choice, more_stopwords=more_stopwords_list,
                                 spellcheck=spellcheck_choice,
                                 stemming=stemming_choice,
                                 remove_numbers=number_choice,
                                 deasciify=deasc_choice,
                                 remove_punkt=punct_choice,
                                 lowercase=case_choice
                                )
    tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                use_idf=use_idf, ngram_range=word_ngramrange, max_features=nmaxfeature)
    polpipe3 = obt.get_lexicon_count_pipeline(tokenizer=prep.identity)
    
    token_weights = dict(tfidfvect=feature_weights["word_tfidf"],
                         polpipe3=feature_weights["lexicon_count"])
    token_transformers_dict = dict(tfidfvect=tfidfvect,  # not to lose above integrity if we change variable names
                                   polpipe3=polpipe3)
    token_transformers = [(k, v) for k, v in token_transformers_dict.items()]
    
    tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                          # ('nadropper', tbt.DropNATransformer()),                                       
                                          ('union1', skpipeline.FeatureUnion(
                                                transformer_list=token_transformers,
                                                transformer_weights=token_weights                                            
                                        )), ]
                                        )
    
    
    
    
    charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=char_ngramrange, lowercase=False)
    
    polpipe1 = tbt.get_polylglot_polarity_count_pipe(lang)
    polpipe2 = tbt.get_polylglot_polarity_value_pipe(lang)
    
    
    text_weights = dict(charngramvect=feature_weights["char_tfidf"],
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

    
    '''
    tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                          #('nadropper', tbt.DropNATransformer()),                                       
                                          ('union1', skpipeline.FeatureUnion(
                                                transformer_list=[                                                   
                                                 ('tfidfvect', tfidfvect),
                                                 #('polarity3', polpipe3),
                                        ])),]
                                        )
    
    
    textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion([                                
                                         #('polarity1', polpipe1),
                                         #('polarity2', polpipe2),
                                         ('charngramvect', charngramvect),
                                         ]),)])
    
    features = skpipeline.FeatureUnion(transformer_list=[
                                        ('tokenbasedfeatures', tokenbasedpipe),
                                        ('textbasedfeatures', textbasedpipe),
                                       ])
    '''
    return features
    
 


def get_tr_sentiment_classifier(config_dict):
    
    d = config_dict.copy()
    
    lang = d["lang"]
    params = d["params"]
    pipeline = _tr_sentiment_features_pipeline(lang=lang,
                                               feature_weights=params["feature_weights"],
                                               stopword_choice=params["stopword_choice"],
                                               more_stopwords_list=params["more_stopwords_list"],
                                               spellcheck_choice=params["spellcheck_choice"],
                                               stemming_choice=params["stemming_choice"],
                                               number_choice=params["number_choice"],
                                               deasc_choice=params["deasc_choice"],
                                               punct_choice=params["punct_choice"],
                                               case_choice=params["case_choice"],
                                               word_ngramrange=params["word_ngramrange"],
                                               char_ngramrange=params["char_ngramrange"],
                                               nmaxfeature=params["nmaxfeature"],
                                               norm=params["norm"],
                                               use_idf=params["use_idf"])

    classifier = config_dict["classifier"]


    sentiment_analyser = TextClassifier(pipeline, classifier)
    return sentiment_analyser






def get_tr_sentiment_data(folder="/home/user/git/cognitus-web/modules/sentiment/datacvs",
                            fname="sentiment_reviews.csv",
                            sep="\t",
                            text_col="text",
                            cat_col="polarity"):
    
    path = os.path.join(folder, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(path, sep, text_col, cat_col)
    
    return instances, labels




def run_tr_sentiment_analyser(instances,
                              labels,
                              config_dict,
                              picklefolder="datamodel",
                              modelname="sentiment1002"):
    
    
    
    analyser = get_tr_sentiment_classifier(config_dict)
    
    analyser.cross_validated_classify(instances, labels)
    
    model, _ = analyser.train_and_save_model(instances, labels, picklefolder, modelname)
    
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim.."]
    
    ypred = analyser.predict(model, test_instances)
    print(ypred)
    



def tests():
    
    model_folder = "/home/dicle/Documents/experiments/tr_sentiment_detection/models/tr_sent_16K"
    model_name = "model.b"
    path = os.path.join(model_folder, model_name)
    
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim.."]
    
    model = joblib.load(path)
    
    ypred = model.predict(test_instances)

    print(ypred)


def sentimentResult(paramtext):
    '''
    model_folder = "/home/user/git/cognitus-web/modules/sentiment/datamodel/tr_sent_100"
    model_name = "model.b"
    path = os.path.join(model_folder, model_name)
    
    test_instances = [paramtext]
    
    model = joblib.load(path)
    
    ypred = model.predict(test_instances)
    '''
    texts, labels = get_tr_sentiment_data()
    N = 1000
    texts, labels = corpus_io.select_N_instances(N, texts, labels)
    run_tr_sentiment_analyser(texts, labels, config_dict=sentiment_conf.tr_sentiment_params,
                              picklefolder="/home/user/git/cognitus-web/modules/sentiment/datamodel",
                              modelname="tr_sent_100")
    
    ypred=testrvs()
    
    #str(predict[0])

    #print(ypred)
    
    return ypred

def testrvs():
    model_folder = "/home/user/git/cognitus-web/modules/sentiment/datamodel/tr_sent_100"
    model_name = "model.b"
    path = os.path.join(model_folder, model_name)
    
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim..",
                      "beğenmedim.."]
    
    model = joblib.load(path)
    
    ypred = model.predict(test_instances)

    #str(predict[0])

    #print(ypred)
    
    return ypred

def train():
    texts, labels = get_tr_sentiment_data()
    N = 1000
    texts, labels = corpus_io.select_N_instances(N, texts, labels)
    run_tr_sentiment_analyser(texts, labels, config_dict=SENTIMENT_CONF.tr_sentiment_params,
                              picklefolder="/home/user/git/cognitus-web/modules/sentiment/datamodel",
                              modelname="tr_sent_100")
    
    ypred=testrvs()
    
if __name__ == "__main__":
    
    
    texts, labels = get_tr_sentiment_data()
    N = 1000
    texts, labels = corpus_io.select_N_instances(N, texts, labels)
    run_tr_sentiment_analyser(texts, labels, config_dict=SENTIMENT_CONF.tr_sentiment_params,
                              picklefolder="/home/user/git/cognitus-web/modules/sentiment/datamodel",
                              modelname="tr_sent_100")
    
    
    ypred=testrvs()
    print(ypred)
    
    
