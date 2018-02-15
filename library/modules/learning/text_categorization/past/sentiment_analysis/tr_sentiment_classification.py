'''
Created on Apr 3, 2017

@author: dicle
'''

import sys
sys.path.append("..")



import os


from sklearn.feature_extraction.text import TfidfVectorizer


import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline




import text_categorization.prototypes.classification as clsf
import text_categorization.prototypes.text_preprocessor as prep
import text_categorization.prototypes.text_based_transformers as tbt
#import text_categorization.prototypes.token_based_transformers as obt
import text_categorization.prototypes.token_based_multilang as obt

import SENTIMENT_CONF as conf

from dataset import corpus_io






def _tr_sentiment_features_pipeline2(feature_params_config_dict  #  {feature_params: {lang: .., weights : .., prep : {}, keywords : []}} see EMAIL_CONF for an example.
                                ):
        

    lang = feature_params_config_dict[conf.lang_key]
    feature_weights = feature_params_config_dict[conf.weights_key]
    prep_params = feature_params_config_dict[conf.prep_key]

    print(prep_params)
    print(prep_params[conf.stemming_key], prep_params[conf.stopword_key])    
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

    
    polpipe3 = obt.get_lexicon_count_pipeline(tokenizer=prep.identity, lexicontype=lang)
    
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
    
    
    
    
    charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=prep_params[conf.charngramrange_key], lowercase=False)
    
    polpipe1 = tbt.get_polylglot_polarity_count_pipe(lang)
    polpipe2 = tbt.get_polylglot_polarity_value_pipe(lang)
    
    
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
    


def get_tr_sentiment_data(folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint",
                            fname="sentiment_reviews.csv",
                            sep="\t",
                            text_col="text",
                            cat_col="polarity"):
    
    path = os.path.join(folder, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(path, sep, text_col, cat_col)
    
    return instances, labels



def tr_best_setting():
    config_dict = conf.tr_sentiment_params
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _tr_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    return features_pipeline, classifier


def tr_sample_setting():
    
    return 

def build_tr_sentiment_analyser(texts, labels,
                                picklefolder, modelname):
     
    
    features_pipeline, classifier = tr_best_setting()
    model, modelfolder = clsf.train_and_save_model(texts, labels, features_pipeline, classifier, picklefolder, modelname)
    
    return model, modelfolder

def cross_validated_sentiment_analyser(texts, labels):
                                
     
    
    features_pipeline, classifier = tr_best_setting()
    accuracy, fscore, duration = clsf.cross_validated_classify(texts, labels, features_pipeline, classifier)
    return accuracy, fscore, duration
        


if __name__ == '__main__':
    
    _tr_sentiment_features_pipeline2(conf.tr_twitter_sentiment_params[conf.feat_params_key])
    
    '''
    #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    
    # movie + product
    #folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint"
    #fname="sentiment_reviews.csv"
    
    # product
    folder = "/home/dicle/Documents/data/tr_sentiment/Turkish_Products_Sentiment"
    fname = "tr_sentiment_product_reviews.csv"
    sep="\t"
    text_col="text"
    cat_col="polarity"
    
    
    
    texts, labels = get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    
    
    N = 100
    texts = texts[:N]
    labels = labels[:N]
    
    
    ######  MEASURE THE PERFORMANCE OF THE TR SENTIMENT ANALYSER  ######
    acc, fscore, d = cross_validated_sentiment_analyser(texts, labels)
    
    
    ########  TRAIN AND SAVE THE MODEL   #####
    picklefolder = "/home/dicle/Documents/karalama"
    modelname = "tr_sentiment_product_stemmed"
    model, modelfolder = build_tr_sentiment_analyser(texts, labels, picklefolder, modelname)
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    #model_folder = "/home/dicle/Documents/karalama/tr_sentiment_test"
    model_folder = os.path.join(picklefolder, modelname)
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim.."]
    test_labels = None
    ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    
    '''
    