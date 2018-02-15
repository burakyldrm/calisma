'''
Created on Apr 3, 2017

@author: dicle
'''



import sys
sys.path.append("..")

import os,json


from sklearn.feature_extraction.text import TfidfVectorizer


import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline




import modules.prototypes.classification as clsf
import modules.prototypes.text_preprocessor as prep
import modules.prototypes.text_based_transformers as tbt
#import text_categorization.prototypes.token_based_transformers as obt
import modules.prototypes.token_based_multilang as obt

from modules.sentiment import sentiment_conf as conf

from modules.dataset import corpus_io
from django.conf import settings

BASE_DIR = os.path.join(settings.BASE_DIR, '../modules/sentiment/')



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


def en_best_setting():
    config_dict = conf.en_sentiment_params
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _en_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    return features_pipeline, classifier




def build_en_sentiment_analyser(texts, labels,
                                picklefolder, modelname):
     
    
    features_pipeline, classifier = en_best_setting()
    model, modelfolder = clsf.train_and_save_model(texts, labels, features_pipeline, classifier, picklefolder, modelname)
    
    return model, modelfolder

def cross_validated_sentiment_analyser(texts, labels):
                                
     
    
    features_pipeline, classifier = en_best_setting()
    accuracy, fscore, duration = clsf.cross_validated_classify(texts, labels, features_pipeline, classifier)
    return accuracy, fscore, duration
        

def sentimentResultEn(paramtext):
    
    modelfolder = BASE_DIR+"datamodel/en_sentiment_test"
    model, analyser = clsf.load_classification_system(modelfolder)    
    
    test_instances = [paramtext]
    result=[]
    ypred,prediction_map = analyser.predict(model, test_instances)
    print(ypred[0])
    print(prediction_map)
    if ypred[0]=="neg":
        result.append("negative")
    elif ypred[0]=="pos":
        result.append("positive")
    #print(ypred)
    
    
    #return -1,1 process-------------
    json_data=json.loads(prediction_map)
    prob_value=json_data["results"][0]["probability"]
    prob_other=(100 - prob_value)
  
    if prob_other > prob_value:
        prob_value=prob_other
    
    prob_result=0.5
            
    if result[0]=="positive":        
        #prob_result=["polarity\":\""+  str(prob_value/100)]
        prob_result = {"polarity": (prob_value / 100)}
    elif result[0]=="negative":
        #prob_result=["polarity\":\""+  str(-prob_value/100)]
        prob_result = {"polarity": (-prob_value / 100)}
    #finish process ----------------
    
    
    return prob_result

def trainEn():
    #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    
    folder=BASE_DIR+"datacvs"
    fname="en_polar_10Kreviews.csv"
    sep="\t"
    text_col="text"
    cat_col="category"
    instances_reviews, labels_reviews = get_en_sentiment_data(folder, fname, sep, text_col, cat_col)
    
    ########  TRAIN AND SAVE THE MODEL   #####
    picklefolder = BASE_DIR+"datamodel"
    modelname = "en_sentiment_test"
    model, modelfolder = build_en_sentiment_analyser(instances_reviews, labels_reviews, picklefolder, modelname)
    
     #####  READ FROM THE DISC AND TEST THE MODEL   #########
    model_folder = BASE_DIR+"datamodel/en_sentiment_test"
    test_instances = ["the movie was nice",
                      "it was a very nice movie",
                      "it was a beautiful movie",
                      "i didn't like it..",
                      "i didn't like it at all.."]
    test_labels = None
    ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    return ypred

if __name__ == '__main__':
    
    ypred=sentimentResultEn("the movie was nice")
    print(ypred)
    
    #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    '''
    folder="/code/django_docker/learning/text_categorization/sentiment_analysis/datacvs"
    fname="en_polar_10Kreviews.csv"
    sep="\t"
    text_col="text"
    cat_col="category"
    instances_reviews, labels_reviews = get_en_sentiment_data(folder, fname, sep, text_col, cat_col)
 
    N = 100
    instances_reviews = instances_reviews[:N]
    labels_reviews = labels_reviews[:N]
    
    
    ######  MEASURE THE PERFORMANCE OF THE TR SENTIMENT ANALYSER  ######
    acc, fscore, d = cross_validated_sentiment_analyser(instances_reviews, labels_reviews)
    
    
    ########  TRAIN AND SAVE THE MODEL   #####
    picklefolder = "/code/django_docker/learning/text_categorization/sentiment_analysis/datamodel"
    modelname = "en_sentiment_test"
    model, modelfolder = build_en_sentiment_analyser(instances_reviews, labels_reviews, picklefolder, modelname)
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    model_folder = "/code/django_docker/learning/text_categorization/sentiment_analysis/datamodel/en_sentiment_test"
    test_instances = ["the movie was nice",
                      "it was a very nice movie",
                      "it was a beautiful movie",
                      "i didn't like it..",
                      "i didn't like it at all.."]
    test_labels = None
    ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    '''