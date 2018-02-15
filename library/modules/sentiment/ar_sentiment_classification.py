'''
Created on Mar 29, 2017

@author: dicle
'''

import sys


import os,json
from time import time
import random
import pandas as pd

from sklearn.externals import joblib
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


from modules.dataset import corpus_io, io_utils
from modules.misc import list_utils, table_utils
from django.conf import settings

BASE_DIR = os.path.join(settings.BASE_DIR, '../modules/sentiment/')

def _ar_sentiment_features_pipeline2(feature_params_config_dict  #  {feature_params: {lang: .., weights : .., prep : {}, keywords : []}} see EMAIL_CONF for an example.
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
    polarity_lexicon_count = obt.get_lexicon_count_pipeline(tokenizer=prep.identity, lexicontype=lang)
    
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
    
    #print("--", lang)
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

    
    print("0000000000", feature_params_config_dict)

    return features





'''
MASC_all-reviews.csv has 9141 (Positive/Negative) reviews
'''
def get_arabic_training_set(filepath,
                            sep="\t",
                            textcol="Text",
                            catcol="Polarity"
                            ):
    
    df = pd.read_csv(filepath, sep=sep)
    df = df.sample(frac=1).reset_index(drop=True)
    
    instances = df[textcol].tolist()
    labels = df[catcol].tolist()
    
    return instances, labels


def build_ar_sentiment_analyser(texts, labels,
                                config_dict,
                                picklefolder,
                                modelname):
    
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _ar_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    analyser = clsf.TextClassifier(features_pipeline, classifier)
        
    model = analyser.train(texts, labels)
    
    modelfolder = clsf.dump_classification_system(model, analyser, picklefolder, modelname)
    
    '''
    test_instances = ["أهنئ الدكتور أحمد جمال الدين، القيادي بحزب مصر، بمناسبة صدور أولى روايته", 
                      "عودة جماعة الإخوان إلى الحياة السياسية بنفس وضعها السابق مستحيلة والطرمخة على جرائم الماضي لن تجعلنا نتقدم شبرا",
                      "دضياء رشوان: أن الدكتور عبد المنعم أبو الفتوح الأكثر اعتدالاً سياسيًا وديننًا، كما أنه اجتهد فى مقاصد الشريعة",
                      "إلى زملائي المحامين الراغبين في الانضمام لمبادرة(سامعين صوتك)التى تهدف لدمج متحدي الاعاقه في المجتمع يسعدنا اشتراككم",
                      "5 هاتلي اخوان أي حاجة مش تنوين ومش ضمير اخوان وبعدها كلمة مش هتلاقي غير لوط وشياطين!"
                     ]
    test_labels = ["Positive",
                  "Negative",
                  "Positive",
                  "Positive",
                  "Negative"
                  ]
    ypred = analyser.predict(model, test_instances)
    print("Actual\tPredicted")
    for actual,predicted in zip(test_labels, ypred):
        print(actual,"\t",predicted)
    '''

    return model, modelfolder




def run_model(modelfolder,
              test_instances, test_labels=None):

    model, analyser = clsf.load_classification_system(modelfolder)
    
    ypred = analyser.predict(model, test_instances)
    ytrue = test_labels
    if not ytrue:
        ytrue = [None]*len(ypred)
         
        
    print("Actual\tPredicted")
    for actual,predicted in zip(test_labels, ypred):
        print(actual,"\t",predicted)
    
    
    return ypred


def sentimentResultAr(paramtext):
    
    modelfolder = BASE_DIR + "datamodel/ar/ar_best_sentiment_analyser2"
    model, analyser = clsf.load_classification_system(modelfolder)    
    
    test_instances = [paramtext]
    
    ypred,prediction_map = analyser.predict(model, test_instances)
    
    print(ypred)
    print(prediction_map)
    #return -1,1 process
    json_data=json.loads(prediction_map)
    prob_value=json_data["results"][0]["probability"]
    prob_other=(100 - prob_value)
  
    if prob_other > prob_value:
        prob_value=prob_other
    
    prob_result=0.5
            
    if ypred[0]=="Positive":
        prob_result = {"polarity": (prob_value / 100)}
        #prob_result=["polarity\":\""+  str(prob_value/100)]
    elif ypred[0]=="Negative":
        prob_result = {"polarity": (-prob_value / 100)}
        #prob_result=["polarity\":\""+  str(-prob_value/100)]
    #finish process
    
    
    return prob_result

def trainAr():
    
    ### TRAINING DATA (texts, labels) ILE MODELI TRAIN EDIYORUZ, MODELI DISKE KAYDEDIYORUZ ###########
    
    arabic_polar_data_file = BASE_DIR + "datacvs/ar/MASC_all-SENTIMENT_reviews.csv"
    texts, labels = get_arabic_training_set(arabic_polar_data_file)
    #N = 1000
    #texts = texts[:N]
    #labels = labels[:N]
    
    features_config_dict = conf.ar_sentiment_params
    picklefolder =  BASE_DIR + "datamodel/ar/"
    modelname = "ar_best_sentiment_analyser2"
    model, model_folder = build_ar_sentiment_analyser(texts, labels, features_config_dict, picklefolder, modelname)
    
    ##########  TRAIN EDILMIS MODELI DISKTEN OKUYORUZ, YENI VERI UZERINDE CALISTIRIYORUZ ############
    
    modelfolder = BASE_DIR + "datamodel/ar/ar_best_sentiment_analyser2"

    test_instances = ["أهنئ الدكتور أحمد جمال الدين، القيادي بحزب مصر، بمناسبة صدور أولى روايته", 
                      "عودة جماعة الإخوان إلى الحياة السياسية بنفس وضعها السابق مستحيلة والطرمخة على جرائم الماضي لن تجعلنا نتقدم شبرا",
                      "دضياء رشوان: أن الدكتور عبد المنعم أبو الفتوح الأكثر اعتدالاً سياسيًا وديننًا، كما أنه اجتهد فى مقاصد الشريعة",
                      "إلى زملائي المحامين الراغبين في الانضمام لمبادرة(سامعين صوتك)التى تهدف لدمج متحدي الاعاقه في المجتمع يسعدنا اشتراككم",
                      "5 هاتلي اخوان أي حاجة مش تنوين ومش ضمير اخوان وبعدها كلمة مش هتلاقي غير لوط وشياطين!"
                     ]
    test_labels = None
    
    predicted_labels, prediction_map = clsf.run_saved_model(modelfolder, test_instances, test_labels)
    
    return predicted_labels
    

if __name__ == '__main__':
    
    pred=sentimentResultAr("أهنئ الدكتور أحمد جمال الدين، القيادي بحزب مصر، بمناسبة صدور أولى روايته")
    print("result")
    print(pred)
    
    ### TRAINING DATA (texts, labels) ILE MODELI TRAIN EDIYORUZ, MODELI DISKE KAYDEDIYORUZ ###########
    '''
    arabic_polar_data_file = "/code/django_docker/learning/text_categorization/sentiment_analysis/datacvs/ar/MASC_all-SENTIMENT_reviews.csv"
    texts, labels = get_arabic_training_set(arabic_polar_data_file)
    N = 100
    texts = texts[:N]
    labels = labels[:N]
    
    features_config_dict = conf.ar_sentiment_params
    picklefolder = "/code/django_docker/learning/text_categorization/sentiment_analysis/datamodel/ar/"
    modelname = "ar_best_sentiment_analyser2"
    model, model_folder = build_ar_sentiment_analyser(texts, labels, features_config_dict, picklefolder, modelname)
    '''
    ########################################
    
    
    ##########  TRAIN EDILMIS MODELI DISKTEN OKUYORUZ, YENI VERI UZERINDE CALISTIRIYORUZ ############
    '''
    modelfolder = "datamodel/ar/ar_best_sentiment_analyser2"
    test_instances = ["أهنئ الدكتور أحمد جمال الدين، القيادي بحزب مصر، بمناسبة صدور أولى روايته"
                     ]
    '''
    '''
    test_instances = ["أهنئ الدكتور أحمد جمال الدين، القيادي بحزب مصر، بمناسبة صدور أولى روايته", 
                      "عودة جماعة الإخوان إلى الحياة السياسية بنفس وضعها السابق مستحيلة والطرمخة على جرائم الماضي لن تجعلنا نتقدم شبرا",
                      "دضياء رشوان: أن الدكتور عبد المنعم أبو الفتوح الأكثر اعتدالاً سياسيًا وديننًا، كما أنه اجتهد فى مقاصد الشريعة",
                      "إلى زملائي المحامين الراغبين في الانضمام لمبادرة(سامعين صوتك)التى تهدف لدمج متحدي الاعاقه في المجتمع يسعدنا اشتراككم",
                      "5 هاتلي اخوان أي حاجة مش تنوين ومش ضمير اخوان وبعدها كلمة مش هتلاقي غير لوط وشياطين!"
                     ]
    '''
    '''
    test_labels = ["Positive"
                  ]
    '''
    '''
    test_labels = ["Positive",
                  "Negative",
                  "Positive",
                  "Positive",
                  "Negative"
                  ]
    '''
    #run_model(modelfolder, test_instances, test_labels)
    '''
    predicted_labels, prediction_map = clsf.run_saved_model(modelfolder, test_instances, test_labels)
    '''
    ################################################
    
    
    