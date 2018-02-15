'''
Created on Jan 20, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import os
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import corpus_io, io_utils
import sklearn.cross_validation as cv
import sklearn.linear_model as sklinear
import sklearn.pipeline as skpipeline

from text_categorization.email_categorization import EMAIL_CONF as conf 
#import EMAIL_CONF as conf
#import email_categorization.EMAIL_CONF as conf
#from email_classification import EMAIL_CONF as conf

import text_categorization.prototypes.classification as clsf
from text_categorization.prototypes.classification import TextClassifier
import text_categorization.prototypes.text_preprocessor as prep
import text_categorization.prototypes.text_based_transformers as txbt
import text_categorization.prototypes.token_based_transformers as tobt


#===============================================================================
# 
# 
# def kmh_email_features_pipeline2(lang):
#         
#         
#     
#     stopword_choice=True
#     more_stopwords_list=None
#     spellcheck_choice=False
#     stemming_choice=False
#     number_choice=False
#     deasc_choice=True
#     punct_choice=True
#     case_choice=True
#     
#     ngramrange = (1, 2)   # tuple
#     nmaxfeature = 10000   # int or None  
#     norm="l2"
#     use_idf=True
#     
#     weights = {"word_tfidf" : 1,
#                "keywords" : 1}
#     
#     
#     keywords = [] # ["arıza", "pstn"]
#     
#     
#     # use a list of (pipeline, pipeline_name, weight)
#                
#         # features found in the processed tokens
#     token_features = []
#     token_weights = {}
#     preprocessor = prep.Preprocessor( lang=lang,
#                                  stopword=stopword_choice, more_stopwords=more_stopwords_list, 
#                                  spellcheck=spellcheck_choice,
#                                  stemming=stemming_choice, 
#                                  remove_numbers=number_choice, 
#                                  deasciify=deasc_choice, 
#                                  remove_punkt=punct_choice,
#                                  lowercase=case_choice
#                                 )
#     
#     tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
#                                 use_idf=use_idf, ngram_range=ngramrange, max_features=nmaxfeature)
# 
#     tfidfvect_name = 'word_tfidfvect'
#     token_features.append((tfidfvect_name, tfidfvect))
#     token_weights[tfidfvect_name] = 1
#        
#     
#     
#         # features found in the whole raw text
#     text_features = []
#     text_weights = {}
#     #charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), lowercase=False)
#     # keyword presence features
#     if keywords:
#         for keyword in keywords:
#             keywordpipe = tbt.get_keyword_pipeline(keyword)
#             feature_name = "has_"+keyword
#             text_features.append((feature_name, keywordpipe))
#             text_weights[feature_name] = 1
#             
#     
#     
#     
#     tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
#                                           #('nadropper', tbt.DropNATransformer()),                                       
#                                           ('union1', skpipeline.FeatureUnion(
#                                                 transformer_list=token_features ,
#                                                 transformer_weights=token_weights                                                
#                                                 )),                                        
#                                         ])
#     
#     textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(
#                                             transformer_list=text_features,
#                                             transformer_weights=text_weights
#                                             ),
#                                           )
#                                         ])
#     
#     
#     #######
#     # add the feature pipes to final_features if all the component weights are non-zero.
#     ########
#     check_zero_list = lambda x : 1 if sum(x) > 0 else 0
#     #  l = [0,0,0] => check_zero(l) gives 0 and l=[0,0,1] => check_zero(l) gives 1.
#     final_features_dict = dict(tokenbasedpipe=tokenbasedpipe,
#                                textbasedpipe=textbasedpipe)        
#     final_weights = dict.fromkeys(final_features_dict, 1)
#             
#     tkweights = list(token_weights.values())
#     if(check_zero_list(tkweights) == 0):
#         name = "tokenbasedpipe"
#         final_weights[name] = 0
#         del final_features_dict[name]
#     txweights = list(text_weights.values())
#     if(check_zero_list(txweights) == 0):
#         name = "textbasedpipe"
#         final_weights[name] = 0  
#         del final_features_dict[name]                                
#     final_features = list(final_features_dict.items())    
#     
#     fweights = list(final_weights.values())
#     if(check_zero_list(fweights) == 0):
#         return None
#     
#     '''
#     features = skpipeline.FeatureUnion(transformer_list=[
#                                         ('tokenbasedfeatures', tokenbasedpipe),
#                                         ('textbasedfeatures', textbasedpipe),                                          
#                                        ],
#                                        transformer_weights=final_weights)
#     '''
#     features = skpipeline.FeatureUnion(transformer_list=final_features,
#                                        transformer_weights=final_weights)
#     return features
#===============================================================================
# skeleton for email features pipeline
#===============================================================================
# def _email_features_pipeline(   lang,
#                                 stopword_choice=True,
#                                 more_stopwords_list=None,
#                                 spellcheck_choice=False,
#                                 stemming_choice=False,
#                                 number_choice=False,
#                                 deasc_choice=True,
#                                 punct_choice=True,
#                                 case_choice=True,
#                                 
#                                 ngramrange = (1, 2),   # tuple
#                                 nmaxfeature = 10000,   # int or None  
#                                 norm="l2",
#                                 use_idf=True,
#                                 keywords = [] # ["arıza", "pstn"]
#                                 ):
#         
#         
#     
#     
#     
#     weights = {"word_tfidf" : 1,
#                "keywords" : 1}
#     
#     
#     
#     
#     
#     # use a list of (pipeline, pipeline_name, weight)
#                
#         # features found in the processed tokens
#     token_features = []
#     token_weights = {}
#     preprocessor = prep.Preprocessor( lang=lang,
#                                  stopword=stopword_choice, more_stopwords=more_stopwords_list, 
#                                  spellcheck=spellcheck_choice,
#                                  stemming=stemming_choice, 
#                                  remove_numbers=number_choice, 
#                                  deasciify=deasc_choice, 
#                                  remove_punkt=punct_choice,
#                                  lowercase=case_choice
#                                 )
#     
#     tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
#                                 use_idf=use_idf, ngram_range=ngramrange, max_features=nmaxfeature)
# 
#     tfidfvect_name = 'word_tfidfvect'
#     token_features.append((tfidfvect_name, tfidfvect))
#     token_weights[tfidfvect_name] = 1
#        
#     
#     
#         # features found in the whole raw text
#     text_features = []
#     text_weights = {}
#     #charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), lowercase=False)
#     # keyword presence features
#     if keywords:
#         for keyword in keywords:
#             keywordpipe = txbt.get_keyword_pipeline(keyword)
#             feature_name = "has_"+keyword
#             text_features.append((feature_name, keywordpipe))
#             text_weights[feature_name] = 1
#             
#     
#     
#     
#     tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
#                                           #('nadropper', tbt.DropNATransformer()),                                       
#                                           ('union1', skpipeline.FeatureUnion(
#                                                 transformer_list=token_features ,
#                                                 transformer_weights=token_weights                                                
#                                                 )),                                        
#                                         ])
#     
#     textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(
#                                             transformer_list=text_features,
#                                             transformer_weights=text_weights
#                                             ),
#                                           )
#                                         ])
#     
#     
#     #######
#     # add the feature pipes to final_features if all the component weights are non-zero.
#     ########
#     check_zero_list = lambda x : 1 if sum(x) > 0 else 0
#     #  l = [0,0,0] => check_zero(l) gives 0 and l=[0,0,1] => check_zero(l) gives 1.
#     final_features_dict = dict(tokenbasedpipe=tokenbasedpipe,
#                                textbasedpipe=textbasedpipe)        
#     final_weights = dict.fromkeys(final_features_dict, 1)
#             
#     tkweights = list(token_weights.values())
#     if(check_zero_list(tkweights) == 0):
#         name = "tokenbasedpipe"
#         final_weights[name] = 0
#         del final_features_dict[name]
#     txweights = list(text_weights.values())
#     if(check_zero_list(txweights) == 0):
#         name = "textbasedpipe"
#         final_weights[name] = 0  
#         del final_features_dict[name]                                
#     final_features = list(final_features_dict.items())    
#     
#     fweights = list(final_weights.values())
#     if(check_zero_list(fweights) == 0):
#         return None
#     
#     '''
#     features = skpipeline.FeatureUnion(transformer_list=[
#                                         ('tokenbasedfeatures', tokenbasedpipe),
#                                         ('textbasedfeatures', textbasedpipe),                                          
#                                        ],
#                                        transformer_weights=final_weights)
#     '''
#     features = skpipeline.FeatureUnion(transformer_list=final_features,
#                                        transformer_weights=final_weights)
#     return features
#===============================================================================
def _email_features_pipeline2(feature_params_config_dict  #  {feature_params: {lang: .., weights : .., prep : {}, keywords : []}} see EMAIL_CONF for an example.
                                ):
        


    lang = feature_params_config_dict[conf.lang_key]
    final_weights = feature_params_config_dict[conf.weights_key]
    prep_params = feature_params_config_dict[conf.prep_key]
    keywords = feature_params_config_dict[conf.keyword_key]

               
        # features found in the processed tokens
    token_features = []
    token_weights = {}

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
                                ngram_range=prep_params[conf.ngramrange_key],
                                max_features=prep_params[conf.nmaxfeature_key])

    tfidfvect_name = 'word_tfidfvect'
    token_features.append((tfidfvect_name, tfidfvect))
    token_weights[tfidfvect_name] = 1
       
    
    
        # features found in the whole raw text
    text_features = []
    text_weights = {}
    # charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), lowercase=False)
    # keyword presence features
    if keywords:
        for keyword in keywords:
            keywordpipe = txbt.get_keyword_pipeline(keyword)
            feature_name = "has_" + keyword
            text_features.append((feature_name, keywordpipe))
            text_weights[feature_name] = 1
            
    
    
    
    tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                          # ('nadropper', tbt.DropNATransformer()),                                       
                                          ('union1', skpipeline.FeatureUnion(
                                                transformer_list=token_features ,
                                                transformer_weights=token_weights                                                
                                                )),
                                        ])
    
    textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(
                                            transformer_list=text_features,
                                            transformer_weights=text_weights
                                            ),
                                          )
                                        ])
    
    
    
    
    '''
    features = skpipeline.FeatureUnion(transformer_list=[
                                        ('tokenbasedfeatures', tokenbasedpipe),
                                        ('textbasedfeatures', textbasedpipe),                                          
                                       ],
                                       transformer_weights=final_weights)
    '''
    #######
    # add the feature pipes to final_features if all the component weights are non-zero.
    ########
    check_zero_list = lambda x : 1 if sum(x) > 0 else 0
    #  l = [0,0,0] => check_zero(l) gives 0 and l=[0,0,1] => check_zero(l) gives 1.
    final_features_dict = {}     
            
    tkweights = list(token_weights.values())
    if(check_zero_list(tkweights) != 0):
        final_features_dict["token_based"] = tokenbasedpipe
    else:
        final_weights["token_based"] = 0
      
    txweights = list(text_weights.values())
    if(check_zero_list(txweights) != 0):
        final_features_dict["text_based"] = textbasedpipe
    else:
        final_weights["text_based"] = 0  
                                        
    final_features = list(final_features_dict.items())    
    
    fweights = list(final_weights.values())
    
    #print(final_weights)
    
    if((check_zero_list(fweights) == 0) or (len(final_features) == 0)):
        return None
    
    
    features = skpipeline.FeatureUnion(transformer_list=final_features,
                                       transformer_weights=final_weights)
    return features





def _email_features_pipeline(lang,
                                stopword_choice=True,
                                more_stopwords_list=None,
                                spellcheck_choice=False,
                                stemming_choice=False,
                                number_choice=False,
                                deasc_choice=True,
                                punct_choice=True,
                                case_choice=True,
                                
                                ngramrange=(1, 2),  # tuple
                                nmaxfeature=10000,  # int or None  
                                norm="l2",
                                use_idf=True,
                                keywords=[],  # ["arıza", "pstn"]
                                final_weights=dict(text_based=1, token_based=1)
                                ):
        

    # use a list of (pipeline, pipeline_name, weight)
               
        # features found in the processed tokens
    token_features = []
    token_weights = {}
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
                                use_idf=use_idf, ngram_range=ngramrange, max_features=nmaxfeature)

    tfidfvect_name = 'word_tfidfvect'
    token_features.append((tfidfvect_name, tfidfvect))
    token_weights[tfidfvect_name] = 1
       
    
    
        # features found in the whole raw text
    text_features = []
    text_weights = {}
    # charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), lowercase=False)
    # keyword presence features
    if keywords:
        for keyword in keywords:
            keywordpipe = txbt.get_keyword_pipeline(keyword)
            feature_name = "has_" + keyword
            text_features.append((feature_name, keywordpipe))
            text_weights[feature_name] = 1
            
    
    
    
    tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                          # ('nadropper', tbt.DropNATransformer()),                                       
                                          ('union1', skpipeline.FeatureUnion(
                                                transformer_list=token_features ,
                                                transformer_weights=token_weights                                                
                                                )),
                                        ])
    
    textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(
                                            transformer_list=text_features,
                                            transformer_weights=text_weights
                                            ),
                                          )
                                        ])
    
    
    #######
    # add the feature pipes to final_features if all the component weights are non-zero.
    ########
    check_zero_list = lambda x : 1 if sum(x) > 0 else 0
    #  l = [0,0,0] => check_zero(l) gives 0 and l=[0,0,1] => check_zero(l) gives 1.
    final_features_dict = {}     
            
    tkweights = list(token_weights.values())
    if(check_zero_list(tkweights) != 0):
        final_features_dict["token_based"] = tokenbasedpipe
    else:
        final_weights["token_based"] = 0
      
    txweights = list(text_weights.values())
    if(check_zero_list(txweights) != 0):
        final_features_dict["text_based"] = textbasedpipe
    else:
        final_weights["text_based"] = 0  
                                        
    final_features = list(final_features_dict.items())    
    
    fweights = list(final_weights.values())
    if((check_zero_list(fweights) == 0) or (len(final_features) == 0)):
        return None
    
    '''
    features = skpipeline.FeatureUnion(transformer_list=[
                                        ('tokenbasedfeatures', tokenbasedpipe),
                                        ('textbasedfeatures', textbasedpipe),                                          
                                       ],
                                       transformer_weights=final_weights)
    '''
    features = skpipeline.FeatureUnion(transformer_list=final_features,
                                       transformer_weights=final_weights)
    return features



def features_from_dict_to_pipeline(lang, features_dict):
    '''
    An example features_dict:
    
    {'case_choice': True,
     'deasc_choice': True,
     'keywords': ['arıza', 'pstn'],
     'more_stopwords_list': 'None',
     'ngramrange': (1, 2),
     'nmaxfeature': 10000,
     'norm': 'l2',
     'number_choice': False,
     'punct_choice': True,
     'spellcheck_choice': False,
     'stemming_choice': False,
     'stopword_choice': True,
     'use_idf': True
     }

    the keys should be exactly the same.
    
       
    '''
    
    pipeline = _email_features_pipeline(lang,
                                        stopword_choice=features_dict["stopword_choice"],
                                        more_stopwords_list=features_dict["more_stopwords_list"],
                                        spellcheck_choice=features_dict["spellcheck_choice"],
                                        stemming_choice=features_dict["stemming_choice"],
                                        number_choice=features_dict["number_choice"],
                                        deasc_choice=features_dict["deasc_choice"],
                                        punct_choice=features_dict["punct_choice"],
                                        case_choice=features_dict["case_choice"],
                                        ngramrange=features_dict["ngramrange"],
                                        nmaxfeature=features_dict["nmaxfeature"],
                                        norm=features_dict["norm"],
                                        use_idf=features_dict["use_idf"],
                                        keywords=features_dict["keywords"])
    return pipeline


def _get_KMH_classifier():
    
    d = conf.email_feature_params.copy() 
    lang = d["lang"]
    features_dict = d["params"]  # can be read from json provided the keys are exactly the same with those in the CONF file.
    
    features_pipeline = features_from_dict_to_pipeline(lang, features_dict)
    classifier = d["classifier"]
    
    email_classifier = TextClassifier(feature_pipelines=features_pipeline,
                                      classifier=classifier)
    
    return email_classifier
    

def get_KMH_classifier2():
    
    d = conf.KMH_param_config
    '''
    feature_params = d[conf.feat_params_key]
    features_pipeline = _email_features_pipeline2(feature_params)
    
    classifier = d[conf.classifier_key]
    
    email_classifier = TextClassifier(feature_pipelines=features_pipeline,
                                      classifier=classifier)
    
    return email_classifier 
    '''
    return get_email_classifier(d)


# email_conf is the configuration dict. see EMAIL_CONF
def get_email_classifier(email_conf):   
    
    feature_params = email_conf[conf.feat_params_key]
    features_pipeline = _email_features_pipeline2(feature_params)
    
    classifier = email_conf[conf.classifier_key]
    
    email_classifier = TextClassifier(feature_pipelines=features_pipeline,
                                      classifier=classifier)
    
    return email_classifier 
 
    
def get_KMH_data(folderpath="/home/dicle/Documents/data/emailset2",
                 fname="has_pstn2.csv",
                 sep=";",
                 text_col="MAIL",
                 cat_col="TIP"):
    
    datapath = os.path.join(folderpath, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(datapath, sep, text_col, cat_col)
    return instances, labels


def run_kmh():

    instances, labels = get_KMH_data(fname="kmh_nosignature.csv", text_col="MAIL_NOSIGNATURE")
    
    #N = 100
    #instances, labels = corpus_io.select_N_instances(N, instances, labels)
    
    email_classifier = get_KMH_classifier2()
    email_classifier.cross_validated_classify(instances, labels)

    model, _ = email_classifier.train_and_save_model(instances, labels,
                                          picklefolder="/home/dicle/Documents/karalama",
                                          modelname="no_signature2")

    test_instances = ["Ankara ulus lokasyonu hat başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  hat basvurusu yapmak istiyorum. ",
                      "Ankara ulus lokasyonu pstn başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  pstn hattı basvurusu yapmak istiyorum. ",
                      "İstanbul'daki pstn hattımızda sorun var, iptal etmek istiyoruz.",
                      "İstanbul'daki pstn hattımızda sorun var",
                      "hattımızın hızını düşürür müsünüz",
                      "hattımızın hızını yükseltir misiniz"]
    ypred, pred_map = email_classifier.predict(model, test_instances)

    print(ypred)
    print(pred_map)



def cross_validated_classification(instances, labels):
    email_classifier = get_KMH_classifier2()
    accuracy, fscore, duration = email_classifier.cross_validated_classify(instances, labels) 
    return accuracy, fscore, duration


def build_trained_model(instances, labels, picklefolder, modelname):
    email_classifier = get_KMH_classifier2()
    model, modelfolder = clsf.train_and_save_model2(instances, labels, email_classifier, picklefolder, modelname)
    return model, modelfolder


def run_kmh2():

    instances, labels = get_KMH_data(fname="kmh_nosignature.csv", text_col="MAIL_NOSIGNATURE")
    
    #N = 100
    #instances, labels = corpus_io.select_N_instances(N, instances, labels)
    
    email_classifier = get_KMH_classifier2()
    email_classifier.cross_validated_classify(instances, labels)

    model, _ = email_classifier.train_and_save_model(instances, labels,
                                          picklefolder="/home/dicle/Documents/karalama",
                                          modelname="no_signature2")

    test_instances = ["Ankara ulus lokasyonu hat başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  hat basvurusu yapmak istiyorum. ",
                      "Ankara ulus lokasyonu pstn başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  pstn hattı basvurusu yapmak istiyorum. ",
                      "İstanbul'daki pstn hattımızda sorun var, iptal etmek istiyoruz.",
                      "İstanbul'daki pstn hattımızda sorun var",
                      "hattımızın hızını düşürür müsünüz",
                      "hattımızın hızını yükseltir misiniz"]
    ypred, pred_map = email_classifier.predict(model, test_instances)

    print(ypred)
    print(pred_map)
    
    
    
    
'''

class NamedPipeline():
    
    name = ""
    itself = None
    weight = 1
    def __init__(self, name, object, weight):
        self.name = name
        self.itself =object
        self.weight = weight
        
           

def classify_cv_kmh(folderpath = "/home/dicle/Documents/data/emailset2",
                    fname = "has_pstn2.csv",
                    text_col="MAIL",
                    cat_col="TIP",
                    sep=";",
                    picklefolder="/home/dicle/Documents/data/emailset2/models",
                    modelname="kmh_5K-1"):
    
    lang = "tr"
    datapath = os.path.join(folderpath, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(datapath, sep, text_col, cat_col)
    #N = 100
    #instances, labels = corpus_io.select_N_instances(N, instances, labels)
    email_classifier = EmailClassifier(lang)
    email_classifier.cross_validated_classify(instances, labels, clsf=None, features=email_classifier.kmh_email_features_pipeline())
   

def train_and_save_kmh(folderpath = "/home/dicle/Documents/data/emailset2",
                    fname = "has_pstn2.csv",
                    text_col="MAIL",
                    cat_col="TIP",
                    sep=";",
                    picklefolder="/home/dicle/Documents/data/emailset2/models",
                    modelname="kmh_5K-1"):
    
    
    lang = "tr"
    datapath = os.path.join(folderpath, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(datapath, sep, text_col, cat_col)
    
    email_classifier = EmailClassifier(lang)
    model = email_classifier.train(train_instances=instances, train_labels=labels, 
                           clsf=None, features=email_classifier.kmh_email_features_pipeline())  
    
    recordfolder = io_utils.ensure_dir(os.path.join(picklefolder, modelname))
    recordpath = os.path.join(recordfolder, "model.b")
    email_classifier.save_model(model, recordpath)

    return model, recordpath


def predict_kmh(test_instances, test_labels, modelpath):

    test_instances = ["Ankara ulus lokasyonu hat başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  hat basvurusu yapmak istiyorum. ",
                      "Ankara ulus lokasyonu pstn başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  pstn hattı basvurusu yapmak istiyorum. ",
                      "İstanbul'daki pstn hattımızda sorun var, iptal etmek istiyoruz.",
                      "İstanbul'daki pstn hattımızda sorun var",
                      "hattımızın hızını düşürür müsünüz",
                      "hattımızın hızını yükseltir misiniz"]
    
    model = txtclassifier.read_model(modelpath)
    ypred = txtclassifier.predict(model, test_instances)
    print(ypred)
    return ypred


'''



    


if __name__ == '__main__':
    
    # classify_cv_kmh()
    # train_and_save_kmh()
    # predict_kmh(test_instances=[], test_labels=None, modelpath="/home/dicle/Documents/data/emailset2/models/kmh_5K-1/model.b")
    print()
    
    #run_kmh()
    
    
    
    # READ DATA
    folderpath = "/home/dicle/Documents/data/emailset2"
    fname="kmh_nosignature.csv"
    sep = ";"
    text_col="MAIL_NOSIGNATURE"
    cat_col = "TIP"
    instances, labels = get_KMH_data(folderpath, fname, sep, text_col, cat_col)
    
    '''
    # MEASURE PERFORMANCE VIA CROSS-VALIDATED CLASSIFICATION
    accuracy, fscore, duration = cross_validated_classification(instances, labels)
    print("Cross-validated classification results:")
    print("Accuracy", accuracy, "\nF-score: ", fscore, "\nDuration (sec): ", duration)
    
    '''
    
    '''
    # TRAIN THE MODEL AND SAVE IT ON DISC
    picklefolder = "/home/dicle/Documents/experiments/email_classification2/models"
    modelname = "kmh_nosignature_5K-emails"
    model, modelfolder = build_trained_model(instances, labels, picklefolder, modelname)
    
    '''
    # READ TRAINED MODEL AND TEST IT ON DATA
    modelfolder = "/home/dicle/Documents/experiments/email_classification2/models/kmh_nosignature_5K-emails"
    test_instances = ["Ankara ulus lokasyonu hat başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  hat basvurusu yapmak istiyorum. ",
                      "Ankara ulus lokasyonu pstn başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  pstn hattı basvurusu yapmak istiyorum. ",
                      "İstanbul'daki pstn hattımızda sorun var, iptal etmek istiyoruz.",
                      "İstanbul'daki pstn hattımızda sorun var",
                      "hattımızın hızını düşürür müsünüz",
                      "hattımızın hızını yükseltir misiniz"]
    #test_labels = None
    test_labels = ['ORD_001', 'ORD_001', 'ORD_001', 'ORD_001', 'ORD_001', 'TT_001', 'ORD_001', 'ORD_001']
    ypred, prediction_map = clsf.run_saved_model(modelfolder, test_instances, test_labels)
    print("\nPredicted labels:")
    print(ypred)
    print("\nPrediction probabilities:")
    print(prediction_map)
    
    
    
