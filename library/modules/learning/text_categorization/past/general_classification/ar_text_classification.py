'''
Created on Mar 20, 2017

@author: dicle
'''


import sys
sys.path.append("..")



import os
from time import time
import random
import pandas as pd

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

import TEXT_CLF_CONF as conf
from dataset import io_utils


def _ar_txt_clf_features_pipeline2(feature_params_config_dict  #  {feature_params: {lang: .., weights : .., prep : {}, keywords : []}} see EMAIL_CONF for an example.
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

    
    
    
    
    token_weights = dict(tfidfvect=feature_weights["word_tfidf"],
                         )
    token_transformers_dict = dict(tfidfvect=tfidfvect,  # not to lose above integrity if we change variable names
                                  
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
                        )
                         
                             
    text_transformers_dict = dict(charngramvect=charngramvect,
                                 )
                            
    
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

    
    #print("0000000000", feature_params_config_dict)

    return features




def run_ar_text_classifier(instances,
                              labels,
                              config_dict=conf.ar_clf_params):
    
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _ar_txt_clf_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    analyser = TextClassifier(features_pipeline, classifier)
    
    analyser.cross_validated_classify(instances, labels)



def get_news_dataset(folderpath):
    
    dataset = []
    
        
    subfolders = io_utils.getfoldernames_of_dir(folderpath)
    for label in subfolders:
        p1 = os.path.join(folderpath, label)
        
        fnames = io_utils.getfilenames_of_dir(p1, removeextension=False)
        
        for fname in fnames:
            p2 = os.path.join(p1, fname)
            text = open(p2, "r").read()
            text = text.strip()
            
            dataset.append((text, label))

    random.shuffle(dataset)

    
    return dataset





def _classify(instances,
              labels,
              config_dict=conf.ar_clf_params,
              picklefolder="/home/dicle/Documents/karalama",
              modelname="ar_txt_clf"):
    
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _ar_txt_clf_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    analyser = TextClassifier(features_pipeline, classifier)
    
    accuracy, fscore, duration = analyser.cross_validated_classify(instances, labels)
    
    return accuracy, fscore, duration


def classify_news(mainfolder="/home/dicle/Documents/arabic_nlp/datasets/textcat_arabic_corpus/encoding_fixed",
                  outputpath="/home/dicle/Documents/experiments/ar_txt_clf/news_clsf_no-char.csv"):
    
    
    news_sets = io_utils.getfoldernames_of_dir(mainfolder)
    
    results = []   # dataset, accuracy, fscore, duration, classifier, params
    all_datasets = []
    
    for folder in news_sets:
        
        print("Reading dataset: ", folder)
        
        path = os.path.join(mainfolder, folder)
        
        dataset = get_news_dataset(path)
        all_datasets.extend(dataset)
        
        texts = [t for t,_ in dataset]
        labels = [l for _,l in dataset]
        accuracy, fscore, duration = _classify(texts, labels)
        result = {"dataset" : folder,
                   "fscore" : fscore,
                   "accuracy" : accuracy,
                   "duration" : duration}
        results.append(result)
        
        
    random.shuffle(all_datasets)
    all_texts = [t for t,_ in all_datasets]
    all_labels = [l for _,l in all_datasets]
    accuracy, fscore, duration = _classify(all_texts, all_labels)
    results.append({"dataset" : "mix",
                   "fscore" : fscore,
                   "accuracy" : accuracy,
                   "duration" : duration
                    })

    results_df = pd.DataFrame(results)
    if outputpath:
        results_df.to_csv(outputpath, sep="\t", index=False)
    
    print("Done.")
    
    return results_df
        


if __name__ == "__main__":
    
    
    ### arabic sample dataset
    folderpath = "/home/dicle/Documents/arabic_nlp/datasets/textcat_arabic_corpus/encoding_fixed/Khaleej-2004"
    news_data = get_news_dataset(folderpath)
    texts = [text for text,_ in news_data]
    labels = [label for _,label in news_data]
    
    
    ##### run cross-validated text classification with the above sample data
    features_config_dict = conf.ar_clf_params
    picklefolder = "/home/dicle/Documents/experiments/ar_txt_clf/models"
    modelname = "news_clsf1"
    run_ar_text_classifier(texts, labels, features_config_dict)
    
    
    '''
    results = classify_news()
    print("Results:")
    print(results)
    '''
