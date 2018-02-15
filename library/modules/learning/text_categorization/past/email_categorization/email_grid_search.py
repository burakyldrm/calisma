'''
Created on Jan 24, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import os
import sklearn.naive_bayes as nb
import sklearn.linear_model as sklinear
import sklearn.neighbors as knn

import pandas as pd

from misc import list_utils
import text_categorization.email_categorization.email_classification as emailclf
import text_categorization.email_categorization.EMAIL_CONF as conf
import text_categorization.prototypes.classification as clf

def generate_KMH_grid2():
    '''
     generates all possible (keyword, preprocessing_parameters, classifier) combinations for classifying KMH data
     returns a list of dicts [{weights : {}, feature_params : {prep_params: {}, keywords: []}, classifier=classifier_obj}]
      -- to be later used by a grid search function.
    '''    
    # @TODO keywords, params are hardcoded. pass them in the function.
    
    
    lang = "tr"
    weights = dict(textbased = 1,
                   tokenbased = 1)
    
    weight_p = dict(text_based=[0, 1],
                    token_based=[0,1]
                    )
    
    classifiers = [nb.MultinomialNB(), 
                   sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                   knn.KNeighborsClassifier(n_neighbors=10)]
    
    weight_choices = list_utils.multiplex_dict(weight_p)
    
    
    
    possible_prep_values = dict(
                        stopword_choice=(True, False),
                        more_stopwords_list=(None,),
                        spellcheck_choice=(True, False),
                        stemming_choice=(True, False),
                        number_choice=(True,False),
                        deasc_choice=(True, False),
                        punct_choice=(True, False),
                        case_choice=(True, False),
                        
                        ngramrange = ((1, 2), (1, 1), (2, 2)),   # tuple
                        nmaxfeature = (10000, None),  # int or None  
                        norm=("l2",),
                        use_idf=(True,),        
                    )
    prep_choices = list_utils.multiplex_dict(possible_prep_values)
    
    possible_keywords = ["arıza", "pstn"]
    keyword_combs = list_utils.get_all_combs(possible_keywords)
    
     
    
    # generate all config (feats + classifier) choices as a list of dicts.
    
    
    weights_key = "weights"
    feat_params_key = "feature_params"
    classifier_key = "classifier"
    lang_key = "lang"
    standard_config = { weights_key : None,
                        feat_params_key : None,
                        classifier_key : None,
                        lang_key : None,
                        }
    
    prep_key = "prep_params"
    keyword_key = "keywords"
    standard_feature_params = {prep_key:None,
                               keyword_key:None,
                               }
    
    confs = []   # will store a list of standard_configs
    for weight_choice in weight_choices:
        
        # don't consider the case where text:0, token:1 since the case keywords=[] is already included in keywords_choices          
        if weight_choice == dict(text_based=1, token_based=0):
            conf_choices = {}
            conf_choices[keyword_key] = keyword_combs  # set of all combinations of the given keywords
            conf_choices[prep_key] = [prep_choices[0]]  # any preprocessing parameter choice since it will not be used
            generated_confs = list_utils.multiplex_dict(conf_choices)
            dicts = [{weights_key : weight_choice, feat_params_key : conf} for conf in generated_confs]
            confs.extend(dicts)
        elif weight_choice == dict(text_based=1, token_based=1):
            conf_choices = {}
            conf_choices[keyword_key] = keyword_combs  # set of all combinations of the given keywords
            conf_choices[prep_key] = prep_choices
            generated_confs = list_utils.multiplex_dict(conf_choices)
            dicts = [{weights_key : weight_choice, feat_params_key : conf} for conf in generated_confs]
            confs.extend(dicts)
        # don't consider the case where both weights are 0, which corresponds to baseline and we don't have to measure the performance of its configurations with some other classifiers or parameters..
        

    # add classifier choices
    
    
        
    final_confs = []
    for classifier_choice in classifiers:
        for cdict in confs:
            cdict[classifier_key] = classifier_choice
            cdict[lang_key] = lang
            final_confs.append(cdict)
    
    
    return final_confs




def generate_KMH_grid():
    '''
     generates all possible (keyword, preprocessing_parameters, classifier) combinations for classifying KMH data
     returns a list of dicts [{weights : {}, feature_params : {prep_params: {}, keywords: []}, classifier=classifier_obj}]
      -- to be later used by a grid search function.
    '''    
    # @TODO keywords, params are hardcoded. pass them in the function.
    
    
    lang = "tr"
    weights = dict(textbased = 1,
                   tokenbased = 1)
    
    weight_p = dict(text_based=[0, 1],
                    token_based=[0,1]
                    )
    
    classifiers = [nb.MultinomialNB(), 
                   sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                   knn.KNeighborsClassifier(n_neighbors=10)]
    
    weight_choices = list_utils.multiplex_dict(weight_p)
    
    
    
    possible_prep_values = dict(
                        stopword_choice=(True, False),
                        more_stopwords_list=(None,),
                        spellcheck_choice=(True, False),
                        stemming_choice=(True, False),
                        number_choice=(True,False),
                        deasc_choice=(True, False),
                        punct_choice=(True, False),
                        case_choice=(True, False),
                        
                        ngramrange = ((1, 2), (1, 1), (2, 2)),   # tuple
                        nmaxfeature = (10000, None),  # int or None  
                        norm=("l2",),
                        use_idf=(True,),        
                    )
    prep_choices = list_utils.multiplex_dict(possible_prep_values)
    
    possible_keywords = ["arıza", "pstn"]
    keyword_combs = list_utils.get_all_combs(possible_keywords)
    
    
    '''
    standard_config = { feat_params_key : None,
                    classifier_key : None,            
                    }

    standard_feature_params = {lang_key : None,
                               weights_key : None,
                               prep_key:None,
                               keyword_key:None,
                               }
    '''
    # generate all config (feats + classifier) choices as a list of dicts.
    
    
    confs = []   # will store a list of standard_configs
    for weight_choice in weight_choices:
        
        # don't consider the case where text:0, token:1 since the case keywords=[] is already included in keywords_choices          
        if weight_choice == dict(text_based=1, token_based=0):
            conf_choices = {}
            conf_choices[conf.keyword_key] = keyword_combs  # set of all combinations of the given keywords
            conf_choices[conf.prep_key] = [prep_choices[0]]  # any preprocessing parameter choice since it will not be used
            generated_confs = list_utils.multiplex_dict(conf_choices)
            
            dicts = []
            for cdict in generated_confs:
                cdict[conf.weights_key] = weight_choice
                cdict[conf.lang_key] = lang
                dicts.append(cdict)
            #dicts = [{weights_key : weight_choice, feat_params_key : conf} for conf in generated_confs]
            confs.extend(dicts)
        elif weight_choice == dict(text_based=1, token_based=1):
            conf_choices = {}
            conf_choices[conf.keyword_key] = keyword_combs  # set of all combinations of the given keywords
            conf_choices[conf.prep_key] = prep_choices
            generated_confs = list_utils.multiplex_dict(conf_choices)
            
            dicts = []
            for cdict in generated_confs:
                cdict[conf.weights_key] = weight_choice
                cdict[conf.lang_key] = lang
                dicts.append(cdict)
            
            #dicts = [{weights_key : weight_choice, feat_params_key : conf} for conf in generated_confs]
            confs.extend(dicts)
        # don't consider the case where both weights are 0, which corresponds to baseline and we don't have to measure the performance of its configurations with some other classifiers or parameters..
        

    # add classifier choices
    
    
        
    final_confs = []
    for classifier_choice in classifiers:
        for cdict in confs:
            final_dict = {}
            final_dict[conf.feat_params_key] = cdict.copy()
            final_dict[conf.classifier_key] = classifier_choice
            final_confs.append(final_dict)
    
    
    return final_confs



# confs = [{params..}]
def run_generated_pipelines(confs, instances, labels):


    for config in confs:
        
        feature_params = config[conf.feat_params_key]
        features_pipeline = emailclf._email_features_pipeline2(feature_params)
        
        classifier = config[conf.classifier_key]
        
        email_classifier = clf.TextClassifier(feature_pipelines=features_pipeline,
                                          classifier=classifier)

        acc, fscore, duration = email_classifier.cross_validated_classify(instances, labels)

        # convert feature_params to df, add accuracy, fscore, duration, record-append
        '''
        current_setting
        
        current_setting["duration"] = duration
        current_setting["accuracy"] = acc
        current_setting["fscore"] = fscore
        '''


def see_results(confs, instances, labels):  #, instances, labels):


    result_rows = []
    for config in confs:
            
        # merge nested dicts in one.
        import copy
        config_copy = copy.deepcopy(config)
        #config_copy = config.copy()
        features = config_copy.pop(conf.feat_params_key, None)
        
        prep = features.pop(conf.prep_key, None)
        
        features.update(prep)  # join all the feature keys in one dict.
        
        config_copy.update(features)   # join classifier + feature keys in one dict.
        
        
        feature_params = config[conf.feat_params_key]
        features_pipeline = emailclf._email_features_pipeline2(feature_params)
        
        classifier = config[conf.classifier_key]
        
        email_classifier = clf.TextClassifier(feature_pipelines=features_pipeline,
                                          classifier=classifier)

        acc, fscore, duration = email_classifier.cross_validated_classify(instances, labels)
        
        config_copy["accuracy"] = acc
        config_copy["f1-score"] = fscore
        config_copy["duration"] = duration
        
        result_rows.append(config_copy)
        
        
        
    results = pd.DataFrame(result_rows)  
    
    return results
        

if __name__ == "__main__":
    
    dicts = generate_KMH_grid()
    print(len(dicts))
    for i,j in enumerate(dicts[:5]):
        print(i," -  ",j,"....")
    
    confs = dicts[:7]
    
    
    import text_categorization.email_categorization.email_classification as ec
    from dataset import corpus_io
    instances, labels = ec.get_KMH_data()
    
    N = 100
    instances, labels = corpus_io.select_N_instances(N, instances, labels)
    
    df = see_results(confs, instances, labels)
    folder = "/home/dicle/Documents/data/emailset2/gridsearch2"
    df.to_csv(os.path.join(folder, "results_7-100.csv"), sep="\t", index=False)
    
    
    '''
    print(df1.shape)
    print(df1.columns)
    
    
    df2.to_csv(os.path.join(folder, "df2.csv"), sep="\t", index=False)
    
    
    
    
    
    
    '''