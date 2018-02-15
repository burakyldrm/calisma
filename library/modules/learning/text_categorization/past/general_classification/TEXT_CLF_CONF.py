'''
Created on Mar 20, 2017

@author: dicle
'''

import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb






stopword_key = "stopword_choice"
more_stopwords_key = "more_stopwords_list" 
spellcheck_key = "spellcheck_choice"
stemming_key = "stemming_choice"
remove_numbers_key = "number_choice" 
deasciify_key = "deasc_choice" 
remove_punkt_key = "punct_choice"
lowercase_key = "case_choice"

wordngramrange_key = "wordngramrange"
charngramrange_key = "charngramrange"
nmaxfeature_key = "nmaxfeature"
norm_key = "norm"
use_idf_key = "use_idf"




feat_params_key = "feature_params"
classifier_key = "classifier"
standard_config = { feat_params_key : None,
                    classifier_key : None,
                    }

lang_key = "lang"
weights_key = "weights"
prep_key = "prep_params"
standard_feature_params = {lang_key : None,
                           weights_key : None,
                           prep_key:None,
                           }





################################################


ar_prep_params = {
    stopword_key : True,
    more_stopwords_key : None,
    spellcheck_key : False ,
    stemming_key : False,
    remove_numbers_key : True,
    deasciify_key : False,
    remove_punkt_key : True,
    lowercase_key : False,
    
    wordngramrange_key : (1, 2),
    charngramrange_key : (2, 2),
    nmaxfeature_key : 10000,
    norm_key : "l2",
    use_idf_key : True,
}

ar_clf_params = {
    
    feat_params_key : {
        lang_key : "ar",
        weights_key : {"word_tfidf" : 1,
                       "char_tfidf" : 0,
                       "named_entity_rate" : 0},
        prep_key : ar_prep_params,
        },
    
    #classifier_key : nb.MultinomialNB()
    classifier_key : sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
    }

##################################################
