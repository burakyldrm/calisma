'''
Created on Jan 24, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb


# feature extraction parameters
# from a pickle file??? or as a class object??
email_feature_params = dict(
    
    lang="tr",
    weights=dict(text_based=1,
                   token_based=1),
    params=dict(
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
                
                keywords=["arıza", "pstn"],
    
            ),
    classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
    )

###################################

stopword_key = "stopword_choice"
more_stopwords_key = "more_stopwords_list" 
spellcheck_key = "spellcheck_choice"
stemming_key = "stemming_choice"
remove_numbers_key = "number_choice" 
deasciify_key = "deasc_choice" 
remove_punkt_key = "punct_choice"
lowercase_key = "case_choice"

ngramrange_key = "ngramrange"
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
keyword_key = "keywords"
standard_feature_params = {lang_key : None,
                           weights_key : None,
                           prep_key:None,
                           keyword_key:None,
                           }

KMH_param_config = {
                        
    feat_params_key : { 
    
        lang_key : "tr",
    
        weights_key : dict(text_based=1,
                       token_based=1),
    
        prep_key : dict(
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
                  ),
        keyword_key : ["arıza", "pstn"],
    },
                    
    classifier_key : sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
}


comment_param_config = {
                        
    feat_params_key : { 
    
        lang_key : "tr",
    
        weights_key : dict(text_based=0,
                       token_based=1),
    
        prep_key : dict(
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
                  ),
        keyword_key : ["arıza", "pstn"],
    },
                    
    classifier_key : sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
    #classifier_key : nb.MultinomialNB()
}


tr_sentiment_params = dict(
    
    lang="tr",
    params=dict(
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
        use_idf=True,
        
        ),
    
    classifier=nb.MultinomialNB()
    
    )




