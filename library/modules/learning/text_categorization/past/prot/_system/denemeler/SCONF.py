'''
Created on Jan 24, 2017

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


tr_prep_params = {
    stopword_key : True,
    more_stopwords_key : None,
    spellcheck_key : False ,
    stemming_key : False,
    remove_numbers_key : False,
    deasciify_key : True,
    remove_punkt_key : True,
    lowercase_key : True,
    
    wordngramrange_key : (1, 2),
    charngramrange_key : (2, 2),
    nmaxfeature_key : 10000,
    norm_key : "l2",
    use_idf_key : True,
}

tr_sentiment_params = {
    
    feat_params_key : {
        lang_key : "tr",
        weights_key : {"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "lexicon_count" : 1,
                           "char_tfidf" : 1},
        prep_key : tr_prep_params,
        },
    
    classifier_key : nb.MultinomialNB()
    
    }



weight_keys = ["word_tfidf",
               "polyglot_value" ,
               "polyglot_count",
               "lexicon_count",
               "char_tfidf"]

#################
class ClsfConf():
    
    lang = ""
    
    weights = dict.fromkeys(weight_keys, int)
    
    classifier = None
    
    prep_params = {
        stopword_key : True,
        more_stopwords_key : None,
        spellcheck_key : False ,
        stemming_key : False,
        remove_numbers_key : False,
        deasciify_key : True,
        remove_punkt_key : True,
        lowercase_key : True,
        
        wordngramrange_key : (1, 2),
        charngramrange_key : (2, 2),
        nmaxfeature_key : 10000,
        norm_key : "l2",
        use_idf_key : True,
    }
    
    
    
###########  ##

##############################################


en_prep_params = {
    stopword_key : True,
    more_stopwords_key : None,
    spellcheck_key : False ,
    stemming_key : True,
    remove_numbers_key : True,
    deasciify_key : False,
    remove_punkt_key : True,
    lowercase_key : True,
    
    wordngramrange_key : (1, 2),
    charngramrange_key : (2, 2),
    nmaxfeature_key : 10000,
    norm_key : "l2",
    use_idf_key : True,
}

en_sentiment_params = {
    
    feat_params_key : {
        lang_key : "en",
        weights_key : {"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "polarity_lexicon_count" : 1,
                           "emoticon_count" : 0,
                           "char_tfidf" : 1,
                           "named_entity_rate" : 0},
        prep_key : en_prep_params,
        },
    
    #classifier_key : nb.MultinomialNB()
    classifier_key : sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
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

ar_sentiment_params = {
    
    feat_params_key : {
        lang_key : "ar",
        weights_key : {"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "polarity_lexicon_count" : 1,
                           "emoticon_count" : 0,
                           "char_tfidf" : 1,
                           "named_entity_rate" : 0},
        prep_key : ar_prep_params,
        },
    
    classifier_key : nb.MultinomialNB()
    #classifier_key : sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42) 
    }






'''
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
'''

