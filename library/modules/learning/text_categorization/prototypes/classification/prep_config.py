'''
Created on May 4, 2017

@author: dicle
'''

import sys
sys.path.append("..")


'''
tr_sentiment_PREP_params = {
    stopword_key : True,
    more_stopwords_key : None,
    spellcheck_key : False ,
    stemming_key : True,
    remove_numbers_key : True,
    deasciify_key : True,
    remove_punkt_key : True,
    lowercase_key : True,
    
    wordngramrange_key : (1, 2),
    charngramrange_key : (2, 2),
    nmaxfeature_key : None,
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
        prep_key : tr_sentiment_PREP_params,
        },
    
    #classifier_key : nb.MultinomialNB()
    classifier_key : sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    
    }
'''

class PrepChoice():
    
    # default values
    stopword = True
    more_stopwords = None
    spellcheck = False 
    stemming = True
    remove_numbers = True
    deasciify = True
    remove_punkt = True
    lowercase = True
    
    wordngramrange = (1, 2)
    charngramrange = (2, 2)
    nmaxfeature = None
    norm = "l2"
    use_idf = True
        
    
    def __init__(self, 
                 stopword, more_stopwords, spellcheck,
                 stemming,
                 remove_numbers,
                 deasciify,
                 remove_punkt,
                 lowercase,
                 wordngramrange,
                 charngramrange,
                 nmaxfeature,
                 norm,
                 use_idf):
        
        self.stopword = stopword
        self.more_stopwords = more_stopwords
        self.spellcheck = spellcheck 
        self.stemming = stemming
        self.remove_numbers = remove_numbers
        self.deasciify = deasciify
        self.remove_punkt = remove_punkt
        self.lowercase = lowercase

        self.wordngramrange = wordngramrange
        self.charngramrange = charngramrange
        self.nmaxfeature = nmaxfeature
        self.norm = norm
        self.use_idf = use_idf


       
    
class FeatureChoice():
    
    prepchoice = None
    lang = ""
    weights = {"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "lexicon_count" : 1,
                           "char_tfidf" : 1}
    classifier = None
    keywords = None
    
    def __init__(self, lang, weights, stopword, more_stopwords, spellcheck,
                 stemming,
                 remove_numbers,
                 deasciify,
                 remove_punkt,
                 lowercase,
                 wordngramrange,
                 charngramrange,
                 nmaxfeature,
                 norm,
                 use_idf, 
                 classifier=None,
                 keywords=None):
        
        self.lang = lang
        self.weights = {"word_tfidf" : 1,
                           "polyglot_value" : 0,
                           "polyglot_count" : 0,
                           "lexicon_count" : 1,
                           "char_tfidf" : 1}
        
        self.prepchoice = PrepChoice(stopword, more_stopwords, spellcheck,
                                     stemming,
                                     remove_numbers,
                                     deasciify,
                                     remove_punkt,
                                     lowercase,                                 
                                     wordngramrange,
                                     charngramrange,
                                     nmaxfeature,
                                     norm,
                                     use_idf)
        self.classifier = classifier
        
        self.keywords = keywords

if __name__ == '__main__':

    p = PrepChoice(False, True, False,
                   False, True, False,
                   False, True, False,
                   True, False, True, False)

    
    
    print(p.remove_numbers)
    p.c = 5
    print(PrepChoice.__dict__)
    print(PrepChoice.__dict__.keys())
    
    print(p.__dict__)
    
    
    f = FeatureChoice("en", None,
                    False, None, False,
                   False, True, False,
                   False, True, (1,1),
                   (2,2), None, "l2", True)
    print()
    
    print(FeatureChoice.__dict__)
    print(f.__dict__)
    # it is possible to have nested dicts.
    
    
    
    
    
    
    