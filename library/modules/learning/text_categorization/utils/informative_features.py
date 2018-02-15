'''
Created on May 15, 2017

@author: dicle
'''



import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.feature_extraction.text as sktext
import sklearn.pipeline as skpipeline



import modules.learning.text_categorization.prototypes.feature_extraction.text_preprocessor as prep
from modules.learning.dataset import corpus_io
import modules.learning.keyword_extraction.detect_topics as topic_detection
import modules.learning.text_categorization.utils.tc_utils as tcutils



# valid only in the binary case
def get_informative_features(instances, labels,
                         lang, N=20,
                         stopword=True, stemming=False,
                         remove_numbers=True, deasciify=False,
                         remove_punkt=True, lowercase=True,
                         wordngramrange=(1,1), nmaxfeature=10000
                         ):
        
    classifier = sklinear.SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)   # object
    #classifier = nb.MultinomialNB()

    
    preprocessor = prep.Preprocessor(lang=lang,
                                     stopword=stopword, more_stopwords=None,
                                     spellcheck=False,
                                     stemming=stemming,
                                     remove_numbers=remove_numbers,
                                     deasciify=deasciify,
                                     remove_punkt=remove_punkt,
                                     lowercase=lowercase
                                    )
        
    tfidfvect = sktext.TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=True,
                                    ngram_range=wordngramrange,
                                    max_features=nmaxfeature)
    

    pipeline_clf = skpipeline.Pipeline([('txtprep', preprocessor),
                                        ('tfidf_vect', tfidfvect),
                                        ('clf', classifier),
                                        ]) 
    
    
    
    '''
    instances_train, instances_test, ytrain, ytest = cv.train_test_split(instances, labels, test_size=0.30, random_state=20)

    pipeline_clf.fit(instances_train, ytrain)
    ypred = pipeline_clf.predict(instances_test)
    
    tc_utils.get_performance(ytest, ypred)
    '''
    pipeline_clf.fit(instances, labels)
    
    topNfeatures =_most_informative_features(tfidfvect, classifier, N)
    print(topNfeatures)
    
    show_most_informative_features(tfidfvect, classifier, n=20)
    '''
    for classlabel in classifier.classes_:
        print(classlabel)
        most_informative_feature_for_class(tfidfvect, classifier, classlabel, n=20)
    '''
    

    print("#########")
    from dataset import corpus_analysis
    corpus_analysis.category_topics(instances, labels, lang)
    
        
        
        
 
'''
finds the first N most informative features (keywords) of the given learning system. 
returns it as [(keyword, weight)] 
'''
def _most_informative_features(vectorizer, classifier, N=20):
    
      
    feature_names = vectorizer.get_feature_names()
            
    pairs = list(zip(feature_names, classifier.coef_[0]))
    pairs.sort(key=lambda x : x[1], reverse=True)
    N_ = min(N, len(pairs))
    topNfeatures = pairs[:N_]
    
    return topNfeatures
    
    
    
    '''
    print("*******************************************************************************")
    try:
        if classlabel in list(classifier.classes_):
            labelid = list(classifier.classes_).index(classlabel)
            feature_names = vectorizer.get_feature_names()
            topn = sorted(zip(classifier.named_steps['clf'].coef_[labelid], feature_names))[-n:]
            keywordList = ""
            for coef, feat in topn:   
                keywordList += ", " + feat+":" +str(round(float(abs(coef))))     
                print(", " + feat.encode('utf-8')+":" +str(round(float(abs(coef)))))    
        
            return keywordList
        else:
            return ""
         
    except IndexError:
        print(IndexError)
    
    
    '''        

def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=10):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print(classlabel, feat, coef)



# from: http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers?noredirect=1&lq=1
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
        




def class_based_informative_features(instances, labels, lang, N=20):
    
    ndim = 1
    # split by labels
    cats = list(set(labels))
    cats_instances = dict.fromkeys(cats, [])
    for t, lbl in zip(instances, labels):
        cats_instances[lbl] = cats_instances[lbl] + [t]
    
    
    cats_keywords = dict.fromkeys(cats, [])
    for cat, txts in cats_instances.items():
        #print(cat," : ", len(txts))

        #detect_topic_svd(txts, labels, "tr", 20)
    
        keywords = topic_detection.detect_topics_nmf(instances=txts, 
                                                     lang=lang, N=N, ndim=ndim,
                                                     stopword=True, stemming=False,
                                                     remove_numbers=True, deasciify=True,
                                                     remove_punkt=True, lowercase=True,
                                                     wordngramrange=(1,1))
    
        cats_keywords[cat] = keywords
    
    # cats_keywords = { cat : { topic_i : [words] }}
    # join all topical words in one list per cat
    cats_keywords2 = dict.fromkeys(cats, [])
    for cat, topical_words in cats_keywords.items():
        keywords = []
        for words in list(topical_words.values()):
            keywords.extend(words)
        cats_keywords2[cat] = keywords
    
    return cats_keywords2

def class_based_informative_features_root(instances, labels, lang, root, N=20):
    ndim = 1
    # split by labels
    cats = list(set(labels))
    cats_instances = dict.fromkeys(cats, [])
    for t, lbl in zip(instances, labels):
        cats_instances[lbl] = cats_instances[lbl] + [t]
    
    
    cats_keywords = dict.fromkeys(cats, [])
    for cat, txts in cats_instances.items():
        #print(cat," : ", len(txts))

        #detect_topic_svd(txts, labels, "tr", 20)
    
        keywords = topic_detection.detect_topics_nmf(instances=txts, 
                                                     lang=lang, N=N, ndim=ndim,
                                                     stopword=True, stemming=False,
                                                     remove_numbers=True, deasciify=True,
                                                     remove_punkt=True, lowercase=True,
                                                     wordngramrange=(1,1))
    
        cats_keywords[cat] = keywords
    
    # cats_keywords = { cat : { topic_i : [words] }}
    # join all topical words in one list per cat
    #cats_keywords2 = dict.fromkeys(cats, [])
    for cat, topical_words in cats_keywords.items():
        keywords = []
        for words in list(topical_words.values()):
            keywords.extend(words)
            #print(words)
        #cats_keywords2[cat] = keywords
        #category ekleme
        
        result=''
        for j in range(0,len(keywords)):
            result=result+keywords[j]+":"+str(N-j)+','
                    
        lsCategory=[x for x in root.categoryList if x.name == str(cat).strip()]
        category=lsCategory[0]
        category.keywords=result        
    return root




if __name__ == '__main__':

    # tweets
    train_data_folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    train_data_fname = "tr_polartweets.csv"
    csvsep="\t"
    text_col="text"
    cat_col="polarity"
    shuffle_dataset = True

 
    N = 20
    
    # READ THE DATASET
    texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
    
    
    '''
    topNfeatures = get_informative_features(texts, labels, "tr", 20)
    print(topNfeatures)
    '''
    
    cats_keywords = class_based_informative_features(texts, labels, "tr", N)
    print(cats_keywords)

    
    
    
     
