'''
Created on May 8, 2017

@author: dicle
'''

import sys
sys.path.append("..")


import sklearn.naive_bayes as nb
import sklearn.linear_model as sklinear


import text_categorization.prototypes.classification.classification_system as clsf_sys
import text_categorization.prototypes.classification.prep_config as prepconfig
import text_categorization.prototypes.tasks.sentiment_analyser as sa_task
from dataset import corpus_io


ar_sent_config_obj = prepconfig.FeatureChoice(lang="ar", weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 1,
                                                                   "char_tfidf" : 1}, 
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=True, deasciify=False, remove_punkt=True, lowercase=False,
                                              wordngramrange=(1,2), charngramrange=(2,2),  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              #classifier=nb.MultinomialNB()
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
                                              )




if __name__ == '__main__':
    
    arabic_polar_data_folder = "/home/dicle/Documents/arabic_nlp/datasets/sentiment/MASC Corpus/MASC Corpus/"
    arabic_polar_data_fname = "MASC_all-SENTIMENT_reviews.csv"
    text_col = "Text"
    cat_col = "Polarity"
    csvsep = "\t"
    shuffle = True
    
    modelrootfolder = "/home/dicle/Documents/experiments/ar_sentiment/models"
    modelname = "ar_sent_masc-all"
    
    test_instances = ["وقال الغانمي لبي بي سي إنه يأمل في هزيمة التنظيم قبل بداية شهر رمضان، والذي يتوقع أن يوافق 26 مايو / \
                        أيار الجاري.",
                      "وأضاف \"أقول إن بقية الموصل ستحرر قبل بداية شهر رمضان المعظم\"."]
    test_labels = None
    
    
    texts, labels = corpus_io.read_dataset_csv(arabic_polar_data_folder, arabic_polar_data_fname, 
                                               text_col, cat_col, csvsep, shuffle)
    
    
    
    ar_sentiment_classifier = clsf_sys.ClassificationSystem(Task=sa_task.SentimentAnalysis,
                                                            task_name="Arabic Sentiment Analysis",
                                                            config_obj=ar_sent_config_obj)
    
    
    
    from text_categorization.systems import Main
    Main.main(ar_sentiment_classifier, 
              texts, labels,
              modelrootfolder, modelname,
              test_instances, test_labels)
    
    
    
    
    