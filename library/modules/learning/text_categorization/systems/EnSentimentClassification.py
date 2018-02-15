'''
Created on May 11, 2017

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





en_sent_config_obj = prepconfig.FeatureChoice(lang="en", weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 1,
                                                                   "char_tfidf" : 1}, 
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=True,
                                              remove_numbers=True, deasciify=False, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=(2,2),  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))




if __name__ == "__main__":
    
    en_sentiment_data_folder = "/home/dicle/Documents/data/en_sentiment"
    en_sentiment_data_fname="en_polar_10Kreviews.csv"
    csvsep = "\t"
    text_col = "text"
    cat_col = "category"
    shuffle_dataset = True
    
    test_instances = ["the movie was nice",
                      "it was a very nice movie",
                      "it was a beautiful movie",
                      "i didn't like it..",
                      "i didn't like it at all.."]
    test_labels = None
    
    
    modelrootfolder = "/home/dicle/Documents/experiments/en_sentiment_detection/models"
    modelname = "en_polar10Kreviews"
    
    
    # READ DATA
    texts, labels = corpus_io.read_dataset_csv(en_sentiment_data_folder, en_sentiment_data_fname, 
                                               text_col, cat_col, csvsep, shuffle_dataset)
    
    
    
    # INITIALIZE CLASSIFICATION SYSTEM
    en_sentiment_classifier = clsf_sys.ClassificationSystem(Task=sa_task.SentimentAnalysis,
                                                            task_name="English Sentiment Analysis",
                                                            config_obj=en_sent_config_obj)
    
    
    from text_categorization.systems import Main
    Main.main(en_sentiment_classifier, 
              texts, labels,
              modelrootfolder, modelname,
              test_instances, test_labels)




