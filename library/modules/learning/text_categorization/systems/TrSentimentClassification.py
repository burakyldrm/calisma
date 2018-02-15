'''
Created on May 8, 2017

@author: dicle
'''


import sklearn.linear_model as sklinear


import text_categorization.prototypes.classification.classification_system as clsf_sys
import text_categorization.prototypes.classification.prep_config as prepconfig
import text_categorization.prototypes.tasks.sentiment_analyser as sa_task
from dataset import corpus_io
from text_categorization.utils import tc_utils

tr_sent_config_obj = prepconfig.FeatureChoice(lang="tr", weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 1,
                                                                   "char_tfidf" : 1}, 
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=True, deasciify=True, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=(2,2),  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))



if __name__ == '__main__':
    
    '''
    train_data_folder = "/home/dicle/Documents/data/tr_sentiment/Turkish_Products_Sentiment"
    train_data_fname = "tr_sentiment_product_reviews.csv"
    '''
    # tweets
    train_data_folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    train_data_fname = "tr_polartweets.csv"
    csvsep="\t"
    text_col="text"
    cat_col="polarity"
    shuffle_dataset = True

    
    modelrootfolder = "/home/dicle/Documents/experiments/tr_sentiment_detection/models"
    modelname = "tweet1_stemmed_all"
    
    
    # READ THE DATASET
    texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
    
    
    
    # 1- INITIALIZE THE SYSTEM
    tr_sentiment_classifier = clsf_sys.ClassificationSystem(Task=sa_task.SentimentAnalysis, 
                                         task_name="TR Sentiment Analysis", 
                                         config_obj=tr_sent_config_obj)
    
    # 2- APPLY CROSS-VALIDATED CLASSIFICATION and GET PERFORMANCE
    accuracy, fscore, duration = tr_sentiment_classifier.get_cross_validated_clsf_performance(texts, labels, nfolds=3)
    
    
    # 3- TRAIN THE MODEL WITH THE ABOVE PARAMETERS; SAVE IT ON THE FOLDER modelrootfolder/modelname
    model, modelfolder = tr_sentiment_classifier.train_and_save_model(texts, labels, modelrootfolder, modelname)
    '''
    import os
    modelfolder = os.path.join(modelrootfolder, modelname)
    '''
    
    test_instances = ["merhaba", "hava çok güzel"]
    test_labels = None
    # 4.a- PREDICT ONLINE (the model is in the memory)
    predicted_labels, prediction_map = tr_sentiment_classifier.predict_online(model, test_instances)
    
    # 4.b- PREDICT OFFLINE (the model is loaded from the disc)
    predicted_labels, prediction_map = tr_sentiment_classifier.predict_offline(modelfolder, test_instances)
    
    print(prediction_map)
    
    # 4.c- GET PREDICTION PERFORMANCE IF TRUE LABELS ARE AVAILABLE
    if test_labels:
        test_acc, test_fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
    # -- if paramaters are modified (tr_sent_config_obj), create a new ClassificationSystem() and run necessary methods.
    
    
    
    '''
    tr_sentiment_classifier.build_system(Task=sa_task.SentimentAnalysis, 
                                         task_name="TR Sentiment Analysis", 
                                         sent_config_obj=tr_sent_config_obj, 
                                         train_data_folder=train_data_folder, 
                                         train_data_fname=train_data_fname, 
                                         text_col=text_col, 
                                         cat_col=cat_col, csvsep=csvsep, 
                                         shuffle_dataset=shuffle_dataset, 
                                         cross_val_performance=cross_val_performance, 
                                         modelfolder=modelfolder, modelname=modelname,
                                         )
    '''
    
    
    
    