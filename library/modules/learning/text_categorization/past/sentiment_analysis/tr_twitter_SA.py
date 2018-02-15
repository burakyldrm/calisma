'''
Created on Apr 21, 2017

@author: dicle
'''

import os

import pandas as pd
import numpy as np

import tr_sentiment_classification as tr_sent
import text_categorization.prototypes.classification as clsf
import SENTIMENT_CONF as conf
import dataset.twitter.twitter_preprocessing as twitter_prep
from dataset import corpus_io
import text_categorization.prototypes.token_based_multilang as tbtrans
import text_categorization.sentiment_analysis.sentiment_feature_extractors as sf

def get_tr_tweet_sentiment_classifier():
    
    d = conf.tr_twitter_sentiment_params 
    return _get_txt_classifier(d, tr_sent._tr_sentiment_features_pipeline2)


# email_conf is the configuration dict. see EMAIL_CONF
def _get_txt_classifier(clsf_conf, _features_pipeline_builder):   
    
    
    
    feature_params = clsf_conf[conf.feat_params_key]
    features_pipeline = _features_pipeline_builder(feature_params)
    
    classifier = clsf_conf[conf.classifier_key]
    
    classification_system = clsf.TextClassifier(feature_pipelines=features_pipeline,
                                      classifier=classifier)
    
    return classification_system 
    #return classification_system, feature_params, features_pipeline



def get_twitter_data(folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint",
                            fname="sentiment_reviews.csv",
                            sep="\t",
                            text_col="text",
                            cat_col="polarity"):
    _tweets, _labels = tr_sent.get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    return preprocess_tweets(_tweets, _labels)
    '''
    tweets1 = twitter_prep.cleanTweets2(_tweets) # 1- clean twitter symbols
    df = pd.DataFrame(data=np.array([tweets1, labels]).T, columns=["text", "label"])
    df = df.drop_duplicates()   # 2- remove duplicates
    not_empty = lambda x : True if len(x.strip()) > 0 else False
    df = df[df["text"].apply(not_empty)]   # 3- clean empty instances
    
    # todo: replace coooool+ with coool+
    
    tweets = df["text"].tolist()
    labels = df["label"].tolist()
    
    return tweets, labels
    #return df
    '''

def preprocess_tweets(_tweets, _labels):

    tweets1 = twitter_prep.cleanTweets2(_tweets) # 1- clean twitter symbols
    
    labels = _labels
    if labels is None:
        labels = [None]*len(tweets1)
    
    df = pd.DataFrame(data=np.array([tweets1, labels]).T, columns=["text", "label"])
    df = df.drop_duplicates()   # 2- remove duplicates
    not_empty = lambda x : True if len(x.strip()) > 0 else False
    df = df[df["text"].apply(not_empty)]   # 3- clean empty instances
    
    # todo: replace coooool+ with coool+
    
    tweets = df["text"].tolist()
    labels = df["label"].tolist()
    
    if _labels is None:
        labels = _labels
    
    return tweets, labels


   
def main():    
    
    #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    '''
    folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    fname = "tr_polartweets.csv"
    '''
    
    # movie + product
    '''
    folder = "/home/dicle/Documents/data/tr_sentiment/movie_product_joint"
    fname = "tr_sentiment_reviews.csv"
    '''
    
    # movie
    folder = "/home/dicle/Documents/data/tr_sentiment/Turkish_Movie_Sentiment"
    fname = "sentiment_movie_reviews3.csv"
    
    sep = "\t"
    text_col = "text"
    cat_col = "polarity"
    
    picklefolder = "/home/dicle/Documents/experiments/tr_sentiment_detection/tr_tweets"
    modelname = "tr_movie2_svm"
    
    texts, labels = tr_sent.get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    
    #N = 1000
    #texts = texts[:N]
    #labels = labels[:N]
    
    
    # INITIALIZE THE CLASSIFIER
    tr_sent_classifier = get_tr_tweet_sentiment_classifier()
    
    
    ######  MEASURE THE PERFORMANCE OF THE TR SENTIMENT ANALYSER  ######
    #acc, fscore, d = tr_sent_classifier.cross_validated_classify(texts, labels)
    
    
    
    ########  TRAIN AND SAVE THE MODEL   #####
    #model, modelfolder = clsf.train_and_save_model2(texts, labels, tr_sent_classifier, picklefolder, modelname)
    
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    #model_folder = "/home/dicle/Documents/karalama/tr_sentiment_test"
    model_folder = os.path.join(picklefolder, modelname)
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "filmler güzelmiş.",
                      "film guzelmis.",
                      "film güzelmiş.",
                      "hiç beğenmedim..",
                      "hic begenmedim..",
                      "merhaba",
                      "bugün hava güzel",
                      "bugün hava güzel kutay",
                      "bugün hava güzel duygu",
                      "bugün hava iyi değil ruşen",
                      "günaydın",]
    test_labels = None
    reordered_test_instances, reordered_test_labels, ypred, prediction_map = predict_tweets_polarity(model_folder, test_instances, test_labels)
    
    for i,j in zip(reordered_test_instances, prediction_map):
        print(i, "  ", j)
    
    print(ypred)
    print(prediction_map)
    
    
    tweet_folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    tweet_fname = "tr_polartweets.csv"
    tweets, tlabels = get_twitter_data(tweet_folder, tweet_fname, sep, text_col, cat_col)
    print("predict tweet polarity")
    reordered_tweets, reordered_tlabels, tpred, tmap = predict_tweets_polarity(model_folder, tweets, tlabels)
    
    
    keywords = ["merhaba", "günaydın", "nasılsın", "selamlar", "merhabalar"]
    reordered_keywords, reordered_klabels, kpred, kmap = predict_tweets_polarity(model_folder, keywords, None)
    print(keywords)
    print("kmap", kmap)
    



def predict_tweets_polarity(model_folder, test_tweets, test_labels):
    
    SINGLE_WORD_PRED_PROB = 0.99
        
    short_instances = []
    strue_labels = []
    
    long_instances = []
    ltrue_labels = []
    
    test_tweets, test_labels = preprocess_tweets(test_tweets, test_labels)
    
    for i,tweet in enumerate(test_tweets):
        words = tweet.split()  # our tokenizer?
        # hashtag segmentation??
        if len(words) < 2:
            short_instances.append(tweet)
            if test_labels:
                strue_labels.append(test_labels[i])
        else:
            long_instances.append(tweet)
            if test_labels:
                ltrue_labels.append(test_labels[i])
            
                
    
    spredicted_labels = []
    sprediction_map = []
    for stweet in short_instances:
        plabel = tbtrans.single_word_polarity(stweet)
        spredicted_labels.append(plabel)
        sprediction_map.append({"predicted_label" : plabel,
                                "prediction_probability" : SINGLE_WORD_PRED_PROB})
        #print(sprediction_map)
    
    _ltrue_labels = ltrue_labels
    if test_labels is None:
        _ltrue_labels = None
        
    
   
    lpredicted_labels = []
    lprediction_map = []
    if len(long_instances) > 0:
        lpredicted_labels, lprediction_map = clsf.run_saved_model(model_folder, long_instances, _ltrue_labels)
    
    
    reordered_instances = []
    reordered_true_labels = []     
    reordered_pred_labels = []
    reordered_prediction_map = []
    
    reordered_instances.extend(short_instances)
    reordered_pred_labels.extend(spredicted_labels)
    reordered_true_labels.extend(strue_labels)
    reordered_prediction_map.extend(sprediction_map)
    
    reordered_instances.extend(long_instances)
    reordered_pred_labels.extend(lpredicted_labels)
    reordered_true_labels.extend(ltrue_labels)
    reordered_prediction_map.extend(lprediction_map)
    
    
    return reordered_instances, reordered_true_labels, reordered_pred_labels, reordered_prediction_map
    







def main_tweet_prediction():    
    '''
    folder = "/home/dicle/Documents/data/tr_twitter_raw25Apr/preprocessing/prep_test1"
    df = get_twitter_data(folder=folder, fname="test50.csv", sep="\t", text_col="text", cat_col="label")
    df.to_csv(folder+"/c_test50.csv", sep="\t", index=False)
    '''
    
    
    '''
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "filmler güzelmiş.",
                      "film guzelmis.",
                      "film güzelmiş.",
                      "hiç beğenmedim..",
                      "hic begenmedim..",
                      "merhaba",
                      "merhaba!",
                      "bugün hava güzel",
                      "bugün hava güzel kutay",
                      "bugün hava güzel duygu",
                      "bugün hava iyi değil ruşen",
                      "günaydın",
                      "hello",
                      ":)",
                      ": )",
                      ":))",
                      ":/",
                      "",
                      " "]
    
    
    #newtweets = preprocess_tweets(test_instances, _labels=None)
    #print(newtweets)
    
    
    modelfolder = "/home/dicle/Documents/experiments/tr_sentiment_detection/tr_tweets/tr_tweet1"
    tweets, truelabels, predlabels, predmap = predict_tweets_polarity(modelfolder, test_instances, test_labels=None)
    print(tweets)
    
    for t,pm in zip(tweets, predmap):
        print(t, pm)
        print("*")
     
    '''   

def test_param_pass():

    c, d, u1 = get_tr_tweet_sentiment_classifier()
    print(d)
    u2 = c.feature_pipelines
    import sklearn.pipeline as skpipeline
        
    p1 = skpipeline.Pipeline([("p", u1)])
    p2 = skpipeline.Pipeline([("p", u2)])    
    print(p1.get_params())
    print("##")
    print(p2.get_params())        


if __name__ == "__main__":
    
    
    #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    
    folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    fname = "tr_polartweets.csv"
    
    
    # movie + product
    '''
    folder = "/home/dicle/Documents/data/tr_sentiment/movie_product_joint"
    fname = "tr_sentiment_reviews.csv"
    '''
    
    # movie
    '''
    folder = "/home/dicle/Documents/data/tr_sentiment/Turkish_Movie_Sentiment"
    fname = "sentiment_movie_reviews3.csv"
    '''
    
    sep = "\t"
    text_col = "text"
    cat_col = "polarity"
    
    picklefolder = "/home/dicle/Documents/experiments/tr_sentiment_detection/tr_tweets"
    modelname = "tr_tweet2_svm5"
    
    #texts, labels = tr_sent.get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    texts, labels = get_twitter_data(folder, fname, sep, text_col, cat_col)
    
    '''
    N = 100
    texts = texts[:N]
    labels = labels[:N]
    '''
    
    # INITIALIZE THE CLASSIFIER
    tr_sent_classifier = get_tr_tweet_sentiment_classifier()
    
    
    ######  MEASURE THE PERFORMANCE OF THE TR SENTIMENT ANALYSER  ######
    acc, fscore, d = tr_sent_classifier.cross_validated_classify(texts, labels)
    
    
    
    ########  TRAIN AND SAVE THE MODEL   #####
    model, modelfolder = clsf.train_and_save_model2(texts, labels, tr_sent_classifier, picklefolder, modelname)
    
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    #model_folder = "/home/dicle/Documents/karalama/tr_sentiment_test"
    model_folder = os.path.join(picklefolder, modelname)
    
    print("\nPredict sentence polarity:")
    '''
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "filmler güzelmiş.",
                      "film guzelmis.",
                      "film güzelmiş.",
                      "film güzelmiş..",
                      "film güzelmiş",
                      "hiç beğenmedim..",
                      "hic begenmedim..",
                      "hiç begenmedim hic beğenmedim",
                      "merhaba",
                      "bugün hava güzel",
                      "bugün hava güzel kutay",
                      "bugün hava güzel duygu",
                      "bugün hava iyi değil ruşen",
                      "günaydın",
                      "iyi akşamlar",
                      "iyi geceler"]
    '''
    test_instances = ["hic begenmedim.."]
    test_labels = None
    reordered_test_instances, reordered_test_labels, ypred, prediction_map = predict_tweets_polarity(model_folder, test_instances, test_labels)
    
    for i,j in zip(reordered_test_instances, prediction_map):
        print(i, "  ", j)
    
    
    test_instances = ["hiç beğenmedim.."]
    test_labels = None
    reordered_test_instances, reordered_test_labels, ypred, prediction_map = predict_tweets_polarity(model_folder, test_instances, test_labels)
    
    for i,j in zip(reordered_test_instances, prediction_map):
        print(i, "  ", j)
    
    test_instances = ["hiç beğenmedim.."]
    test_labels = None
    reordered_test_instances, reordered_test_labels, ypred, prediction_map = predict_tweets_polarity(model_folder, test_instances, test_labels)
    
    for i,j in zip(reordered_test_instances, prediction_map):
        print(i, "  ", j)
        
    test_instances = ["hiç beğenmedim"]
    test_labels = None
    reordered_test_instances, reordered_test_labels, ypred, prediction_map = predict_tweets_polarity(model_folder, test_instances, test_labels)
    
    for i,j in zip(reordered_test_instances, prediction_map):
        print(i, "  ", j)
    
    '''
    tweet_folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    tweet_fname = "tr_polartweets.csv"
    tweets, tlabels = get_twitter_data(tweet_folder, tweet_fname, sep, text_col, cat_col)
    print("predict tweet polarity")
    reordered_tweets, reordered_tlabels, tpred, tmap = predict_tweets_polarity(model_folder, tweets, tlabels)
    '''
    
    print("\nPredict keyword polarity")
    keywords = ["merhaba", "günaydın", "nasılsın", "selamlar", "merhabalar",
                "Bu arada farkettim de hepsi Turkcell Kullanıyor :)",
                "İyi gun kotu gun dostum mesaj atan turkcell"]
    reordered_keywords, reordered_klabels, kpred, kmap = predict_tweets_polarity(model_folder, keywords, None)
    for keyword, results in zip(reordered_keywords, kmap):
        print(keyword, "  ", results)
    
    

    
      
    