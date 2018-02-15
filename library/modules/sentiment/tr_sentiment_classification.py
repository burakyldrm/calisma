'''
Created on Apr 3, 2017

@author: dicle
'''

import sys
sys.path.append("..")
import re, pandas,numpy


import os,json


from sklearn.feature_extraction.text import TfidfVectorizer


import sklearn.linear_model as sklinear
import sklearn.naive_bayes as nb
import sklearn.pipeline as skpipeline




import modules.prototypes.classification as clsf
import modules.prototypes.text_preprocessor as prep
import modules.prototypes.text_based_transformers as tbt
#import text_categorization.prototypes.token_based_transformers as obt
import modules.prototypes.token_based_multilang as obt

from modules.sentiment import sentiment_conf as conf
#import sentiment.sentiment_conf as conf
from modules.learning.language_tools.tr_deasciifier import *
from modules.dataset import corpus_io
from django.conf import settings

BASE_DIR = os.path.join(settings.BASE_DIR, '../modules/sentiment/')




def _tr_sentiment_features_pipeline2(feature_params_config_dict  #  {feature_params: {lang: .., weights : .., prep : {}, keywords : []}} see EMAIL_CONF for an example.
                                ):
        

    lang = feature_params_config_dict[conf.lang_key]
    feature_weights = feature_params_config_dict[conf.weights_key]
    prep_params = feature_params_config_dict[conf.prep_key]

               
        # features found in the processed tokens


    preprocessor = prep.Preprocessor(lang=lang,
                                     stopword=prep_params[conf.stopword_key], more_stopwords=prep_params[conf.more_stopwords_key],
                                     spellcheck=prep_params[conf.spellcheck_key],
                                     stemming=prep_params[conf.stemming_key],
                                     remove_numbers=prep_params[conf.remove_numbers_key],
                                     deasciify=prep_params[conf.deasciify_key],
                                     remove_punkt=prep_params[conf.remove_punkt_key],
                                     lowercase=prep_params[conf.lowercase_key]
                                )
    
    tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                use_idf=prep_params[conf.use_idf_key],
                                ngram_range=prep_params[conf.wordngramrange_key],
                                max_features=prep_params[conf.nmaxfeature_key])

    
    polpipe3 = obt.get_lexicon_count_pipeline(tokenizer=prep.identity, lexicontype=lang)
    
    token_weights = dict(tfidfvect=feature_weights["word_tfidf"],
                         polpipe3=feature_weights["lexicon_count"])
    token_transformers_dict = dict(tfidfvect=tfidfvect,  # not to lose above integrity if we change variable names
                                   polpipe3=polpipe3)
    token_transformers = [(k, v) for k, v in token_transformers_dict.items()]
    
    tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                          # ('nadropper', tbt.DropNATransformer()),                                       
                                          ('union1', skpipeline.FeatureUnion(
                                                transformer_list=token_transformers,
                                                transformer_weights=token_weights                                            
                                        )), ]
                                        )
    
    
    
    
    charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=prep_params[conf.charngramrange_key], lowercase=False)
    
    polpipe1 = tbt.get_polylglot_polarity_count_pipe(lang)
    polpipe2 = tbt.get_polylglot_polarity_value_pipe(lang)
    
    
    text_weights = dict(charngramvect=feature_weights["char_tfidf"],   # @TODO hardcoded
                             polpipe1=feature_weights["polyglot_count"],
                             polpipe2=feature_weights["polyglot_value"])
    text_transformers_dict = dict(charngramvect=charngramvect,
                             polpipe1=polpipe1,
                             polpipe2=polpipe2)
    text_transformers = [(k, v) for k, v in text_transformers_dict.items()]
    '''
    textpipes = [('charngramvect', charngramvect),]
    textpweights = {'charngramvect' : 1.5}
    textpweights = dict(charngramvect = 1 if charngramvect else 0)
    '''
    textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(transformer_list=text_transformers,
                                                                            transformer_weights=text_weights),)])
    
    
    final_transformers_dict = dict(tokenbasedpipe=tokenbasedpipe,
                                   textbasedpipe=textbasedpipe)
    final_transformers = [(k, v) for k, v in final_transformers_dict.items()]
    
    
    '''        
    #tweights = {k : 1 if v else 0 for k,v in final_transformers.items()}
    check_zero = lambda x : 1 if sum(x) > 0 else 0
    x = list(tokenbasedpipe.get_params(False).values())
    print(len(x), x[0])
    print(x[0][1])   # convert x[0] tuple to dict, then get transformer weights
    print("**")
    print(x,"\n--")
    print(list(textbasedpipe.get_params(False).values()))
    tweights = {k : check_zero(list(k.get_params(False).values())[0][0][1].get_params(False)["transformer_weights"].values())
                      for _, k in final_transformers_dict.items()}
    '''

    features = skpipeline.FeatureUnion(transformer_list=final_transformers,
                                       # transformer_weights=tweights   # weight assignment is not necessary as the number of features is small
                                       )

    return features
    


def get_tr_sentiment_data(folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint",
                            fname="sentiment_reviews.csv",
                            sep="\t",
                            text_col="text",
                            cat_col="polarity"):
    
    path = os.path.join(folder, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(path, sep, text_col, cat_col)
    
    return instances, labels

def get_data_from_xlsx(fname, folder, X_name, y_name, mapping = False, shuffle = False):
    path = os.path.join(folder, fname)
    veri = pandas.read_excel(path)
    veri = veri.drop_duplicates(['Content Id'])
    if mapping:
        veri[y_name] = veri[y_name].map(mapping)
    if shuffle:
        veri = veri.sample(frac = 1).reset_index(drop = True)
    instances = veri[X_name].values.tolist()
    instances = [str(i) for i in instances]
    labels = veri[y_name].values.tolist()
    instances = [preprocess(i).strip() for i in instances]
    return instances, labels


def get_data_from_csv(X_name, y_name, fname, folder, sep="\t", mapping = False, shuffle = False):
    path = os.path.join(folder, fname)
    veri = pandas.read_csv(path, sep=sep)
    veri = veri.drop_duplicates(['text'])
    if mapping:
        veri[y_name] = veri[y_name].map(mapping)
    if shuffle:
        veri = veri.sample(frac = 1).reset_index(drop = True)
    instances = veri[X_name].values.tolist()
    instances = [str(i) for i in instances]
    labels = veri[y_name].values.tolist()
    instances = [preprocess(i).strip() for i in instances]
    return instances, labels


def preprocess(instances):
    return deasciify_word(tweetIsnumeric(tweetIsalnum(tweetCleanerV2(instances))))


def tweetCleanerV2(data, mapping = False): # import pandas, re
    if not mapping:
        mapping = { r'([a-z])\1+': r'\1',                       # remove repeat letter
                    '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})':'URLs',
                    '(?:\@+[\w_]+[\w\'_\-]*[\w_]+)': ' ',       # Mention ..
                    '(?:\#+[\w_]+[\w\'_\-]*[\w_]+)': ' ',       # Hashtag..
                    '(?:^|\W)rt': ' ',                          # Remove spesific word
                    '(?:^|\W)at': ' ',
                    '(?:^|\s):': ' ',
                    '(?:[\W_]+)':' ',
                    '(\d+[A-Za-z_-]+|[A-Za-z_-]+\d+)': ' '
                    }
    harfler = {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"}
    if isinstance(data, pandas.Series):
        data = data.str.lower()
        data = data.replace(harfler, regex = True)
        for key in mapping:
            data = data.replace(key, mapping[key], regex = True)
        data = data.tolist()
        return data
    elif isinstance(data, str):
        data = data.lower()
        # data = re.sub(harfler, data)
        for key in mapping:
            data = re.sub(key, mapping[key], data)
        return data
    elif isinstance(data, list):
        data = pandas.Series(data)
        data = data.str.lower()
        data = data.replace(harfler, regex = True)
        for key in mapping:
            data = data.replace(key, mapping[key], regex = True)
        data = data.tolist()
        return data
    else:
        pass


def tweetIsalnum(instances):
    liste = []
    if isinstance(instances, list):
        for cumle in instances:
            liste.append(' '.join(kelime for kelime in cumle.split() if kelime.isalnum()))
        return liste
    elif isinstance(instances, str):
        return(' '.join(kelime for kelime in instances.split() if kelime.isalnum()))
    else:
        pass


def tweetIsnumeric(instances):
    liste = []
    if isinstance(instances, list):
        for cumle in instances:
            liste.append(' '.join(kelime for kelime in cumle.split() if not kelime.isnumeric()))
        return liste
    elif isinstance(instances, str):
        return(' '.join(kelime for kelime in instances.split() if not kelime.isnumeric()))
    else:
        pass

def pandasConcat(X_name, y_name, datas):
    veriler = []
    for data in datas:
        # print({X_name:data[0], y_name:data[1]})
        veriler.append(pandas.DataFrame({X_name:data[0], y_name:data[1]}))

    veri = pandas.concat(veriler, ignore_index=True)
    veri[X_name].replace('', numpy.nan, inplace=True)
    veri.dropna(subset=[X_name], inplace=True)
    veri = veri.reset_index()
    instances = veri[X_name].tolist()
    labels = veri[y_name].tolist()
    return instances, labels


def tr_best_setting():
    config_dict = conf.tr_sentiment_params
    feature_params = config_dict[conf.feat_params_key]
    features_pipeline = _tr_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key]
    return features_pipeline, classifier


def tr_sample_setting():
    
    return 

def build_tr_sentiment_analyser(texts, labels,
                                picklefolder, modelname):
     
    
    features_pipeline, classifier = tr_best_setting()
    model, modelfolder = clsf.train_and_save_model(texts, labels, features_pipeline, classifier, picklefolder, modelname)
    
    return model, modelfolder

def cross_validated_sentiment_analyser(texts, labels):
                                
     
    
    features_pipeline, classifier = tr_best_setting()
    accuracy, fscore, duration = clsf.cross_validated_classify(texts, labels, features_pipeline, classifier)
    return accuracy, fscore, duration

def test2():
    config_dict = conf.tr_sentiment_params
    feature_params = config_dict[conf.feat_params_key]
    #features_pipeline = _tr_sentiment_features_pipeline2(feature_params)
    classifier = config_dict[conf.classifier_key] 
    return config_dict       

def sentimentResultTR(paramtext):
    
    #model_folder = "/home/user/git/cognitus-web/modules/sentiment/datamodel/tr_sentiment_test"
    model_folder = BASE_DIR+"datamodel/tr_sentiment_test"
    test_instances = [paramtext]
    test_labels = None
    reordered_test_instances, reordered_test_labels, ypred, prediction_map = predict_tweets_polarity(model_folder, test_instances, test_labels)
    #ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    #print(prediction_map)
    #return -1,1 process
    
    json_data=json.loads(prediction_map)
    prob_value=json_data["results"][0]["probability"]
    prob_other=(100 - prob_value)
  
    if prob_other > prob_value:
        prob_value=prob_other
    
    prob_result = {"polarity": 0.1}
    
    if ypred[0]=="negative" or ypred[0]=="neg":        
        prob_result = {"polarity": -float("{0:.2f}".format((((prob_value/100)*0.7)+0.3)))}
    elif ypred[0]=="positive" or ypred[0]=="pos":
        prob_result = {"polarity": float("{0:.2f}".format((((prob_value/100)*0.7)+0.3)))}
    else :
        if prob_value >50:
            prob_result = {"polarity": float("{0:.2f}".format((((prob_value/100)*0.3))))}
        else:
            prob_result = {"polarity": -float("{0:.2f}".format((((prob_value/100)*0.3))))}
    
    '''       
    if ypred[0]=="positive" or ypred[0]=="pos" or ypred[0]=="neutral":
        prob_result = {"polarity": (prob_value/100)}
    elif ypred[0]=="negative" or ypred[0]=="neg":
        prob_result = {"polarity": (-prob_value/100)}
    '''
    #finish process
    
    return prob_result
 

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

def predict_tweets_polarity(model_folder, test_tweets, test_labels):
    
    SINGLE_WORD_PRED_PROB = 99
        
    short_instances = []
    strue_labels = []
    
    long_instances = []
    ltrue_labels = []
    
    #test_tweets, test_labels = preprocess_tweets(test_tweets, test_labels)
    
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
    sprediction_map = '{\"results\":['
    for stweet in short_instances:
        plabel = obt.single_word_polarity(stweet)
        spredicted_labels.append(plabel)
        sprediction_map +=  "{ \"category\":\""+ plabel+ "\","+ " \"probability\": "+str(SINGLE_WORD_PRED_PROB) + "}"
    sprediction_map += ']}'
        #sprediction_map.append({"\"results\"":[{"\"category\"" : plabel,
        #                        "\"probability\"" : SINGLE_WORD_PRED_PROB}]})
        #print(sprediction_map)
    
    _ltrue_labels = ltrue_labels
    if test_labels is None:
        _ltrue_labels = None
        
    
    statu=0
    #print(len(long_instances), type(long_instances))
   
    lpredicted_labels = []
    lprediction_map = []
    if len(long_instances) > 0:
        statu=1
        lpredicted_labels, lprediction_map = clsf.run_saved_model_sentiment_tr(model_folder, long_instances, _ltrue_labels)
        if len(short_instances)==0:
            return long_instances, ltrue_labels,lpredicted_labels,lprediction_map
    else:
        return short_instances,strue_labels,spredicted_labels,sprediction_map
    #print("L", lpredicted_labels)
    
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
    


def sentimentTrain():
      #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    _folder=BASE_DIR+"datacvs"
    
    mapping = {'ilgisiz'  : 'neutral',
               'Positive' : 'positive',
               'positive' : 'positive',
               'Negative' : 'negative',
               'negative' : 'negative',
               'Neutral'  : 'neutral',
               'neutral'  : 'neutral',
               'Karar Veremedim' : 'neutral'}

    instances1, labels1 = get_data_from_xlsx('disagreed.xlsx', folder=_folder, X_name='Content', y_name='new_cat', mapping=mapping)
    instances2, labels2 = get_data_from_xlsx('sentiment_agreed_skip.xlsx', folder = _folder, X_name='Content', y_name='new_cat', mapping=mapping)
    instances3, labels3 = get_data_from_xlsx('sentiment_agreed_neutral_v2.xlsx', folder = _folder, X_name='Content', y_name='new_cat', mapping=mapping)
    instances4, labels4 = get_data_from_csv('text', 'polarity', 'tr_polartweets.csv', folder=_folder, mapping = mapping)
    #instances5, labels5 = get_data_from_csv('text', 'polarity', 'sentiment_reviews.csv', folder=_folder, mapping = mapping)
    
    '''
    fname="sentiment_reviews.csv"
    sep="\t"
    text_col="text"
    cat_col="polarity"
    
    
    
    #texts, labels = get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    instances5, labels5 = get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    '''
    
    texts, labels = pandasConcat('Content', 'new_cat', [[instances1, labels1],
                                                            [instances2, labels2],
                                                            [instances3, labels3],
                                                            [instances4, labels4]])
    
    #N = 100
    #texts = texts[:N]
    #labels = labels[:N]
    
     ########  TRAIN AND SAVE THE MODEL   #####
    picklefolder = BASE_DIR+"datamodel"
    modelname = "tr_sentiment_test"
    model, modelfolder = build_tr_sentiment_analyser(texts, labels, picklefolder, modelname)
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    model_folder = BASE_DIR+"datamodel/tr_sentiment_test"
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim.."]
    test_labels = None
    ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    
    return ypred


def test():
      #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    folder=BASE_DIR+"datacvs"
    fname="sentiment_reviews.csv"
    sep="\t"
    text_col="text"
    cat_col="polarity"
    
    
    
    texts, labels = get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    #N = 100
    #texts = texts[:N]
    #labels = labels[:N]
    
     ########  TRAIN AND SAVE THE MODEL   #####
    picklefolder = BASE_DIR+"datamodel"
    modelname = "tr_sentiment_test"
    model, modelfolder = build_tr_sentiment_analyser(texts, labels, picklefolder, modelname)
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    model_folder = BASE_DIR+"datamodel/tr_sentiment_test"
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim.."]
    test_labels = None
    ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    
    return ypred

if __name__ == '__main__':
    
    
    
    #######  READ THE BEST PERFORMING TR SENTIMENT DATA #############
    folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint"
    fname="sentiment_reviews.csv"
    sep="\t"
    text_col="text"
    cat_col="polarity"
    
    
    
    texts, labels = get_tr_sentiment_data(folder, fname, sep, text_col, cat_col)
    N = 100
    texts = texts[:N]
    labels = labels[:N]
    
    
    ######  MEASURE THE PERFORMANCE OF THE TR SENTIMENT ANALYSER  ######
    acc, fscore, d = cross_validated_sentiment_analyser(texts, labels)
    
    
    ########  TRAIN AND SAVE THE MODEL   #####
    picklefolder = "/home/dicle/Documents/karalama"
    modelname = "tr_sentiment_test"
    model, modelfolder = build_tr_sentiment_analyser(texts, labels, picklefolder, modelname)
    
    
    #####  READ FROM THE DISC AND TEST THE MODEL   #########
    model_folder = "/home/dicle/Documents/karalama/tr_sentiment_test"
    test_instances = ["film çok hoş",
                      "film güzelmiş.",
                      "film guzelmis.",
                      "hiç beğenmedim..",
                      "hic begenmedim.."]
    test_labels = None
    ypred, prediction_map = clsf.run_saved_model(model_folder, test_instances, test_labels)
    
    
    