'''
Created on Feb 6, 2017

@author: dicle
'''

import sys
from django_docker.learning.text_categorization import tc_utils2
from django_docker.learning.dataset import io_utils
sys.path.append("..")

import pandas as pd
import os, random

import sklearn.naive_bayes as nb
from sklearn.externals import joblib

from dataset import corpus_io

#import email_classification as emc

import text_categorization.email_categorization.email_classification as emc





# baseline
def train_and_save_spam_detector():

    smsfolder = "/home/dicle/Documents/data/tr_spam/TurkishSMS"
    smsfile = "tr_spam_850sms.csv"   
    sms_instances, sms_labels = corpus_io.read_labelled_texts_csv(path=os.path.join(smsfolder, smsfile), 
                                                          sep="\t", 
                                                          textcol="text", catcol="category")
    
    
    emailfolder = "/home/dicle/Documents/data/tr_spam/TurkishEmail"
    emailfile = "tr_spam_800emails.csv"
    email_instances, email_labels = corpus_io.read_labelled_texts_csv(path=os.path.join(emailfolder, emailfile), 
                                                          sep="\t", 
                                                          textcol="text", catcol="category")
    
    
    # mix the two sets
    pairs = []
    pairs = [(i, l) for i,l in zip(sms_instances, sms_labels)]
    pairs_email = [(i, l) for i,l in zip(email_instances, email_labels)]
    pairs.extend(pairs_email)
    random.shuffle(pairs)
    instances = [i for i,_ in pairs]
    labels = [j for _,j in pairs]
    
    
    #N = 100
    #instances, labels = corpus_io.select_N_instances(N, instances, labels)
    
    lang_key = "lang"
    weights_key = "weights"
    prep_key = "prep_params"
    keyword_key = "keywords"
    feat_params_key = "feature_params"
    classifier_key = "classifier"
    spam_conf = {
                        
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
                        
        classifier_key : nb.MultinomialNB()
    }
    
    spam_detector = emc.get_email_classifier(spam_conf)
    
    #email_classifier = emc.get_KMH_classifier2()
    spam_detector.cross_validated_classify(instances, labels)

    model, _ = spam_detector.train_and_save_model(instances, labels,
                                          picklefolder="/home/dicle/Documents/experiments/tr_spam_detection/models",
                                          modelname="spam_tr_sms-email")

    test_instances = ["Ankara ulus lokasyonu hat başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  hat basvurusu yapmak istiyorum. ",
                      "Ankara ulus lokasyonu pstn başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  pstn hattı basvurusu yapmak istiyorum. ",
                      "İstanbul'daki pstn hattımızda sorun var, iptal etmek istiyoruz.",
                      "İstanbul'daki pstn hattımızda sorun var",
                      "hattımızın hızını düşürür müsünüz",
                      "hattımızın hızını yükseltir misiniz"]
    ypred = spam_detector.predict(model, test_instances)

    print(ypred)



def run_best_model(instances, true_labels=None,
                   modelpath="/home/dicle/Documents/experiments/tr_spam_detection/models/spam_tr_sms-email/model.b"):
    
    
    model = joblib.load(modelpath)
    
    predicted_labels = model.predict(instances)
    
    if true_labels:
        tc_utils2.get_performance(true_labels, predicted_labels, verbose=True)
    
    return predicted_labels
      


if __name__ == '__main__':

    # comment data
    folder = "/home/dicle/Documents/data/fb_comment"
    akbank = "akbanbkfacebookcomment.csv"
    tcell = "turkcellfacebookcomment.csv"
    
    a_comments, _ = corpus_io.read_labelled_texts_csv(os.path.join(folder, akbank), sep="\t", 
                                                      textcol="message", catcol="from.name")
    
    t_comments, _ = corpus_io.read_labelled_texts_csv(os.path.join(folder, tcell), sep="\t", 
                                                      textcol="message", catcol="from.name")
    

    a_labels = run_best_model(a_comments)
    t_labels = run_best_model(t_comments)
    
    print()
    
    #io_utils.todisc_list(os.path.join(folder, "spam-pred_"+akbank), a_labels)
    #io_utils.todisc_list(os.path.join(folder, "spam-pred_"+tcell), t_labels)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    