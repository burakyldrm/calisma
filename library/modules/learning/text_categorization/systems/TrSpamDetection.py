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
import text_categorization.prototypes.tasks.general_text_classifier as gen_txt
from dataset import corpus_io

tr_spam_detect_config_obj = prepconfig.FeatureChoice(lang="tr", weights=dict(text_based=1,
                                                                        token_based=1),
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=False, deasciify=True, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=None,  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                                              )




if __name__ == "__main__":
    
    
    spam_email_folder = "/home/dicle/Documents/data/tr_spam/TurkishEmail"
    spam_email_fname = "tr_spam_800emails.csv"
    text_col = "text"
    cat_col = "category"
    csvsep = "\t"
    shuffle = True
    
    test_instances = ["gnctrkcll'li, Meshur McDonald's kampanyasi, 1 menu alana 1 menu hediye, gelecek Cuma basliyor. Ilk kampanya sifren gnctrkcll'den sana hediye. Sifren:51321",
                      "Tatil icin aklinizda neresi varsa, size ozel cok cazip faiz oranlari ve vade secenekleri ile krediniz VakifBankta. Vakifbank, burasi sizin yeriniz.",
                      "Bu sefer eminim :)",
                      "İstatistik sınavında en son hangi konu çıkıcak biliyormusun?"]
    test_labels = ["spam", "spam", "normal", "normal"]
    
    
    modelrootfolder = "/home/dicle/Documents/experiments/tr_spam_detection/models"
    modelname = "tr_spam_email1"
    
    
    # READ THE DATASET
    texts, labels = corpus_io.read_dataset_csv(spam_email_folder, spam_email_fname, 
                                               text_col, cat_col, csvsep, shuffle)
   
    
    # INITIALIZE THE CLASSIFICATION SYSTEM
    tr_spam_classifier = clsf_sys.ClassificationSystem(Task=gen_txt.GeneralTextClassification,
                                                         task_name="Tr Spam Detection",
                                                         config_obj=tr_spam_detect_config_obj
                                                         )
    
    
    from text_categorization.systems import Main
    Main.main(tr_spam_classifier, 
              texts, labels,
              modelrootfolder, modelname,
              test_instances, test_labels)
    
    
    
    
