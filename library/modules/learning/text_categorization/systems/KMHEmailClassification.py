'''
Created on May 8, 2017

@author: dicle
'''

import sklearn.linear_model as sklinear


import text_categorization.prototypes.classification.classification_system as clsf_sys
import text_categorization.prototypes.classification.prep_config as prepconfig
import text_categorization.prototypes.tasks.email_classifier as email_task
from dataset import corpus_io
from text_categorization.utils import tc_utils



kmh_email_config_obj = prepconfig.FeatureChoice(lang="tr", weights=dict(text_based=1,
                                                                        token_based=1),
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=False, deasciify=True, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=None,  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                                              keywords=["arıza", "pstn"])





def main(clsf_system, 
         modelrootfolder, modelname,
         test_instances, test_labels):
    
 
        
    # 2- APPLY CROSS-VALIDATED CLASSIFICATION and GET PERFORMANCE
    accuracy, fscore, duration = clsf_system.get_cross_validated_clsf_performance(texts, labels, nfolds=3)
    
    
    # 3- TRAIN THE MODEL WITH THE ABOVE PARAMETERS; SAVE IT ON THE FOLDER modelrootfolder/modelname
    model, modelfolder = clsf_system.train_and_save_model(texts, labels, modelrootfolder, modelname)
    '''
    import os
    modelfolder = os.path.join(modelrootfolder, modelname)
    '''
    
    # 4.a- PREDICT ONLINE (the model is in the memory)
    predicted_labels, prediction_map = clsf_system.predict_online(model, test_instances)
    
    # 4.b- PREDICT OFFLINE (the model is loaded from the disc)
    predicted_labels, prediction_map = clsf_system.predict_offline(modelfolder, test_instances)
    
    print(prediction_map)
    
    # 4.c- GET PREDICTION PERFORMANCE IF TRUE LABELS ARE AVAILABLE
    if test_labels:
        test_acc, test_fscore = tc_utils.get_performance(test_labels, predicted_labels, verbose=True)
    
    # -- if paramaters are modified (tr_sent_config_obj), create a new ClassificationSystem() and run necessary methods.
    
    

if __name__ == '__main__':
    
    train_data_folder = "/home/dicle/Documents/data/emailset2"
    train_data_fname = "kmh_nosignature.csv"
    csvsep = ";"
    text_col="MAIL_NOSIGNATURE"
    cat_col = "TIP"
    shuffle_dataset = False
    
    modelrootfolder = "/home/dicle/Documents/experiments/email_classification2/models"
    modelname = "KMH_v2"
    
    test_instances = ["Ankara ulus lokasyonu hat başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  hat basvurusu yapmak istiyorum. ",
                      "Ankara ulus lokasyonu pstn başvurusu yapmak istiyoruz, saygilar .",
                      "Ankara ulus lokasyonu için  pstn hattı basvurusu yapmak istiyorum. ",
                      "İstanbul'daki pstn hattımızda sorun var, iptal etmek istiyoruz.",
                      "İstanbul'daki pstn hattımızda sorun var",
                      "hattımızın hızını düşürür müsünüz",
                      "hattımızın hızını yükseltir misiniz"]
    #test_labels = None
    test_labels = ['ORD_001', 'ORD_001', 'ORD_001', 'ORD_001', 'ORD_001', 'TT_001', 'ORD_001', 'ORD_001']
    
    # INITIALIZE THE SYSTEM
    kmh_email_classifier = clsf_sys.ClassificationSystem(Task=email_task.EmailClassification,
                                                         task_name="KMH Email Classification",
                                                         config_obj=kmh_email_config_obj
                                                         )
    
    # READ THE DATASET
    texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
   
   
    main(clsf_system=kmh_email_classifier, 
         modelrootfolder=modelrootfolder, modelname=modelname, 
         test_instances=test_instances, test_labels=test_labels)
    
    
    
    
    
    