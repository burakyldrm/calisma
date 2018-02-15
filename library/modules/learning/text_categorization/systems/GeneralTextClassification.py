'''
Created on May 11, 2017

@author: dicle
'''



import sklearn.linear_model as sklinear
from sklearn.externals import joblib

import modules.learning.text_categorization.prototypes.classification.classification_system as clsf_sys
import modules.learning.text_categorization.prototypes.classification.prep_config as prepconfig
import modules.learning.text_categorization.prototypes.tasks.general_text_classifier as gen_txt
import modules.learning.text_categorization.utils.informative_features as infeatures
from modules.learning.dataset import corpus_io
from modules.learning.text_categorization.utils import tc_utils
from modules.learning.text_categorization.systems import Main
import os,sys,traceback,uuid,shutil
from django.core.files.storage import default_storage
from django.conf import settings
from user.models import Module
from user.models import ModuleData
from django.core.files import File


BASE_DIR = os.path.join(settings.BASE_DIR, '../trained_data/')
'''
gen_txt_clsf_config_obj = prepconfig.FeatureChoice(lang="tr", weights=dict(text_based=1,
                                                                        token_based=1),
                                              stopword=True, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=False,
                                              remove_numbers=False, deasciify=True, remove_punkt=True, lowercase=True,
                                              wordngramrange=(1,2), charngramrange=None,  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                                              )

'''


class Category:
    predict = ''
    results = ''
    
def roundFloat(val):
    return int(round(val*100))  


def removeFile(fileName):
    try:
        os.remove(fileName)
    except OSError:
        pass

def removeDirectory(path):
    try:
        shutil.rmtree(path)
    except OSError as e:        
        print("klasor silme hatası :"+ e.strerror)
        print(e.errno)
        pass


def classify_execute(module_id,  content, categoryMap):
        
    
    moduleLoad=Module.objects.filter(id=module_id)
    modelfolder=""
    if moduleLoad[0].model_name == None or moduleLoad[0].model_name=='' :
        category = Category()
        category.predict = ''
        category.results = 'No trained data' 
    else:   
        language=moduleLoad[0].language.code
        gen_txt_clsf_config_obj=get_classify_properties(module_id,language)
        
        '''
        stopword_v=moduleLoad[0].stopword
        stemming_v=moduleLoad[0].stemming
        remove_numbers_v=moduleLoad[0].remove_numbers
        deasciify_v=moduleLoad[0].deasciify
        remove_punkt_v=moduleLoad[0].remove_punkt
        lowercase_v=moduleLoad[0].lowercase
        wordngramrange_start=moduleLoad[0].wordngram_start
        wordngramrange_end=moduleLoad[0].wordngram_end
        
        
        gen_txt_clsf_config_obj = prepconfig.FeatureChoice(lang="tr", weights=dict(text_based=1,
                                                                        token_based=1),
                                              stopword=stopword_v, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=stemming_v,
                                              remove_numbers=remove_numbers_v, deasciify=deasciify_v, remove_punkt=deasciify_v, lowercase=lowercase_v,
                                              wordngramrange=(wordngramrange_start,wordngramrange_end), charngramrange=None,  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                                              )
        '''
        category = Category() 
        if os.path.exists(BASE_DIR+moduleLoad[0].model_name):
            modelfolder = BASE_DIR+moduleLoad[0].model_name 
        else:
            #file dosyalar cekiliyor
            f1 = default_storage.open(str(moduleLoad[0].model_name)+"_classifier", 'r')
            f2 = default_storage.open(str(moduleLoad[0].model_name)+"_model", 'r')
            #model olarak alınıyor
            model_f1 = joblib.load(f1)
            model_f2=  joblib.load(f2)
            #directory olusturuluyor
            directory_model=os.path.dirname(BASE_DIR+moduleLoad[0].model_name+"/classifier.b")
            if not os.path.exists(directory_model):
                os.mkdir(directory_model)
            #locale kaydediliyor 
            joblib.dump(model_f1, BASE_DIR+moduleLoad[0].model_name+"/classifier.b") 
            joblib.dump(model_f2, BASE_DIR+moduleLoad[0].model_name+"/model.b")    
            #model folder gosteriliyr         
            modelfolder = BASE_DIR+moduleLoad[0].model_name 
            
        
        # INITIALIZE THE SYSTEM
        gen_txt_classifier = clsf_sys.ClassificationSystem(Task=gen_txt.GeneralTextClassification,
                                                         task_name="General Text Classification",
                                                         config_obj=gen_txt_clsf_config_obj
                                                         )
        
        # 4.b- PREDICT OFFLINE (the model is loaded from the disc)
        
        if isinstance(content,list):
            test_instances=content
        else:
            test_instances=[content]
        
        
        predicted_labels, prediction_map,comparasion = gen_txt_classifier.predict_offline(modelfolder, test_instances)
        
        print("prediction map")
        print(prediction_map)
        category.predict = predicted_labels
        category.results = comparasion  
                
    return category


def train_process(clsf_system, 
         texts, labels,
         modelrootfolder, modelname,
         test_instances, test_labels,language='tr'):
    
 
        
    # 2- APPLY CROSS-VALIDATED CLASSIFICATION and GET PERFORMANCE
    accuracy, fscore, duration,root = clsf_system.get_cross_validated_clsf_performance(texts, labels, nfolds=3)
    '''
    print("en yukardan erisim kontrol")
    print("root bilgileri")
    print(root.name)
    print(root.size)
    print(root.accuracy)
    for i in range(len(root.categoryList)):
        print("ornek "+str(i))
        _ct=root.categoryList[i]
        print("categor bilgileri")
        print(_ct.name)
        print(_ct.precision)
        print(_ct.recall)
        print(_ct.accuracy)
        print(_ct.size)
        print(_ct.truePositive)
        print(_ct.falsePositive)
    '''
    #getkeyword #tr kısmı configden okunacak.
    root = infeatures.class_based_informative_features_root(texts, labels, language,root,20)
    
    
    # 3- TRAIN THE MODEL WITH THE ABOVE PARAMETERS; SAVE IT ON THE FOLDER modelrootfolder/modelname
    model, modelfolder = clsf_system.train_and_save_model(texts, labels, modelrootfolder, modelname)
    
    return root
    

def get_classify_properties(module_id,language):    
    
    moduleLoad=Module.objects.filter(id=module_id)
    
    module_type_id=moduleLoad[0].module_type_id
    stopword_v=moduleLoad[0].stopword
    stemming_v=moduleLoad[0].stemming
    remove_numbers_v=moduleLoad[0].remove_numbers
    deasciify_v=moduleLoad[0].deasciify
    remove_punkt_v=moduleLoad[0].remove_punkt
    lowercase_v=moduleLoad[0].lowercase
    wordngramrange_start=moduleLoad[0].wordngram_start
    wordngramrange_end=moduleLoad[0].wordngram_end
     
    #classification=1, sentiment =2   
    if module_type_id==1:
        gen_txt_clsf_config_obj = prepconfig.FeatureChoice(lang=language, weights=dict(text_based=1,
                                                                    token_based=1),
                                          stopword=stopword_v, more_stopwords=None, 
                                          spellcheck=False,
                                          stemming=stemming_v,
                                          remove_numbers=remove_numbers_v, deasciify=deasciify_v, remove_punkt=deasciify_v, lowercase=lowercase_v,
                                          wordngramrange=(wordngramrange_start,wordngramrange_end), charngramrange=None,  
                                          nmaxfeature=10000, norm="l2", use_idf=True,
                                          classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                                          )
    
    else:
        gen_txt_clsf_config_obj = prepconfig.FeatureChoice(lang=language, weights={"word_tfidf" : 1,
                                                                   "polyglot_value" : 0,
                                                                   "polyglot_count" : 0,
                                                                   "lexicon_count" : 1,
                                                                   "char_tfidf" : 1}, 
                                              stopword=stopword_v, more_stopwords=None, 
                                              spellcheck=False,
                                              stemming=stemming_v,
                                              remove_numbers=remove_numbers_v, deasciify=deasciify_v, remove_punkt=deasciify_v, lowercase=lowercase_v,
                                              wordngramrange=(wordngramrange_start,wordngramrange_end), charngramrange=(2,2),  
                                              nmaxfeature=10000, norm="l2", use_idf=True,
                                              classifier=sklinear.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
    
    return gen_txt_clsf_config_obj


def train_exec(traindata, delimiterParam, module_id, language='tr'):
    
    gen_txt_clsf_config_obj=get_classify_properties(module_id,language)
    
        
    f = default_storage.open(str(traindata), 'r')
    
    
    #**********
    csvsep = delimiterParam
    text_col="text"
    cat_col = "label"
    shuffle_dataset = True
    
    modelrootfolder = BASE_DIR
    modelname=str(uuid.uuid4())+"_model"
    ##modelname = "tr_spam_850sms"
    
    test_instances = ["Bugün çok güzel haberlerimiz var"]
    #test_labels = None
    test_labels = None
    
    # INITIALIZE THE SYSTEM
    gen_txt_classifier = clsf_sys.ClassificationSystem(Task=gen_txt.GeneralTextClassification,
                                                         task_name="General Text Classification",
                                                         config_obj=gen_txt_clsf_config_obj
                                                         )
    
    # READ THE DATASET
    #texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
    texts, labels = corpus_io.read_dataset_file( f, csvsep)
    #texts, labels = corpus_io.read_dataset_csv2( f, csvsep,text_col, cat_col, shuffle_dataset)
   
    
    
    root =train_process(gen_txt_classifier, 
              texts, labels,
              modelrootfolder, modelname,
              test_instances, test_labels,language)
    #**********
    
    save_model_name(module_id,modelname)
    
   
    return root


def get_module_data(module_id):
    
    labels=[]
    texts=[]
    moduleDataLoad=ModuleData.objects.filter(module_id=module_id)
    for row in moduleDataLoad:
        labels.append(row.label)
        texts.append(row.text)
    
    return texts,labels

def train_exec_module_data(module_id, language='tr'):
    
    gen_txt_clsf_config_obj=get_classify_properties(module_id,language)
    
        
    #f = default_storage.open(str(traindata), 'r')
    
    
    #**********
    #csvsep = delimiterParam
    #text_col="text"
    #cat_col = "label"
    #shuffle_dataset = True
    
    modelrootfolder = BASE_DIR
    modelname=str(uuid.uuid4())+"_model"
    ##modelname = "tr_spam_850sms"
    
    test_instances = ["Bugün çok güzel haberlerimiz var"]
    #test_labels = None
    test_labels = None
    
    # INITIALIZE THE SYSTEM
    gen_txt_classifier = clsf_sys.ClassificationSystem(Task=gen_txt.GeneralTextClassification,
                                                         task_name="General Text Classification",
                                                         config_obj=gen_txt_clsf_config_obj
                                                         )
    
    # READ THE DATASET
    #texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
    #texts, labels = corpus_io.read_dataset_file( f, csvsep)
    #texts, labels = corpus_io.read_dataset_csv2( f, csvsep,text_col, cat_col, shuffle_dataset)
    texts, labels= get_module_data(module_id)
    
    
    root =train_process(gen_txt_classifier, 
              texts, labels,
              modelrootfolder, modelname,
              test_instances, test_labels,language)
    #**********
    
    save_model_name(module_id,modelname)
    
   
    return root



def train_pre_exec(traindata, delimiterParam, module_id, language='tr'):
    
    
    gen_txt_clsf_config_obj=get_classify_properties(module_id,language)
    
     
    f = default_storage.open(str(traindata), 'r')
    
    
    #**********
    csvsep = delimiterParam
    text_col="text"
    cat_col = "label"
    shuffle_dataset = True
    
    modelrootfolder = BASE_DIR
    modelname=str(uuid.uuid4())+"_model"
    ##modelname = "tr_spam_850sms"
    
    test_instances = ["Bugün çok güzel haberlerimiz var"]
    #test_labels = None
    test_labels = None
    
    # INITIALIZE THE SYSTEM
    gen_txt_classifier = clsf_sys.ClassificationSystem(Task=gen_txt.GeneralTextClassification,
                                                         task_name="General Text Classification",
                                                         config_obj=gen_txt_clsf_config_obj
                                                         )
    
    # READ THE DATASET
    #texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
    texts, labels = corpus_io.read_dataset_file( f, csvsep)
    '''
    number_category_statu=0
    file_category = set(labels)        
    for item in file_category:
        label_count=labels.count(item)
        if label_count <5 :
            number_category_statu=1
            break  
    
    if number_category_statu==1:
        return_value=0
        return return_value, number_category_statu
    '''    
    
    
    #texts, labels = corpus_io.read_dataset_csv2( f, csvsep,text_col, cat_col, shuffle_dataset)
   
    accuracy, fscore, duration,root = gen_txt_classifier.get_cross_validated_clsf_performance(texts, labels, nfolds=3)
       
    #**********
    return_value=int(round(float(fscore)*100))
    
    print("accuracy")
    print(accuracy)
    print("fscore")
    print(fscore)
    print("duration")
    print(duration)
   
    return return_value


def save_model_name(module_id,model_name):
    moduleNew=Module.objects.filter(id=module_id)
    last_model=moduleNew[0].model_name
    moduleNew.update(model_name=model_name)
    
    #yeni model s3 e upload ediliyor
    #classifier
    file_path=BASE_DIR+model_name+"/classifier.b"
    myfile1=open(file_path,'rb')
    #model
    file_path=BASE_DIR+model_name+"/model.b"
    myfile2=open(file_path,'rb')
    #myfile = File(file_v)
    
    try:
        print("dosya s3 yazma")
        new_file_name1 = default_storage.save(model_name+"_classifier", myfile1)
        new_file_name2 = default_storage.save(model_name+"_model", myfile2)
        print("new_file_name1 : "+new_file_name1 +" - new_file_name2 : "+new_file_name2 )
    except Exception as e:
        print("dosya s3 yazma hata")
        print(e.__str__())
        return None
    
    #varsa eski model siliniyor
    #localden
    if last_model!=None and last_model!='':
        removeDirectory(BASE_DIR + str(last_model));
    #s3 den de silinmeli ...


def test():
    train_exec_module_data(31,"tr")

if __name__ == '__main__':
    
    train_exec_module_data(31,"tr")
    
    
    
    '''
    train_data_folder = "/home/user/git/cognitus-web/modules/learning/datacsv"
    train_data_fname = "tr_spam_850sms_semicolon-sep.csv"
    csvsep = ";"
    text_col="text"
    cat_col = "label"
    shuffle_dataset = True
    
    modelrootfolder = "/home/user/git/cognitus-web/modules/learning/datamodel"
    modelname = "tr_spam_850sms"
    
    test_instances = ["Bugün çok güzel haberlerimiz var",
                      "Kahrolsun faşizm",
                      "Yaşasın Nuriye ve Semih'in Direnişi",
                      "Ayaklanın! Hakkınızı Alın!"]
    #test_labels = None
    test_labels = None
    
    # INITIALIZE THE SYSTEM
    gen_txt_classifier = clsf_sys.ClassificationSystem(Task=gen_txt.GeneralTextClassification,
                                                         task_name="General Text Classification",
                                                         config_obj=gen_txt_clsf_config_obj
                                                         )
    
    # READ THE DATASET
    texts, labels = corpus_io.read_dataset_csv(train_data_folder, train_data_fname, text_col, cat_col, csvsep, shuffle_dataset)
   
   
    from text_categorization.systems import Main
    Main.main(gen_txt_classifier, 
              texts, labels,
              modelrootfolder, modelname,
              test_instances, test_labels)
   
    '''
    '''
    main(clsf_system=gen_txt_classifier, 
         modelrootfolder=modelrootfolder, modelname=modelname, 
         test_instances=test_instances, test_labels=test_labels)
    '''
    
    