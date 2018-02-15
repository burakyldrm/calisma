# encoding: utf-8

'''
Created on Oct 18, 2016

@author: dicle
'''

from modules.custom_classify import SimpleTokenHandler

import sklearn.naive_bayes as nb
import sklearn.cross_validation as cv
import sklearn.feature_extraction.text as txtfeatext
import sklearn.pipeline as skpipeline
import sklearn.metrics as mtr
import codecs
from sklearn.externals import joblib
import nltk
import os,sys,traceback,uuid
#from serdoo.models import query_log
import hashlib
from django.core.files.storage import default_storage
from django.conf import settings
from user.models import Module
from django.core.files import File

#BASE_DIR = '/code/trained_data/'
BASE_DIR = os.path.join(settings.BASE_DIR, '../trained_data/')


class Category:
    def __init__(self):
        self.name = ''
        self.description=''
        self.size = 3
        self.accuracy =0
        self.precision =0
        self.recall =0
        self.truePositive=0
        self.trueNegative=0
        self.falsePositive=0
        self.falseNegative=0
        self.keywords = ''    
        
class Root:
    def __init__(self, catSize):
        self.categoryList =  [Category() for i in range(catSize)]
        self.size = 0
        self.accuracy =0
        self.precision =0
        self.recall =0
        self.truePositive=0
        self.trueNegative=0
        self.falsePositive=0
        self.falseNegative=0
        self.keywords = ''
        
        
class Statistic:
    
    def __init__(self):
        self.descriptionMap = {}
        self.topicMap = {}
        self.precisionMap = {}
        self.recallMap = {}
        self.f1ScoreMap = {}
        self.supportMap = {}
        self.unique = []
        self.topic = [] 
        self.question = []
        self.module_id=0
        self.clf=None
        self.vectorizer=None
    
def checkKeyValuePairExistence(dic, key, value):
    try:
        dic[key] += 1
        return dic
    except KeyError:
        dic[key] = 1
        return dic
        
def checkKeyValuePairDescriptionExistence(dic, key, value):
    try:
        if dic.has_key(key):
            return dic
        else:
            dic[key] = value
            return dic
    except KeyError:
        dic[key] = value
        return dic
    
def roundFloat(val):
    return int(round(val*100))    


def execute(traindata, delimiterParam, module_id, language='tr'):
    
    removeFile(str(module_id) + "_output_model");
    removeFile(str(module_id) + "_output_vectorizer");
    
    statistic = Statistic()
    statistic.module_id = module_id
    get_data(traindata, delimiterParam, statistic)
    return classify_texts(statistic, language)

    
def process_cm(confusion_mat, i, categoryName, to_print=False):
    print("**************************************************************************")
    print(categoryName)
    try:
        to_print=True
        TP = confusion_mat[i,i]  # correctly labeled as i
        FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
        FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
        TN = confusion_mat.sum().sum() - TP - FP - FN
        if to_print:
            print('TP: {}'.format(TP))
            print('FP: {}'.format(FP))
            print('FN: {}'.format(FN))
            print('TN: {}'.format(TN))
        return TP, FP, FN, TN
    except:
        return 0,0,0,0


def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=20):
    print("classifier")
    print(classifier)
    try:
          print("classifier.classes_")
          print(classifier.classes_)
          if classlabel in list(classifier.classes_):
              labelid = list(classifier.classes_).index(classlabel)
              feature_names = vectorizer.get_feature_names()
              topn = sorted(zip(classifier.named_steps['clf'].coef_[labelid], feature_names))[-n:]
              keywordList = ""
              for coef, feat in topn:   
                  keywordList += ", " + feat+":" +str(round(float(abs(coef))))     
        
              #print("descriptin yazilior ---------------------------------")
              #print(labelid)
              #print(keywordList.encode(encoding='utf_8', errors='strict'))
              return keywordList
          else:
             return ""
    except (IndexError):
      return ""
    
    
def get_data(traindata, delimiterParam, statistic):
    f = default_storage.open(str(traindata), 'r')
    #f = codecs.open(traindata, encoding='utf-8')
   
    for line in f:
            line_cl=line.decode().rstrip("\;\n")
            #data = line.decode().split(delimiterParam)
            data = line_cl.rsplit(delimiterParam,1)
            #print(data)
            category = data[1].replace(' ', '').encode('utf-8')
            statistic.topic.append(category)

            statistic.topicMap = checkKeyValuePairExistence(statistic.topicMap, category, 1)
            
            head, sep, tail = data[0].partition(u"İyi çalışmalar")
            head, sep, tail = head.partition(u"İyi çalışmalar.")
            head, sep, tail = head.partition(u"Bu e-posta ve ekleri")
            head, sep, tail = head.partition(u"İyiçalışmalar")
            head, sep, tail = head.partition(u"Bilginize")
            head, sep, tail = head.partition(u"Saygılarımla")
            head, sep, tail = head.partition(u"saygılarımla")
            head, sep, tail = head.partition(u"Saygılarımla.")
            head, sep, tail = head.partition(u"Saygılarımızla")
            head, sep, tail = head.partition(u"Saygilarimizla")
            head, sep, tail = head.partition(u"Saygilarimizla.")
            head, sep, tail = head.partition(u"SAYGILARIMIZLA")
            head, sep, tail = head.partition(u"Saygılarımızla.")
            head, sep, tail = head.partition(u"Bu e-posta")
            head, sep, tail = head.partition(u"Bu mesaj")
            head, sep, tail = head.partition(u"Disclaimer:")
            head, sep, tail = head.partition(u"***")
            head, sep, tail = head.partition('-->')
            head, sep, tail = head.partition(u"Teşekkürler")
            head, sep, tail = head.partition(u"Kolay Gelsin")
            head, sep, tail = head.partition('**********************************************************************************************')
            head, sep, tail = head.partition(u'İYİ ÇALIŞMALAR.')
            head, sep, tail = head.partition(u"Mail-imza")
            head, sep, tail = head.partition(u"UYARI")
            head, sep, tail = head.partition(u"Yasal Uyarı")
            head, sep, tail = head.partition(u"elektronik posta")
            
            statistic.question.append(head)
            #hashCode = hashlib.md5(data[1].strip().encode('utf-8')).hexdigest();
            
            #query_logSize = query_log.objects.filter(query=hashCode).filter(module_id=statistic.module_id).count()

            #if (int(query_logSize) > 0):
             #   queryData = query_log.objects.filter(query=hashCode).filter(module_id=statistic.module_id)
             #   queryDataTemp = queryData[0]
             #   queryDataTemp.changed_predict=category
             #   queryDataTemp.save();
                  
            #if (len(data) == 3) :
            # statistic.descriptionMap = checkKeyValuePairDescriptionExistence(statistic.descriptionMap, category, data[2])
             
    sequence = statistic.topic

    [statistic.unique.append(item) for item in sequence if item not in statistic.unique]
    print(statistic.unique)
    
    
def get_performance(ytrue, ypred, statistic):
    
    root = Root(len(statistic.unique))
    acc = mtr.accuracy_score(ytrue, ypred)
    
    root.accuracy =roundFloat(acc)
    
    confmatrix = mtr.confusion_matrix(ytrue, ypred, labels=statistic.unique)
    
    print("confmatix")
    print(confmatrix)
    report = mtr.classification_report(ytrue, ypred)
    print("report")
    print(report)

    classification_report(report, statistic);
    
    subCategory(confmatrix, statistic, root)
    return root

def subCategory(confmatrix, statistic, root):
    i = 0   
    for uni in statistic.unique:
       # uni = str(uni.replace(' ', '')).encode('utf-8')
        unikey = str(uni.decode().replace(' ', '').encode('utf-8'))
        root.categoryList[i].name = uni;
        if statistic.descriptionMap != None and unikey in statistic.descriptionMap:
            root.categoryList[i].description = statistic.descriptionMap[unikey];
        else :
            root.categoryList[i].description = uni
        try:
            root.categoryList[i].precision =roundFloat(float(statistic.precisionMap[unikey]));
            root.categoryList[i].recall =roundFloat(float(statistic.recallMap[unikey]));
            root.categoryList[i].accuracy = roundFloat(float(statistic.f1ScoreMap[unikey]));            
        except:           
            root.categoryList[i].precision = 0;
            root.categoryList[i].recall = 0;
            root.categoryList[i].accuracy=0;
            #root.categoryList[i].size = 10;
            
        calculatedValues = process_cm(confmatrix, i, unikey,  to_print=True); 
        root.categoryList[i].truePositive = calculatedValues[0];
        root.categoryList[i].trueNegative = calculatedValues[3];
        root.categoryList[i].falsePositive = calculatedValues[1];
        root.categoryList[i].falseNegative = calculatedValues[2];
        root.categoryList[i].size = calculatedValues[0] + calculatedValues[2];
    
        root.categoryList[i].keywords =  most_informative_feature_for_class(statistic.vectorizer, statistic.clf, uni, n=20)
        root.size += root.categoryList[i].size
        
        i+=1
    
def classification_report(report, statistic):
    
    lines = report.split('\n')
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        className = str(t[0].replace(' ', ''))
        
        statistic.precisionMap[className] = t[1]
        statistic.recallMap[className] = t[2]
        statistic.f1ScoreMap[className] = t[3]
        statistic.supportMap[className] = t[4]
        
def removeFile(fileName):
    try:
        os.remove(fileName)
    except OSError:
        pass

def _classify_tr_texts(statistic):
    
    classifier = nb.MultinomialNB()   # object
    stopword_choice = True  # boolean
    ngramrange = (1, 2)   # tuple
    nmaxfeature = 10000   # int or None
    
    tokenizer = SimpleTokenHandler.TRSimpleTokenHandler(stopword=True, stemming=True)
                
    txt_vect = txtfeatext.CountVectorizer(tokenizer=tokenizer, 
                                                        ngram_range=ngramrange,
                                                        max_features=nmaxfeature)  
    txt_transformer = txtfeatext.TfidfTransformer()

    text_clf = skpipeline.Pipeline([('vect', txt_vect),
                                ('tfidf', txt_transformer),
                                ('clf', classifier),
                               ])  
    
    statistic.vectorizer = txt_vect
    
    statistic.clf = text_clf
    
    return _run_classifier(text_clf, statistic)

def _classify_en_texts(statistic):
    
    classifier = nb.MultinomialNB()   # object
    stemming_choice = True  # boolean 
    stopword_choice = True  # boolean
    ngramrange = (1, 2)   # tuple
    nmaxfeature = 10000   # int or None
    
    tokenizer = SimpleTokenHandler.ENSimpleTokenHandler(stem=stemming_choice, stopword=stopword_choice)
                
    txt_vect = txtfeatext.CountVectorizer(tokenizer=tokenizer, 
                                                        ngram_range=ngramrange,
                                                        max_features=nmaxfeature)  
    statistic.vectorizer = txt_vect

    txt_transformer = txtfeatext.TfidfTransformer()

    text_clf = skpipeline.Pipeline([('vect', txt_vect),
                                ('tfidf', txt_transformer),
                                ('clf', classifier),
                               ])  
    
    statistic.clf = text_clf
    
    return _run_classifier(text_clf, statistic)


def save_model_name(module_id,model_name):
    moduleNew=Module.objects.filter(id=module_id)
    last_model=moduleNew[0].model_name
    moduleNew.update(model_name=model_name)
    
    #yeni model s3 e upload ediliyor
    file_path=BASE_DIR+model_name
    myfile=open(file_path,'rb')
    #myfile = File(file_v)
    try:
        print("dosya s3 yazma")
        new_file_name = default_storage.save(model_name, myfile)
        print("new_file_name : "+new_file_name)
    except Exception as e:
        print("dosya s3 yazma hata")
        print(e.__str__())
        return None
    
    #varsa eski model siliniyor
    #localden
    if last_model!=None and last_model!='':
        removeFile(BASE_DIR + str(last_model));
    #s3 den de silinmeli ...
    
    
    

def _run_classifier(pipeline_clf, statistic):

    X_train, X_test, y_train, y_test = cv.train_test_split(statistic.question, statistic.topic, test_size=0.30, random_state=20)
    
    pipeline_clf.fit(X_train, y_train)
    
    modelname=str(uuid.uuid4())+"_model"
    
    joblib.dump(pipeline_clf, BASE_DIR+modelname)
    
    #save model name to db
    save_model_name(statistic.module_id,modelname)
    
    
    ypred = pipeline_clf.predict(X_test)
    root = get_performance(y_test, ypred, statistic)
    return root


def classify_texts(statistic, lang):

    if lang in ["en", "english"]:
      return _classify_en_texts(statistic)
    if lang in ["tr", "turkish"]:
      return _classify_tr_texts(statistic)

if __name__ == '__main__':

    module_id=40
    lang = "tr"
    path = "/code/data/tip.csv"  # CHANGE
    delimiter = ";"
    root = execute(path, delimiter, module_id)
    print(root)
    print("Root Bilgileri")
    print(root.size)
    print(root.accuracy)
    print(root.precision)
    
    for category in root.categoryList:
        print("*******************************************************************************")
        print("category name")
        print(category.name)
        print("category size")
        print(category.size)
        print("precision")
        print(category.precision)
        print("recall")
        print(category.recall)
        print("TP")
        print(category.truePositive)
        print("TN")
        print(category.trueNegative)
        print("FP")
        print(category.falsePositive)
        print("FN")
        print(category.falseNegative)
        print("*******************************************************************************")
        