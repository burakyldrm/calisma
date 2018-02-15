# encoding: utf-8

from sklearn.externals import joblib
import os,json
from user.models import Module
import hashlib
from django.conf import settings
from django.core.files.storage import default_storage
#BASE_DIR = '/code/trained_data/'
BASE_DIR = os.path.join(settings.BASE_DIR, '../trained_data/')


class Category:
    predict = ''
    results = ''
    
def roundFloat(val):
    return int(round(val*100))   
   
def extract_time(json):
    try:
        # Also convert to int since update_time will be string.  When comparing
        # strings, "10" is smaller than "2".
        return int(json['probability'])
    except KeyError:
        return 0
       
def getValues(classList, lists, categoryMap):
    comparasion = '{\"results\":['
    i = 0
    for className in classList:
             
             if i != 0:
                 comparasion += ','
             if (className != None and className != ''):     
                 #comparasion +=  "{ \"category\":\"" + (str(className.decode())+ "\"," + " \"probability\": " + str(roundFloat(lists[i]))) + ", \"description\": \"" + str(categoryMap[className.decode()]) + "\"}"
                 comparasion +=  "{ \"category\":\"" + (str(className.decode())+ "\"," + " \"probability\": " + str(roundFloat(lists[i]))) + "\"}"                 
                 i+=1
             
    comparasion += ']}'
    
    return comparasion
    
def execute(module_id,  content, categoryMap):
    moduleLoad=Module.objects.filter(id=module_id)
    if moduleLoad[0].model_name == None or moduleLoad[0].model_name=='' :
        category = Category()
        category.predict = ''
        category.results = 'No trained data' 
    else:    
        if os.path.exists(BASE_DIR+moduleLoad[0].model_name):
            model = joblib.load(BASE_DIR+moduleLoad[0].model_name) 
        else:
            f = default_storage.open(str(moduleLoad[0].model_name), 'r')
            model_storage = joblib.load(f) 
            joblib.dump(model_storage, BASE_DIR+moduleLoad[0].model_name)
            model = joblib.load(BASE_DIR+moduleLoad[0].model_name)
            
        
        X_test = [content]
        predictProbe  = model.predict_proba(X_test)
        predict  = model.predict(X_test)
        result1= sorted(zip(model.classes_, predictProbe))
        lists = (result1[0])[1]
        results = getValues(model.classes_, lists, categoryMap)
        category = Category()
        category.predict = predict
        category.results = results  
                
    return category