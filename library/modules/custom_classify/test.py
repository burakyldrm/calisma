from django.core.files.storage import default_storage
import pandas as pd

if __name__ == '__main__':
    f = '/home/user/git/cognitus-web/modules/custom_classify/datacsv/tr_spam_850sms_semicolon-sep.csv'
    #pathfile= default_storage.path(str(traindata))
    #f = codecs.open(traindata, encoding='utf-8')
    df=pd.read_csv(f,';',header=None,usecols=[0,1])
    instances=df[0]
    labels=df[1]
    
    #print(len(labels))
    #print(labels)
    b=set(labels)
    print(b)
    
    #for i in range(0,len(df)):
    #    print(str(i)+"-data1:"+instances[i]+"-data2:"+labels[i])
        
         
