
import sys,os

sys.path.append('/home/user/git/cognitus-web')

os.environ['DJANGO_SETTINGS_MODULE'] = 'cognitus.settings.dev'
import django
django.setup()
import modules.learning.text_categorization.systems.GeneralTextClassification as manager

import modules.sentiment.tr_sentiment_classification as sentmanager
from user.models import ModuleData


if __name__ == '__main__':
    
    sentmanager.sentimentTrain()
    
    '''
    labels=[]
    texts=[]
    module_id=31    
    moduleDataLoad=ModuleData.objects.filter(module_id=module_id)
    for row in moduleDataLoad:
        labels.append(row.label)
        texts.append(row.text)
    
    
    print(labels)
    '''