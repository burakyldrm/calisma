'''
Created on May 1, 2017

@author: dicle
'''

import os, re
import pandas as pd


def sentiturknet_to_lexicon():
    
    sentipath = "/home/dicle/Documents/lexicons/SentiTurkNet/STN.xlsx"
    outfolder = "/home/dicle/Documents/lexicons/tr_sentiment_boun/sentiturknet_split"
    df = pd.read_excel(sentipath)
    g = df.groupby(["Polarity Label"])
    
    for i in "opn":
        d = g.get_group(i)
        
        polarity_list = []
        for j in d.index.values.tolist():
            
            synset = str(d.loc[j, "synonyms"])
            postag = str(d.loc[j, "POS tag"])
            words = synset.split(",")
            
            
            for w in words:
                polarity_list.append(w.strip().lower()+"#"+postag)
        
            polarity_list = [w for w in polarity_list if not w.startswith("nan#")]
            polarity_list = list(set(polarity_list))
            polarity_list.sort()
            open(os.path.join(outfolder, i+".txt"), "w").write("\n".join(polarity_list))





def join_lexicons():
    
    bounfolder = "/home/dicle/git/serdoo-servis2/django_docker/learning/_lexicons/tr_sentiment_boun"
    sentifolder = "/home/dicle/Documents/lexicons/tr_sentiment_boun/sentiturknet_split"
    names = ["positive", "negative"]

    for n in names:
        boun = open(os.path.join(bounfolder, n+".txt"), "r").readlines()
        boun = [w.strip() for w in boun]
        senti = open(os.path.join(sentifolder, n+".txt"), "r").readlines()
        senti = [w.strip() for w in senti]
        
        boun_words = [w.split("#")[0] for w in boun]
        newlist = [w for w in senti if w.split("#")[0] not in boun_words]
        newlist = list(set(newlist))
        newlist.sort()
        open(os.path.join(bounfolder, n+"_n2.txt"), "w").write("\n".join(newlist))
        
        
    
def remove_gerund():
    
    bounfolder = "/home/dicle/git/serdoo-servis2/django_docker/learning/_lexicons/tr_sentiment_boun"
    names = ["positive", "negative"]
    
    for n in names:
        boun = open(os.path.join(bounfolder, n+".txt"), "r").readlines()
        boun = [w.strip() for w in boun]
        
        
        _newlist = [re.sub(r"m[ae]k\#v$", "", w) for w in boun]
        newlist = []
        for w in _newlist:
            if "#" not in w:
                w = w + "#v"
            newlist.append(w)
                
        import icu
        collator = icu.Collator.createInstance(icu.Locale('tr_TR.UTF-8'))
        newlist.sort(key=collator.getSortKey)
        open(os.path.join(bounfolder, n+"_n2.txt"), "w").write("\n".join(newlist))
    


def replace_gerund():
    bounfolder = "/home/dicle/git/serdoo-servis2/django_docker/learning/_lexicons/tr_sentiment_boun"
    names = ["positive_n2", "negative_n2"]
    
    for n in names:
        boun = open(os.path.join(bounfolder, n+".txt"), "r").readlines()
        boun = [w.strip() for w in boun]
                
        _newlist = [re.sub("\set(tir)?(me)?\#[nvpb]", "", w).strip() for w in boun]
        _newlist = [re.sub("\syap(tır)?(ma)?\#[nvpb]", "", w).strip() for w in _newlist]
    
        newlist = []
        for w in _newlist:
            if "#" not in w:
                w = w + "#n"
            newlist.append(w)
        
        newlist = list(set(newlist))
        import icu
        collator = icu.Collator.createInstance(icu.Locale('tr_TR.UTF-8'))
        newlist.sort(key=collator.getSortKey)
        open(os.path.join(bounfolder, n+"_n4.txt"), "w").write("\n".join(newlist))   
        

if __name__ == '__main__':
    
    print()
    
    #sentiturknet_to_lexicon()
    #join_lexicons()
    #remove_gerund()
    replace_gerund()
    
    