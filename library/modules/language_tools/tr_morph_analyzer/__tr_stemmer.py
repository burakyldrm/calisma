'''
Created on Mar 17, 2017

@author: dicle
'''

import json, os, sys, fcntl, re, getopt, io

import scripts.disambiguate as morphdis


def _stem_word(word):
        
    #####onlybest = True  ######
    model_file = 'scripts/1M.m2'
   
    flookup_cmd="../linux64/flookup -b -x ../trmorph.fst"
     
    
    try:
        mfile = open(model_file, 'r', encoding='utf-8')
    except:
        print("Cannot open the morphological analysis model file.")
        print("Run `{} --help' for help.".format(sys.argv[0]))
        sys.exit(-1)
    model = json.load(mfile, encoding='utf-8')
    mfile.close()
    
    trmorph = morphdis.flookup_open(flookup_cmd)
    

    word = word.strip()
    
    if len(word) == 0:
        return word
    
    alist = morphdis.get_analyses(trmorph, word)

    if len(alist) == 0:
        slist = [(-1, '???')]
    else:
        slist = morphdis.score_astrings(model, alist)

   
    
    (score, root) = slist[0]
    
    
    tag_pattern = r"\<(\w+:?\w*)+\>"  # "\<\w{1,4}\>"
    root = re.sub(tag_pattern, "", root)

    
    if root.endswith("?"):  # no root could be found, especially for NEs.
        root = word
                
    return root

def stem_word(word):
    
    return morphdis.stem_word(word)

if __name__ == '__main__':
    


    sentence = """Seher vakti habersizce girdi gara ekspres
    kar içindeydi
    ben paltomun yakasını kaldırmış perondaydım
    peronda benden başka da kimseler yoktu """
    
    words = sentence.split()
    
    for w in words:
        root = stem_word(w)
        print(w, root)
        
        
        
        

