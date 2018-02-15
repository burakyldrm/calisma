'''
Created on Dec 21, 2016

@author: dicle
'''

import sys
sys.path.append("..")

import os
import re
import subprocess
from time import time

import snowballstemmer
from polyglot.text import Word

from modules.learning.dataset import corpus_io
from modules.learning.language_tools import TOOL_CONSTANTS

#import tr_morph_analyzer.scripts.disambiguate as tr_morph

import modules.learning.language_tools.tr_morph_analyzer.scripts.disambiguate as tr_morph

###### using method: https://github.com/coltekin/TRmorph   ####
# -- faster version in langauge_tools/tr_morph_analyzer; using scripts from https://github.com/coltekin/TRmorph.
def stem(word):
    
    return tr_morph.stem_word(word)


def stem_words(word_list):
    
    return tr_morph.stem_word_list(word_list)   


def stem_words_in_text(text):
    words = text.split()
    return stem_words(words)
##################################################

'''
method: https://github.com/coltekin/TRmorph
--slower version
'''
def _stem(word):                                                            
    # return  subprocess.Popen("echo '" + word + "' | flookup Documents/tools/tr_morph/coltekin/TRmorph/stem.fst", shell=True, stdout=subprocess.PIPE).stdout.read().split()
    
    # problems with apostrophe
    apost_pattern = r"[\"'’´′ʼ]"
    w = re.sub(apost_pattern, "", word)
    
    '''
    items = subprocess.Popen("echo '" + w + "' | flookup " + TOOL_CONSTANTS.PATH_TO_STEM_FST, 
                             shell=True, stdout=subprocess.PIPE).stdout.read().split()
    '''
    proc = subprocess.Popen("echo '" + w + "' | flookup " + TOOL_CONSTANTS.PATH_TO_STEM_FST,
                             shell=True, stdout=subprocess.PIPE).stdout
    items = proc.read().split()
    proc.close()
    items = [str(i, "utf-8") for i in items]
    # print(items)
    root = items[-1]
    
    tag_pattern = r"\<(\w+:?\w*)+\>"  # "\<\w{1,4}\>"
    root = re.sub(tag_pattern, "", root)

    
    if root.endswith("?"):  # no root could be found, especially for NEs.
        return word
    else:
        return root




def stem2(word):
    
    
    stemmer = snowballstemmer.stemmer("turkish")
    stemmed = stemmer.stemWord(word)
    
    if stemmed == "fatur":
        stemmed = "fatura"
    elif stemmed == "hatt":
        stemmed = "hat"
    
    return stemmed


# hasim sak's morphological analyser
def stem3(word):

    ''' # doesn't work
    #command = "python2 " + TOOL_CONSTANTS.PATH_TO_SAK_PARSER_PY + " " + word
    command = "python2 /home/dicle/Documents/tools/tr_morph/boun_hasim-sak_morph-parser/MP-1.0-Linux64/tr_morph_analyser.py geliyorum" 
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
    items = proc.read().split()
    proc.close()
    return items
    # '''
    

def stem4(word):
    w = Word(word, language="tr")
    return w.morphemes[0]

if __name__ == '__main__':
    
    '''
    words = ['Bu', "faturalar","faturalar", "Faturan",
 'mesajın',
 'gönderilmek',
 'istendiği',
 'kişi',
 'değilseniz',
 '(ya',
 'da',
 'bu',
 "e-posta'yı",
 'yanlışlıkla',
 'aldıysanız),',
 'lütfen',
 'yollayan',
 'kişiyi',
 'haberdar',
 'ediniz']

    from time import time
    t0 = time()
    for w in words:
        print(stem(w))
    
    t1 = time()
    
    roots = stem_words(words)
    t2 = time()
    
    print(t1-t0)
    print(t2-t1)
    '''
    
    #print(_stem("istiyoruz"))
    
    
    # test stemming 200 docs.
    
    '''
    folderpath = "/home/dicle/Documents/data/emailset2"
    fname = "only_complaint2_clean_hizmetno.csv"
    fpath = os.path.join(folderpath, fname)
    text_col = "MAIL"
    cat_col = "TIP"

    start_t = time()
    
    instances, labels = corpus_io.get_emails(os.path.join(folderpath, fname), 
                                             sep=";", textcol=text_col, catcol=cat_col)
    
    print(len(instances))
    N = 1300
    instances, _ = corpus_io.select_N_instances(N, instances, labels)
    
    roots = []
    # 1. straight
    roots = [stem(w) for text in instances for w in text.split()]
    print(len(roots))
    
    end_t = time()
    print("Took ", str(end_t-start_t),"sec.")
    
    # 2. inside trtokenhandler
    '''
    
    
    '''
    s = ["merhaba", "merhabalar"]
    for i in s:
        ss = stem(i)
        if i == ss:
            print(ss)
        print(i, " ", ss)
    
    
    t = "Bu arada farkettim de hepsi Turkcell Kullanıyor :)"
    words = t.split()
    roots = stem_words(words)
    print(roots)
    '''
    
    from dataset import corpus_io
    folder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/"
    fname = "tr_polartweets.csv"
    csvsep="\t"
    text_col="text"
    cat_col="polarity"
    texts, _ = corpus_io.read_labelled_texts_csv(os.path.join(folder, fname), 
                                                 csvsep, text_col, cat_col, shuffle=True)
    
    Ns = [20, 50, 100, 200, 500, 1000, 2000]
    ts = []
    

    for N in Ns:
        stexts = texts[:N]
        words = []
        for t in stexts:
            words.extend(t.split())
        print(len(words))   
        t0 = time()    
        roots = stem_words(words)
        t1 = time()
        ts.append((N, t1-t0))
        print(N, t1-t0)
    
    print(ts)
    
    for N in Ns:
        stexts = texts[:N]
                    
        t0 = time()    
        #roots = [x.tokenize(stext) for stext in stexts]
        #roots = [tokenize(stext) for stext in stexts]
        #roots = [tr_stemmer.stem_words(t.split()) for t in stexts]
        for t in stexts:
            words = t.split()
            print(words[:5])
            stem_words(words)
        t1 = time()
        ts.append((N, t1-t0))
        print(N, t1-t0)
    
    print(ts)
    
