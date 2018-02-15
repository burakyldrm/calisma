# -*- coding: utf-8 -*-
'''
Created on Oct 20, 2016

@author: dicle
'''

'''
Replace broken characters with the proper ones
'''

import os, codecs
from dataset import io_utils


char_map = {'ý' : 'ı',
            'Ý' : 'İ',
            'þ' : 'ş',
            'Þ' : 'Ş',
            'ð' : 'ğ',
            'Ð' : 'Ğ'        
    }


def read_encoded_file(path, encoding='ISO-8859-15'):
    f = codecs.open(path, encoding=encoding)
    rawtext = f.read()
    f.close()
    return rawtext


def fix_text(text):

    
    for broken, proper in char_map.items():
        print(broken, "  - ", proper)
        text = text.replace(broken, proper)
    
    # text = text.decode('utf-8','ignore').encode("utf-8")  # there are special chars for quotation which are redundant and can be deleted

    return text


def fix_lexicon():
    
    infolder = "/home/dicle/Documents/lexicons/tr_sentiment_boun"
    outfolder = "/home/dicle/Documents/lexicons/"
    
    for fname in io_utils.getfilenames_of_dir(infolder, False):
        p1 = os.path.join(infolder, fname)
        text = read_encoded_file(p1, encoding="utf-8")
        text = fix_text(text)
        with open(os.path.join(outfolder, fname), "w") as f:
            f.write(text) 


def fix_yildiz_tweets():
    
    infolder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/enhanced_data/orig"
    outfolder = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet/enhanced_data/fixed"
    
    folders = io_utils.getfoldernames_of_dir(infolder)
    
    for folder in folders:
        inp1 = os.path.join(infolder, folder)
        outp1 = io_utils.ensure_dir(os.path.join(outfolder, folder))
        
        files = io_utils.getfilenames_of_dir(inp1, False)
        for file in files:
            
            inp2 = os.path.join(inp1, file)
            text = read_encoded_file(inp2)
            text = fix_text(text)
            outp2 = os.path.join(outp1, file)
            with open(outp2, "w") as f:
                f.write(text)
                

def fix_yildiz_corpora():
    
    print("Reading")
    folder = "/home/dicle/Documents/data/tr/kemik/text_collection/new_fixed"
    infilename = "yildiz.txt"
    outfilename = "fixed--" + infilename
    
    inpath = os.path.join(folder, infilename)
    text = read_encoded_file(inpath)
    
    c = list(char_map.keys())[0]
    print(c, "   ", list(char_map.keys())[0] in text) 
    
    t = text.replace(c, char_map[c])
    print(t[:100])
       
    text = fix_text(text)
    print(text[:100])
    
    outpath = os.path.join(folder, outfilename)
    with open(outpath, "w") as f:
        f.write(text)
    
    print("Finished.")
    
    

if __name__ == '__main__':
    
    # fix_lexicon()
    fix_yildiz_tweets()
    
    
    
