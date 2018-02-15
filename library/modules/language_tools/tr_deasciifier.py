'''
Created on Dec 2, 2016

@author: dicle
'''

import string

from turkish.deasciifier import Deasciifier

from dataset import io_utils


# checks if a word has only punctuation char.s
def is_punctuation(word):
    punkts = string.punctuation
    tested_chars = [i for i in word if i in punkts]
    return len(word) == len(tested_chars)


def deasciify_tr_text(text):

    words = text.split()
    
    punkts = string.punctuation
    
    npunct1 = 0
    npunct2 = 0
    
    correct_words = []
    
    for w in words:
        
        lpunct = ""  # to remove the punctuation upon spelling correction and put back them afterwards 
        rpunct = ""  # leading:1 char and ending: 3chars
        correct_word = ""
        
        if is_punctuation(w):
            correct_word = w
        else: 
        
            lw = w.lstrip(punkts)  # remove leading punctuation
            npunct1 = len(w) - len(lw)  # take the difference to put the punkts back if not 0
            lpunct = w[:npunct1]
            
            rw = w.rstrip(punkts)
            npunct2 = len(w) - len(rw)
            if npunct2 > 0:  # otherwise the slicer selects the whole string
                rpunct = w[-npunct2:]
            
            no_punct_word = w.strip(punkts)
            
            suggested_word = Deasciifier(no_punct_word).convert_to_turkish() 
            correct_word = lpunct + suggested_word + rpunct
        
        correct_words.append(correct_word)
        
        '''
        print(w, len(w), no_punct_word, len(no_punct_word))
        print("l:", lpunct, "-s:", suggested_word, "-r:", rpunct)
        print("####\n")
        '''
        
    correcttext = " ".join(correct_words)   
    return correcttext


def deasciify_df(df, text_col):

    texts = df[text_col].tolist()
    texts = [deasciify_tr_text(text) for text in texts]
    df[text_col] = texts
    return df


def deasciify_word(word):
    return Deasciifier(word).convert_to_turkish()

if __name__ == '__main__':
    
    t = "sozlesmesini yaptiginiz aramadiginiz, aramalarina donmediginiz! aboneleriniz dahil mi buna?"
    
    print(deasciify_tr_text(t))
    
    print(deasciify_word("Şehir".lower()))
    
    #===========================================================================
    # # Deasciify tweets
    # import os
    # '''
    # folderpath = "/home/dicle/Documents/data/tr_sentiment/sentiment-3000tweet"
    # fname1 = "polar3000tweets.csv"
    # fname2 = "polar3000tweets_deasciified.csv"
    # textcol="text"
    # '''
    # 
    # folderpath = "/home/dicle/Documents/data/tcell_email/cleaning"
    # fname1 = "df_clean1.csv"
    # fname2 = "df_clean1_deasc.csv"
    # textcol = "clean_text_regex"
    # 
    # df = io_utils.readcsv(os.path.join(folderpath, fname1))
    # df = deasciify_df(df, text_col=textcol)
    # io_utils.tocsv(df, os.path.join(folderpath, fname2))
    #===========================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
