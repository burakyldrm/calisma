'''
Created on Oct 21, 2016

@author: dicle
'''

import string

import autocorrect

from language_tools.spellchecker import tr_spell


def correct_text(text, lang):
    
    words = text.split()
    
    punkts = string.punctuation
    
    lpunct = ""  # to remove the punctuation upon spelling correction and put back them afterwards 
    rpunct = ""  # leading:1 char and ending: 3chars
    
    npunct1 = 0
    npunct2 = 0
    
    correct_words = []
    
    for w in words:
        
        lw = w.lstrip(punkts)  # remove leading punctuation
        npunct1 = len(w) - len(lw)  # take the difference to put the punkts back if not 0
        lpunct = w[:npunct1]
        
        rw = w.rstrip(punkts)
        npunct2 = len(w) - len(rw)
        if npunct2 > 0:  # otherwise the slicer selects the whole string
            rpunct = w[-npunct2:]
        
        no_punct_word = w.strip(punkts)
        if lang == "tr":
            suggested_word = tr_spell.correction(no_punct_word)
        if lang == "en":
            suggested_word = autocorrect.spell(no_punct_word)
        
        correct_word = lpunct + suggested_word + rpunct
        
        correct_words.append(correct_word)
        
    correcttext = " ".join(correct_words)   
    return correcttext



if __name__ == '__main__':
    
    sentence = "gostericiye, 'bıletimi' veryorum."

    print(correct_text(sentence, lang="tr"))
    # print(correction("gosteriye"))
    
    print(correct_text("'merhaba müşterimiz çiftetelli Kampanyası_90 Gün kampanyadan yararlanmaktadır paketi Yalın ULTRANET LiMiTSiZ 75 ADSL müşterimiz çiftetelli Kampanyası_90 Gün kampanyasından Yalın ULTRANET LiMiTSiZ 100 geçiş yapmak istemektedir paket çıkmaktadır konu hakkında yardımlarımızı bekliyorum'", lang="tr"))

    
    
    
