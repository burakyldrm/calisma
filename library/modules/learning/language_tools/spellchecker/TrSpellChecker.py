"""Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""
'''
edited for tr. 1) tranined with tr dataset; 2) letters in edit1() are tr characters; 3) uppercase/mixed case handling added.
dicle
'''
################ Spelling Corrector 
'''
from collections import Counter

TEXTSPATH = "/home/dicle/Documents/data/tr/tr_gazete_siir/tr_text_compilation.txt"
#def words(text): return re.findall(r'\w+', text.lower())
def words(text): return re.findall(r'\w+', text)
#def words(text): return text.split()
WORDS = Counter(words(open(TEXTSPATH).read()))
'''

import os
import re
import string
from time import time

from modules.learning.language_tools.spellchecker import dump_dict


class TrSpellChecker():

    WORDS = None

    TEXTSPATH = "/home/dicle/Documents/data/tr/tr_gazete_siir/tr_text_compilation.txt"
    pWORDSPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WORDS.p")

    def __init__(self):
    
        self.WORDS = dump_dict.getWORDS(self.pWORDSPATH)
        



    def P(self, word):  # N=sum(WORDS.values())
        "Probability of `word`."
        
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N
    
    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)
    
    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
    
    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)
    
    def edits1(self, word):
        "All edits that are one edit away from `word`."
        if word.isupper() or word.istitle():
            letters = 'ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ'
        else:
            letters = 'abcçdefgğhıijklmnoöprsştuüvyz'
        splits = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    '''
    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcçdefgğhıijklmnoöprsştuüvyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    '''
    
    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    
    def correct_text(self, text):
    
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
            
            suggested_word = self.correction(no_punct_word)
          
            
            correct_word = lpunct + suggested_word + rpunct
            
            correct_words.append(correct_word)
            
        correcttext = " ".join(correct_words)   
        return correcttext


################ Test Code 

'''
def unit_tests():
    assert correction('speling') == 'spelling'              # insert
    assert correction('korrectud') == 'corrected'           # replace 2
    assert correction('bycycle') == 'bicycle'               # replace
    assert correction('inconvient') == 'inconvenient'       # insert 2
    assert correction('arrainged') == 'arranged'            # delete
    assert correction('peotry') =='poetry'                  # transpose
    assert correction('peotryy') =='poetry'                 # transpose + delete
    assert correction('word') == 'word'                     # known
    assert correction('quintessential') == 'quintessential' # unknown
    assert words('This is a TEST.') == ['this', 'is', 'a', 'test']
    assert Counter(words('This is a test. 123; A TEST this is.')) == (
           Counter({'123': 1, 'a': 2, 'is': 2, 'test': 2, 'this': 2}))
    assert len(WORDS) == 32198
    assert sum(WORDS.values()) == 1115585
    assert WORDS.most_common(10) == [
     ('the', 79808),
     ('of', 40024),
     ('and', 38311),
     ('to', 28765),
     ('in', 22020),
     ('a', 21124),
     ('that', 12512),
     ('he', 12401),
     ('was', 11410),
     ('it', 10681)]
    assert WORDS['the'] == 79808
    assert P('quintessential') == 0
    assert 0.07 < P('the') < 0.08
    return 'unit_tests pass'
'''

'''
def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))
    
def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong) 
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

'''



if __name__ == '__main__':
   
    spellchecker = TrSpellChecker()
    sentence = "gostericiye bıletimi veryorum."
    words = sentence.split()
    s2 = " ".join([spellchecker.correction(w) for w in words])
    
    print(s2)
