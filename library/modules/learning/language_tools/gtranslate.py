'''
Created on Mar 10, 2017

@author: dicle
'''

import sys
sys.path.append("..")

import textblob

from translate import translator


def translate_textblob(source, _from="ar", _to="en"):
    
    s = textblob.TextBlob(source)
    target = ""
    try:
        target = s.translate(from_lang="ar", to="en")
    except textblob.exceptions.NotTranslated:
        target = source
    finally:
        return target
    


def translate_py(source, _from="ar", _to="en"):
    
    target = translator(_from, _to, source)   # returns server error..
    return target
    
if __name__ == '__main__':
    
    print()