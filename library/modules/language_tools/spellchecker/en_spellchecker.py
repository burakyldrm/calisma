'''
Created on Jan 3, 2017

@author: dicle
'''

import enchant


def spellcheck(word):
    
    checker = enchant.Dict("en_US")
    
    if checker.check(word):  # word has correct spelling
        return word
    else:
        return checker.suggest(word)[0]
    
    
    

if __name__ == '__main__':
    pass
