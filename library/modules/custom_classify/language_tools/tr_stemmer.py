# encoding: utf-8

'''
Created on Dec 21, 2016

@author: dicle
'''

import subprocess
import re

import snowballstemmer

from modules.custom_classify.language_tools import TOOL_CONSTANTS

import os
from time import time

def stem(word):                                                            
    #return  subprocess.Popen("echo '" + word + "' | flookup Documents/tools/tr_morph/coltekin/TRmorph/stem.fst", shell=True, stdout=subprocess.PIPE).stdout.read().split()
    
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
    #print(items)
    root = items[-1]
    
    tag_pattern = r"\<(\w+:?\w*)+\>"   #"\<\w{1,4}\>"
    root = re.sub(tag_pattern, "", root)

    
    if root.endswith("?"):   # no root could be found, especially for NEs.
        return word
    else:
        return root

def stem2(word):
    
    
    stemmer = snowballstemmer.stemmer("turkish")
    return stemmer.stemWord(word)

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

if __name__ == '__main__':
    print("helloo")
