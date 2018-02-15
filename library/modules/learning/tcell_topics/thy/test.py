'''
Created on Apr 7, 2017

@author: dicle
'''


import os, json


if __name__ == '__main__':
    
    
    outfolder = "/home/dicle/Documents/experiments/thy_topics/jsons"
    fname = "milesNsmiles.json"
    
    p = os.path.join(outfolder, fname)
    f = open(p, "r")
    x = json.load(f)
    print(type(x) is list)
    
    
    