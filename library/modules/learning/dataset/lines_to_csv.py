'''
Created on Mar 2, 2017

@author: dicle
'''

import os
import random

import pandas as pd

import io_utils
'''
collect texts stored as lines in txt file to rows in a csv file and add the given category beside the row.
'''


def texts_to_csv(lines, cat, sep="\t"):
        
    rows = []
    for line in lines:
        
        row = line + sep + cat
        rows.append(row)

    return row


# the subfolders in infolder: categories
# the textual content of the files in the subfolders: instances
def make_dataset(infolder, outcsvpath, csvsep=";"):
    folders = io_utils.getfoldernames_of_dir(infolder)
    
    rows = []
    for folder in folders:
        inp1 = os.path.join(infolder, folder)
        
        fnames = io_utils.getfilenames_of_dir(inp1, removeextension=False)
        
        
        for fname in fnames:
            
            inp2 = os.path.join(inp1, fname)
            text = open(inp2, "r").read()
            text = text.strip()
            rows.append({"text" : text, "label" : folder})
        
    
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(outcsvpath, sep=csvsep)
    return df

if __name__ == '__main__':
    
    
    make_dataset(infolder="/home/dicle/Documents/data/tr_plain_texts/1150haber/charfixed", 
                 outcsvpath="/home/dicle/Documents/data/tr_plain_texts/1150haber/tr_1150haber_5cat_semicolon-sep.csv", 
                 csvsep=";")
    
    
    
    
    
    