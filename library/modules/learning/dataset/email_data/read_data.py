'''
Created on Nov 7, 2016

@author: dicle
'''

import os
import pandas as pd

from modules.learning.dataset import io_utils

def get_categories_from_users(dfemail, dfcalisan):
    
    for i in range(dfemail.shape[0]):
        sid = dfemail.loc[i, "Sahibi"]
        cat = "UNKNOWN"
        try:
            cat = dfcalisan.loc[sid, "Bölüm"]
        except:
            pass
        dfemail.loc[i, "category"] = cat

    return dfemail


'''
def get_messages(dfpath, sep="\t", text_colname="Açıklama", cat_colname="cats2"):
    
    #cols = ["Açıklama", "category"]
    df = pd.read_csv(dfpath, sep=sep)
    df[text_colname] = df[text_colname].astype("str")
    instances = df[text_colname].tolist()
    instances = [i.strip() for i in instances]
    cats = df[cat_colname].tolist()
    return instances, cats
'''   
    
def get_messages(dfpath, include_subject=False, sep="\t", 
                 text_colname="Açıklama", subject_colname="Konu", cat_colname="cats2"):
    
       
    #cols = ["Açıklama", "category"]
    df = pd.read_csv(dfpath, sep=sep)
    
    # @TODO remove empty instances
    
    df[text_colname] = df[text_colname].astype("str")
    df[subject_colname] = df[subject_colname].astype("str")
    subjects = [i.strip() for i in df[subject_colname].tolist()]
    messages = [i.strip() for i in df[text_colname].tolist()]
    instances = [i+j for i,j in zip(subjects, messages)]
    cats = df[cat_colname].tolist()
    return instances, cats




# strip cells from spaces etc.
def format_cells(inpath, outpath):   
    df = io_utils.readcsv(inpath)
    cols = df.columns.values.tolist()
    
    for col in cols:
        df[col] = [i.strip() for i in df[col].tolist()]
        # @TODO remove html etc.
    
    io_utils.tocsv(df, outpath)
    

if __name__ == '__main__':
    
    infolder = "/home/dicle/Documents/experiments/ttnet_email2/filtered4/temp1"
    outfolder = "/home/dicle/Documents/experiments/ttnet_email2/filtered4/temp2"
    
    fnames = io_utils.getfilenames_of_dir(infolder, removeextension=False)
    for fname in fnames:
        print()
    
    