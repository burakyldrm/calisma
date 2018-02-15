'''
Created on Sep 7, 2016

@author: dicle
'''


import sys
sys.path.append("..")


import os, codecs

import pandas as pd



# returns the names of the files and dirs in the given directory *path*
def getfilenames_of_dir(path, removeextension=True):
    files = os.listdir(path)
    filenames = []
    for fileitem in files:
        if os.path.isfile(path + os.sep + fileitem):
            if removeextension:
                filename = fileitem.split(".")[0]  # remove extension if any
            else:
                filename = fileitem
            filenames.append(filename)
        
    return filenames

def getfoldernames_of_dir(path):
    files = os.listdir(path)
    foldernames = []
    for fileitem in files:
        if os.path.isdir(path + os.sep + fileitem):
            foldernames.append(fileitem)
    return foldernames


# ensures if the directory given on *f* exists. if not creates it.
def ensure_dir(f):
    # d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
    return f 



def append_csv_cell_items(items, path, sep="\t"):

    with open(path, "a") as f:
        f.write(sep.join(items) + "\n")




def todisc_list(path, lst, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
  
    for item in lst:
        f.write(item + "\n")
        
    f.close()

def todisc_df_byrow(path, df, keepIndex=False, csvsep="\t"):
    
    colids = df.columns.values.tolist()
    if keepIndex:
        rowids = df.index.values.tolist()
    nr, _ = df.shape
    
    colids2 = [item + "," for colid in colids for item in colid]
    # header = csvsep.join([str(s).encode("utf-8") for s in colids])
    header = csvsep.join(colids2)
    
    if keepIndex:
        header = "\t" + header
    todisc_txt(header, path, mode="w")
    
    for i in range(nr):
        rowitems = df.iloc[i, :].tolist()
        rowstr = csvsep.join([str(s) for s in rowitems])
        if keepIndex:
            rowstr = rowids[i] + rowstr
        todisc_txt("\n" + rowstr, path, mode="a")

def readtxtfile(path):
    f = codecs.open(path, encoding='utf8')
    rawtext = f.read()
    f.close()
    return rawtext


def readtxtfile2(path):
    rawtext = open(path, "r").read()
    return rawtext

def todisc_txt(txt, path, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
    f.write(txt)
    f.close()
    
    

def readcsv(csvpath, keepindex=False, sep="\t"):
    if keepindex:
        df = pd.read_csv(csvpath, sep=sep, header=0, index_col=0, encoding='utf-8')
    else:
        df = pd.read_csv(csvpath, sep=sep, header=0, encoding='utf-8')
 
    try:
        df = df.drop('Unnamed: 1', 1)
    except:
        pass
    return df


def tocsv(df, csvpath, keepindex=False):
    df.to_csv(csvpath, index=keepindex, header=True, sep="\t", encoding='utf-8')



def initialize_csv_file(header, path, sep="\t"):
    
    headerstr = sep.join(header)
    with open(path, "w") as f:
        f.write(headerstr + "\n")
    
    
    
    
