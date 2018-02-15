'''
Created on Oct 12, 2016

@author: dicle
'''

import sys
sys.path.append("..")


import os, codecs
import random

from sklearn.datasets import fetch_20newsgroups

import modules.learning.dataset.twitter.twitter_preprocessing as tweet_prep
from modules.learning.dataset import LEX_CONSTANTS
from modules.learning.dataset import io_utils
import numpy as np
import pandas as pd
import string




def get_random_data(nrows=200, ncols=75):
    X = np.random.rand(nrows, ncols)
    y = np.random.randint(1, 5, size=(nrows, 1))
    return X, y


def read_20newsgroup():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    dataset = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
    instances = dataset.data
    labels = dataset.target
    return instances, labels




def read_and_merge_n_files(folderpath, N):
    
    filenames = io_utils.getfilenames_of_dir(folderpath, False)[:N]
    texts = []
    for fname in filenames:
        fpath = os.path.join(folderpath, fname)
        content = io_utils.readtxtfile2(fpath)
        texts.append(content)
        
    return texts


def read_delimited_lines(path, delimiter):
    
    categories = []
    textinstances = []
    
    with open(path) as f:
        for line in f:
            data = line.split(delimiter)
            category = str(data[0].replace(' ', ''))
            categories.append(category)
            textinstances.append(data[1])
    
    return textinstances, categories




def read_twitter_data(folder="/home/dicle/Documents/data/tr_sentiment/movie_product_joint",
                            fname="sentiment_reviews.csv",
                            sep="\t",
                            text_col="text",
                            cat_col="polarity",
                            shuffle=False):
    _tweets, _labels = read_dataset_csv(folder, fname, sep, text_col, cat_col)
    tweets, labels = tweet_prep.preprocess_twitter_dataset(_tweets, _labels)
    
    if shuffle:
        tweets, labels = shuffle_dataset(tweets, labels)

    return tweets, labels




    
# returns texts, labels
def read_dataset_csv(folder, fname, textcol="text", catcol="category", sep="\t", shuffle=False):
    
    path = os.path.join(folder, fname)
    
    # columns = ["text", "category"]  # do not include id or any other columns
    df = pd.read_csv(path, sep=sep)
    
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # return as instances and labels
    instances = df[textcol].values.tolist()
    instances = [str(i) for i in instances]
    labels = df[catcol].values.tolist()
    instances = [i.strip() for i in instances]
    return instances, labels 

# returns texts, labels
def read_dataset_csv2(path, sep="\t", textcol="text", catcol="category", shuffle=False):
    
    # columns = ["text", "category"]  # do not include id or any other columns
    df = pd.read_csv(path, sep=sep)
    
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # return as instances and labels
    instances = df[textcol].values.tolist()
    instances = [str(i) for i in instances]
    labels = df[catcol].values.tolist()
    instances = [i.strip() for i in instances]
    return instances, labels 


# returns X, y
def read_data_and_category_csv(path, catcol, sep="\t"):
    
    df = pd.read_csv(path, sep=sep)
    
    x_cols = [i for i in df.columns.values.tolist() if i != catcol]
    X = df.loc[:, x_cols]
    y = df.loc[:, catcol]
    return X, y



def read_dataset_file(file, sep="\t"):
    labels=[]
    instances=[]
    #printable=set(string.printable)
    for line in file:
        if not isinstance(line, str):
            line = line.decode()
        if line is not None and line!='':
            try:
                line_cl=str(line).rstrip("\,\!\;\r\n")
                #filter(lambda x:x in printable,line_cl)
                #data = line.decode().split(delimiterParam)
                data = line_cl.rsplit(sep,1)
                labels.append(data[1])
                instances.append(data[0])
            except Exception:
                pass
    
    return instances, labels
 

#####################
### LEXICONS  ###

def get_boun_polarity_lexicon(label, remove_tags=True):
    ext = ".txt"
    type_sep = "#"  # the sep char in the lexicon that separates word and its pos tag in each line
    
    fname = ""
    if label in ["positive", "pos"]:
        fname = "positive"
    elif label in ["negative", "neg"]:
        fname = "negative"
        
    folderpath = LEX_CONSTANTS.BOUN_POLARITY_LEXICON_FOLDER
    path = os.path.join(folderpath, fname + ext)
    
    with open(path, "r") as f:
        words = f.readlines()
    words = [w.strip() for w in words]
    if remove_tags:
        words = [w.split(type_sep)[0] for w in words]
    return words


def get_emoticon_lexicon(label):
    
    ext = ".txt"
    fname = ""
    
    if label in ["positive", "pos"]:
        fname = "positive"
    elif label in ["negative", "neg"]:
        fname = "negative"
        
    folderpath = LEX_CONSTANTS.POLAR_EMOTICON_FOLDER
    path = os.path.join(folderpath, fname + ext)
    
    f = codecs.open(path, encoding='utf8')
    words = f.readlines()
    words = [w.strip() for w in words]
    words = [w for w in words if len(w) > 0 or not w.isspace()]
    return words

def get_english_polarity_lexicon(label):
    
    ext = ".txt"
    fname = ""
    
    if label in ["positive", "pos"]:
        fname = "positive-words"
    elif label in ["negative", "neg"]:
        fname = "negative-words"
        
    folderpath = LEX_CONSTANTS.EN_POLARITY_LEXICON_FOLDER
    path = os.path.join(folderpath, fname + ext)
    
    f = codecs.open(path, encoding='utf8')
    words = f.readlines()
    words = [w.strip() for w in words]
    words = [w for w in words if len(w) > 0 or not w.isspace()]
    return words


def _read_polar_lexicon(path):
    f = codecs.open(path, encoding='utf8')
    words = f.readlines()
    words = [w.strip() for w in words]
    words = [w for w in words if len(w) > 0 or not w.isspace()]
    return words


def _get_polarity_lexicon(folderpath):
    
    pospath = os.path.join(folderpath, "pos.txt")
    negpath = os.path.join(folderpath, "neg.txt")
    
    pos_terms = _read_polar_lexicon(pospath)
    neg_terms = _read_polar_lexicon(negpath)

    return pos_terms, neg_terms

def get_arabic_polarity_lexicon1():
    
    folderpath = LEX_CONSTANTS.AR_POLARITY_LEXICON_FOLDER1
    
    return _get_polarity_lexicon(folderpath)


# @TODO improve readers of eng and tr


def _split_emoticon_lex():
    
    path = "/home/dicle/Documents/lexicons/EmoticonSentimentLexicon/EmoticonSentimentLexicon.txt"
    f = codecs.open(path, encoding='utf8')
    lines = f.readlines()
    
    pos = []
    neg = []
    for line in lines:
        items = line.split("\t")
        symbol = items[0]
        val = int(items[1])
        if val > 0:
            pos.append(symbol)
        elif val < 0:
            neg.append(symbol)
    
    p1 = "/home/dicle/Documents/lexicons/EmoticonSentimentLexicon/pos.txt"
    p2 = "/home/dicle/Documents/lexicons/EmoticonSentimentLexicon/neg.txt"
    io_utils.todisc_txt("\n".join(pos), p1)
    io_utils.todisc_txt("\n".join(neg), p2)
    



def select_N_instances(N, instances, labels):
    
    n_indices = random.sample(range(0, len(instances)), N)
    n_instances = [instances[i] for i in n_indices]
    n_labels = [labels[i] for i in n_indices] 
    return n_instances, n_labels


def select_random_rows(N, dfX, vy):
    
    ninstances = len(vy)
    indices = random.sample(range(0, ninstances), N)
    dfX2 = dfX.iloc[indices, :]
    vy2 = [vy[i] for i in indices]
    return dfX2, vy2


def record_labels(ytrue, ypred, path, sep="\t"):
    
    header = ["ytrue", "ypred", "error"]
    
    io_utils.todisc_txt(sep.join(header) + "\n", path, "w")
    for i, j in zip(ytrue, ypred):
        error = "-"
        if i != j:
            error = "1"
        
        io_utils.todisc_txt(sep.join([i, j, error]) + "\n", path, "a")
    

def shuffle_dataset(instances, labels):

    dataset = [(i, l) for i,l in zip(instances, labels)]
    random.shuffle(dataset)
    
    new_instances = [i for i,_ in dataset]
    new_labels = [l for _,l in dataset]
    return new_instances, new_labels

   
if __name__ == "__main__":
    
    
    print(shuffle_dataset(["a", "b", "c"], [1,2,3]))
    
    # _split_emoticon_lex()
    print()
    
    
    
