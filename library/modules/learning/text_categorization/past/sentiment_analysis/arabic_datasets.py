'''
Created on Mar 22, 2017

@author: dicle
'''


import os
import pandas as pd
import csv

def read_tweets(csvpath, sep="\t", 
                textcol="text", catcol="label"):

    df = pd.read_csv(csvpath, sep=sep, quoting=csv.QUOTE_NONE)
    df = df.sample(frac=1).reset_index(drop=True)
    texts = df[textcol].tolist()
    labels = df[catcol].tolist()
    return texts, labels, df


if __name__ == '__main__':
    
    csvpath = '/home/dicle/Documents/arabic_nlp/datasets/twitter_sentiment/ASTD-master/10K_polar-tweets.csv'
    
    t, l = read_tweets(csvpath)
    print(len(t))
    print(t[:5])



    