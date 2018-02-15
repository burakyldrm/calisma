'''
Created on Apr 26, 2017

@author: dicle
'''

import re

import numpy as np
import pandas as pd

# Removes hashtags, mentions, links
# from https://github.com/mertkahyaoglu/twitter-sentiment-analysis/blob/master/utils.py
def __cleanTweets(tweets):
    clean_data = []
    for tweet in tweets:
        item = ' '.join(word.lower() for word in tweet.split() \
            if not word.startswith('#') and \
               not word.startswith('@') and \
               not word.startswith('http') and \
               not word.startswith('RT'))
        if item == "" or item == "RT":
            continue
        clean_data.append(item)
    return clean_data



# Replaces links and mentions (usernames) with constants; removes rt's.
# Adapted from https://github.com/mertkahyaoglu/twitter-sentiment-analysis/blob/master/utils.py
def _clean_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        
        tweet = re.compile("RT \@.+:").sub("", tweet).strip()
        
        words = tweet.split()
        
        newwords = []
        for word in words:
            if word.startswith("@"):
                word = "@<USER>"
            elif word.startswith("http"):
                word = "<URL>"
            newwords.append(word)

        tweet = " ".join(newwords)
        tweet = tweet.strip()
        clean_tweets.append(tweet)
        
        
        # todo: 1) via @user  2) @user aracılığıyla       
        
    return clean_tweets




def preprocess_twitter_dataset(_tweets, _labels):

    tweets1 = _clean_tweets(_tweets) # 1- clean twitter symbols
    
    labels = _labels
    if labels is None:
        labels = [None]*len(tweets1)
    
    df = pd.DataFrame(data=np.array([tweets1, labels]).T, columns=["text", "label"])
    df = df.drop_duplicates()   # 2- remove duplicates
    not_empty = lambda x : True if len(x.strip()) > 0 else False
    df = df[df["text"].apply(not_empty)]   # 3- clean empty instances
    
    # todo: replace coooool+ with coool+
    
    tweets = df["text"].tolist()
    labels = df["label"].tolist()
    
    if _labels is None:
        labels = _labels
    
    return tweets, labels

if __name__ == '__main__':
    
    '''
    infile = "/home/dicle/Documents/data/tr_twitter_raw25Apr/2100trtweets.csv"
    outfile = "/home/dicle/Documents/data/tr_twitter_raw25Apr/preprocessing/2100_prep.csv"
    prep1(infile, outfile)
    '''
    
    tweets = ["RT @xx:",
              "http://..",
              "@x qqq"]
    
    cleaned = _clean_tweets(tweets)
    print(cleaned)
    print(len(cleaned))
    
    
    
    
    
    
    
    