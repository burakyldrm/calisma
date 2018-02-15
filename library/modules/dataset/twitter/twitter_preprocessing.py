'''
Created on Apr 26, 2017

@author: dicle
'''

import re

import pandas as pd

# Removes hashtags, mentions, links
# from https://github.com/mertkahyaoglu/twitter-sentiment-analysis/blob/master/utils.py
def cleanTweets(tweets):
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
def cleanTweets2(tweets):
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

def prep1(inpath, outpath):
    
    tdf = pd.read_csv(inpath, sep="\t")
    tweets = tdf["body"].tolist()
    clean_tweets = cleanTweets(tweets)
    tdf["clean_body"] = clean_tweets
    tdf.to_csv(outpath, sep="\t", index=False)
    return tdf



if __name__ == '__main__':
    
    '''
    infile = "/home/dicle/Documents/data/tr_twitter_raw25Apr/2100trtweets.csv"
    outfile = "/home/dicle/Documents/data/tr_twitter_raw25Apr/preprocessing/2100_prep.csv"
    prep1(infile, outfile)
    '''
    
    tweets = ["RT @xx:",
              "http://..",
              "@x qqq"]
    
    cleaned = cleanTweets2(tweets)
    print(cleaned)
    print(len(cleaned))
    
    
    
    
    
    
    
    