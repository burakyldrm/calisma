'''
Created on Mar 1, 2017

@author: dicle
'''

'''
utilities for pandas dataframes

'''


import os

import pandas as pd


def merge_shuffle_dfs(dfpaths, sep):
    
    dfs = []
    
    for path in dfpaths:
        df = pd.read_csv(path, sep=sep)
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    return df_all



def merge_shuffle_dfs2(dfpaths, sep, keep_cols):
    
    dfs = []
    
    for path in dfpaths:
        df = pd.read_csv(path, sep=sep)
        if keep_cols:
            df = df.loc[:, keep_cols]
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    return df_all





def texts_to_csv(lines, cat):    
    '''
    collect texts stored as lines in txt file to rows in a csv file and add the given category beside the row.
    '''   
    rows = []
    for line in lines:
        
        rows.append([line, cat])

    return rows


# fileset = [(path_to_txt, category, path_to_output_df)]
def make_dataset(fileset, outputpath, text_col="text", cat_col="category", sep="\t"):
    
    dfs = []
    for txtpath, cat, dfpath in fileset:
       
        lines = open(txtpath, "r").readlines()
        rows = [{text_col : line, cat_col : cat} for line in lines]
        df = pd.DataFrame(rows)
        
        if dfpath:
            df.to_csv(dfpath, index=False, sep=sep)
        
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all = df_all.sample(frac=1).reset_index(drop=True)
    df_all.to_csv(outputpath, sep=sep, index=False)
    return df_all


# select N texts from the most occurring category to keep category balance
# N is the number of instances the other less populous categories have
def category_balance(df, cat_col, popular_cat):
    
    # @TODO do not pass popular_cat. find it here.
    
    # split the df into rows of only popular_cat and the rest
    df_cat = df.loc[(df[cat_col] == popular_cat), :]
    df_noncat = df.loc[~(df[cat_col] == popular_cat), :]
    
    # select random instances randomly from df_cat as many as those in non_cat
    N, _ = df_noncat.shape
    df_cat2 = df_cat.sample(n=N)
    
    # append the non_cat and the random cat instances
    df_final = pd.concat([df_cat2, df_noncat])
    # shuffle and re_index
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    return df_final


def category_balance3(df, text_col, cat_col):

    g = df.groupby([cat_col])
    counts = g.count()[text_col]
    cats = counts.index.values.tolist()
    
    #weak_cat = counts.argmin()
    N = counts.min()
   
    dfs = []
    # select N random element from each cat and collect them in dfs
    for cat in cats:
        dfc = g.get_group(cat)
        dfc = dfc.sample(n=N)
        dfs.append(dfc)
    
    df_final = pd.concat(dfs)
    
    # shuffle and re_index
    df_final = df_final.sample(frac=1).reset_index(drop=True)
        
    return df_final


# select N texts from the most occurring category to keep category balance
# N is the number of instances the other less populous categories have
def category_balance2(df, text_col, cat_col):

    counts = df.groupby([cat_col]).count()[text_col]
    popular_cat = counts.argmax()
    N = counts.max()
    if N > (counts.sum() / 2):    # in case there are more than 2 distinct categories
        N = counts.sum() - counts.max()
    
    # split the df into rows of only popular_cat and the rest
    df_cat = df.loc[(df[cat_col] == popular_cat), :]
    df_noncat = df.loc[~(df[cat_col] == popular_cat), :]
    
    # select random instances randomly from df_cat as many as those in non_cat
       
    #N, _ = df_noncat.shape
    df_cat2 = df_cat.sample(n=N)
    
    # append the non_cat and the random cat instances
    df_final = pd.concat([df_cat2, df_noncat])
    # shuffle and re_index
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    return df_final

def replace_column_value(df, cat_col, old_val, new_val):
    
    df1 = df.copy()
    
    oldcol = df1[cat_col].tolist()
    newcol = []
    
    for val in oldcol:
        if val == old_val:
            newcol.append(new_val)
        else:
            newcol.append(val)
        
    df1[cat_col] = newcol
    return df1

def main_make_datasets():
    
    infolder = "/home/dicle/Documents/data/en_sentiment/PangLee/rt-polaritydataV1-5331snippets/rt-polaritydata"
    outfolder = "/home/dicle/Documents/data/en_sentiment/PangLee/rt-polaritydataV1-5331snippets/rt-polaritydata/tables"
    fileset = [("rt-polarity.pos", "pos", "pos_reviews.csv"),
               ("rt-polarity.neg", "neg", "neg_reviews.csv")]
    
    fileset = [(os.path.join(infolder, i), j, os.path.join(outfolder, k)) for i,j,k in fileset]
    
    tablename = "polar_reviews.csv"
    outtablepath = os.path.join(outfolder, tablename)

    text_col = "text"
    cat_col = "category"
    sep = "\t"
    make_dataset(fileset, outtablepath, text_col, cat_col, sep)


def main_merge():
    
    folder = "/home/dicle/Documents/experiments/fb_comments"
    files1 = ["annot-dicle_isbankasi.csv", "annot-rusen_turkcell.csv", "annot-ismail_akbank.csv"]
    paths = [os.path.join(folder, "balanced_"+fname) for fname in files1]
    sep = "\t"
    keep_cols = ["message", "LABEL"]
    
    df_all = merge_shuffle_dfs2(paths, sep, keep_cols)
    df_all.to_csv("/home/dicle/Documents/experiments/fb_comments/merged/balanced_all3.csv", index=False, sep="\t")
    
    '''
    folder = "/home/dicle/Documents/data/en_sentiment/twitter2"
    fnames = ['__en_neg.csv', '__en_pos.csv']
    paths = [os.path.join(folder, fname) for fname in fnames]
    sep = "\t"
    
    df_all = merge_shuffle_dfs(paths, sep)
    df_all.to_csv(os.path.join(folder, "all.csv"), index=False, sep="\t")

    '''



def inspect_txt_lines():
    
    path = "/home/dicle/Documents/data/en_sentiment/PangLee/rt-polaritydataV1-5331snippets/rt-polaritydata/rt-polarity.pos"
    with open(path, "r") as f:
        i = 0
        line = f.readline()
        while line:
            line = f.readline()
            print(i, line)
            i += 1

if __name__ == '__main__':

    main_make_datasets()
    #inspect_txt_lines()
    
    