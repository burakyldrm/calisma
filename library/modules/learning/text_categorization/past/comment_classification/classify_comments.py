'''
Created on Feb 14, 2017

@author: dicle
'''


import os
import pandas as pd
import random

from collections import Counter

from dataset import corpus_analysis, io_utils, corpus_io
from django_docker.learning.text_categorization.email_categorization import EMAIL_CONF,\
    email_classification



def remove_irrelevant_texts(df, text_col, length=3):

    # @TODO remove repeating?? 
    #  how to detect? using also the username?

    df1 = df.copy()
    
    # convert to str
    make_str = lambda x : str(x)
    df1[text_col] = df1[text_col].apply(make_str)
    
    # remove texts of length less than 3 chars
    strip = lambda x  : x.strip()
    df1[text_col] = df1[text_col].apply(strip)
    
    df2 = df.loc[~(df1[text_col].str.len() <= length), :]
    return df2


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
    print(counts)
    print(popular_cat)
    print(df_cat.shape, df_noncat.shape)
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
    

def preprocess_comments():   
   
    
    # metadata
    datafolder = "/home/dicle/Documents/data/fb_comment"
    #files = ["annot-dicle_isbankasi", "annot-rusen_turkcell"]
    files = ["annot-ismail_akbank"]
    ext = ".xlsx"
    lang = "tr"
    text_col = "message"
    cat_col = "LABEL"
    dfs = {}   # {fname : df }
    
    
        
    newdata_folder = "/home/dicle/Documents/experiments/fb_comments"
    
    statsfolder = "/home/dicle/Documents/experiments/fb_comments/stats"
        
    for fname in files:
        
        # read
        path = os.path.join(datafolder, fname+ext)
        df = pd.read_excel(path)
        
        instances = df[text_col].tolist()
        instances = [str(s) for s in instances]
        labels = df[cat_col].tolist()

        
        print(fname)
        print(" before: ", df.shape)
        
        # preprocess
        df = remove_irrelevant_texts(df, text_col)
        newpath = os.path.join(newdata_folder, "clean_"+fname+".csv")
        df.to_csv(newpath, index=False, sep="\t")
        
        print(" after: ", df.shape)
        
        # get cats have equal number of instances
        bdf = category_balance2(df, text_col, cat_col)
        bpath = os.path.join(newdata_folder, "balanced_"+fname+".csv")
        bdf.to_csv(bpath, index=False, sep="\t")
                
        # get only binary cats
        df2 = replace_column_value(df, cat_col, old_val="spam", new_val="ilgisiz")
        binarycat_fname = "binary_"+fname
        df2.to_csv(os.path.join(newdata_folder, binarycat_fname+".csv"), index=False, sep="\t")
        
        # get binary cats have equal number of instances
        bdf = category_balance2(df2, text_col, cat_col)
        bpath = os.path.join(newdata_folder, "balanced_"+binarycat_fname+".csv")
        bdf.to_csv(bpath, index=False, sep="\t")
  
        print(" balanced: ", bdf.shape)
        print()
        '''
        # category analysis
        current_outputfolder = os.path.join(outputfolder, "cat_analysis-"+fname[:-5])
        current_outputfolder = io_utils.ensure_dir(current_outputfolder)
        corpus_analysis.category_stats(instances, labels, current_outputfolder, lang)
        '''


def classify_comment_sets(foldername, filenames, text_col, cat_col, sep):

    #header = ["fname", "acc", "fscore", "duration"]
    results = []

    config = EMAIL_CONF.comment_param_config
    classifier = email_classification.get_email_classifier(config)

    # single
    all_data = []   # will store (instance, label) pairs
    for fname in filenames:
        
        inpath = os.path.join(foldername, fname)
        instances, labels = corpus_io.read_labelled_texts_csv(inpath, sep, text_col, cat_col)
        
        print("Classification for ", fname)
        
        acc, fscore, duration = classifier.cross_validated_classify(instances, labels)
        
        cat_count = str(dict(Counter(labels)))
        '''d = {}
        d["acc"] = acc
        d["fscore"] = fscore
        d["duration"] = duration
        d["fname"] = fname'''
        results.append([fname, acc, fscore, duration, cat_count])
        
        for i, l in zip(instances, labels):
            all_data.append((i,l))
    
    
    # mix
    random.shuffle(all_data)
    all_instances = [i for i,_ in all_data]
    all_labels = [l for _,l in all_data]
    acc, fscore, duration = classifier.cross_validated_classify(all_instances, all_labels)
    cat_count = str(dict(Counter(all_labels)))
    results.append(["mix_"+"+".join(filenames), acc, fscore, duration, cat_count])
    
    return results



def classify_sets():
    
    
    # classify single and mixed
    
    
    datafolder = "/home/dicle/Documents/experiments/fb_comments"
    
    # original data (removed short/empty texts)
    files1 = ["annot-dicle_isbankasi.csv", "annot-rusen_turkcell.csv", "annot-ismail_akbank.csv"]
    files2 = ["balanced_"+i for i in files1]
    files3 = ["binary_"+i for i in files1]
    files4 = ["balanced_"+i for i in files3]
    
    #files2 = ["balanced_annot-dicle_isbankasi.csv", "balanced_annot-rusen_turkcell.csv"]
    #files3 = ["binary_annot-dicle_isbankasi.csv", "binary_annot-rusen_turkcell.csv"]
    #files4 = ["balanced_binary_annot-dicle_isbankasi.csv", "balanced_binary_annot-rusen_turkcell.csv"]
    
    sets = [files1, files2, files3, files4]
    
    text_col = "message"
    cat_col = "LABEL"
    sep = "\t"
        
    
    results_folder = "/home/dicle/Documents/experiments/fb_comments"
    results_path = os.path.join(results_folder, "SGD-classify_3_comment_sets.csv")
    header = ["fname", "acc", "fscore", "duration", "category_count"]
    results = []
    
    for files in sets:
        result_list = classify_comment_sets(datafolder, files, text_col, cat_col, sep)
        results.extend(result_list)
    
    rdf = pd.DataFrame(results, columns=header)
    rdf.to_csv(results_path, index=False, sep="\t")

    print()
    


def train_and_save_comment_classifier(train_instances, train_labels,
                                      picklefolder,
                                      modelname):
    
    config = EMAIL_CONF.comment_param_config
    classifier = email_classification.get_email_classifier(config)
    
    model, _ = classifier.train_and_save_model(train_instances, train_labels, picklefolder, modelname)
    
    test_instances = ["kredi başvurusu yapabilir miyim",
                      "paraya ihtiyacı olan bize başvurabilir.",
                      "sgsdshdhs"]
    
    ypred = classifier.predict(model, test_instances)
    
    return ypred

if __name__ == '__main__':


    data_folder = "/home/dicle/Documents/experiments/fb_comments/merged"   # @degistir
    fname = "balanced_annotated_isbank-turkcell.csv"  # "balanced_all3.csv"
    text_col = "message"
    cat_col = "LABEL"
    sep = "\t"
    datapath = os.path.join(data_folder, fname)
    instances, labels = corpus_io.read_labelled_texts_csv(datapath, sep, text_col, cat_col)
    
    
    picklefolder = "/home/dicle/Documents/experiments/fb_comments/models"  # @degistir
    modelname = "balanced_fb_1200comments"
    ypred = train_and_save_comment_classifier(instances, labels, picklefolder, modelname)
    
    print(ypred)
    
    #preprocess_comments()
    #classify_sets()
    

    '''
    # replace spam with ilgisiz to have binary cats.
    datafolder = "/home/dicle/Documents/experiments/fb_comments"
    # original data (removed short/empty texts)
    files = ["annot-ismail_akbank.csv"]
    cat_col = "LABEL"
    for file in files:
        path = os.path.join(datafolder, file)
        df = pd.read_csv(path, sep="\t")
        df2 = replace_column_value(df, cat_col, old_val="spam", new_val="ilgisiz")
        df2.to_csv(os.path.join(datafolder, "binary_"+file), index=False, sep="\t")
    '''


    
    '''
    # category balance for binary cat setting
    datafolder = "/home/dicle/Documents/experiments/fb_comments"

    files = ["binary_annot-ismail_akbank.csv"]
    cat_col = "LABEL"
    for file in files:
        path = os.path.join(datafolder, file)
        df = pd.read_csv(path, sep="\t")
    
        bdf = category_balance(df, cat_col, popular_cat="ilgili")
        bpath = os.path.join(datafolder, "balanced_"+file+".csv")
        bdf.to_csv(bpath, index=False, sep="\t")
    '''
    

        
    
    
    
    
    