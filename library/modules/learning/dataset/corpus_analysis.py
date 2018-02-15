'''
Created on Oct 19, 2016

@author: dicle
'''
'''
returns # of tokens, words, sentences, entities in the input text.
'''

import sys
sys.path.append("..")



from collections import Counter
import os
import re

from polyglot.text import Text

from modules.learning.dataset import corpus_io, io_utils
from modules.learning.keyword_extraction import topic_extraction_decompose
from modules.learning.language_tools import polyglot_NER, spellchecker, TokenHandler
from modules.learning.language_tools.spellchecker import TrSpellChecker
from modules.learning.misc import list_utils
import pandas as pd


def get_unit_counts(text, lang):
    t = Text(text, hint_language_code=lang)
    # text = Text(blob, hint_language_code='en')
    # t.language = lang
    
    words = text.split()
    nwords = len(words)
    
    tokens = t.tokens
    ntokens = len(tokens)
    
    sentences = t.sentences
    nsentences = len(sentences)
    
    nchars = len(text)
    
    named_entities = t.entities
    nentities = len(named_entities)
    
    return (nwords, ntokens, nsentences, nchars, nentities)





 
def category_stats(instances, labels, outputfolderpath, lang, fname):
 
    # add avg sentence length
 
    print(fname)
    
    
    header = ["text", "category"]
    matrix = [[text, label] for text, label in zip(instances, labels)]
    df = pd.DataFrame(data=matrix, columns=header)
    df["text"] = [text.strip() for text in df.loc[:, "text"]]  # remove boundary spaces in text
     
    # calculate text stats for each instance
    dfstats = df.copy()
    statheaders = ["nwords", "ntokens", "nsentences", "nchars", "nentities"]
    instance_stats = []
    # dfstats[statheaders] = np.zeros((dfstats.shape[0], len(statheaders)), dtype=int)
    for i in df.index.values.tolist():
        
        instance_text = df.loc[i, "text"]
        print("--Processing text ", str(i), instance_text[:15])
        nwords, ntokens, nsentences, nchars, nentities = get_unit_counts(instance_text, lang)
        instance_stats.append([nwords, ntokens, nsentences, nchars, nentities])
        print("--\n")
    # dfstats.loc[:, statheaders] = instance_stats
    dfstats = pd.concat([dfstats, pd.DataFrame(data=instance_stats, columns=statheaders)], axis=1)
    path = os.path.join(outputfolderpath, "instance_stats.csv")
    dfstats.to_csv(path, sep="\t", index=False)   
    
    catgroup = dfstats.groupby("category")
    cat_stats = catgroup.sum()[statheaders]
    cat_stats["ntexts"] = catgroup.count()["text"] 
    cat_stats = list_utils.get_col_avg(cat_stats, denominator_cols=statheaders, nominator_col="ntexts")  # take the avg of the values of the cols in statsheader
     
    path = os.path.join(outputfolderpath, "cat_stats.csv")
    cat_stats.to_csv(path, sep="\t")
     
    '''
    nwords, ntokens, nsentences, nchars, nentities
     
     
     
    catsmembers = catgroup.groups
     
    for cat, members in catsmembers.items():
        # compute text stats for the current cat
        for memberid in members:
            membertext = df.loc[memberid, "text"]
            nwords, ntokens, nsentences, nchars, nentities = text_analysis.get_unit_counts(membertext)
     
     
    '''


# df = [(text, category, id)] in type pandas.DataFrame
def instance_entities(df, lang, recordpath=None):

    new_header = ["id", "category", "entities"]
    matrix = []
    for i in df.index.values.tolist():
        instance_text = df.loc[i, "text"]
        newid = "cat" + str(df.loc[i, "category"]) + "_" + "id" + str(df.loc[i, "id"])
        print(newid)
        
        iid = df.loc[i, "id"]
        cat = df.loc[i, "category"]
        entities = polyglot_NER.get_named_entities(instance_text, lang)  # entities = [(entity_literal, tag)]
        formatted_entities = "; ".join(["(" + entity + "_" + tag + ")" for entity, tag in entities])
        
        matrix.append([iid, cat, formatted_entities])
    
    ent_df = pd.DataFrame(data=matrix, columns=new_header)
    if recordpath:
        ent_df.to_csv(recordpath, sep="\t", index=False)
    return ent_df



def category_entities(data_df, lang, recordpath=None):
    new_header = ["category", "entities"]
    matrix = []
    
    instance_entity_df = instance_entities(data_df, lang)
    labelnames = list(set(instance_entity_df.loc[:, "category"].values.tolist()))
      
    cat_group = instance_entity_df.groupby("category")
    
    # merge the entities in each category
    for label in labelnames:
        edf = cat_group.get_group(label)
        entities = edf["entities"].tolist()
        entities = [i for i in entities if len(i) > 0]
        entities_str = "; ".join(entities)
        matrix.append([label, entities_str])
    
    cat_entities_df = pd.DataFrame(data=matrix, columns=new_header)
    
    if recordpath:
        cat_entities_df.to_csv(recordpath, sep="\t", index=False)
    
    return cat_entities_df

def rawdata2df(instances, labels, dfpath=None):
    
    header = ["text", "category"]
    matrix = [[text, label] for text, label in zip(instances, labels)]
    df = pd.DataFrame(data=matrix, columns=header)
    df["text"] = [text.strip() for text in df.loc[:, "text"]]  # remove boundary spaces in text
    df["id"] = df.index.values.tolist()
    if dfpath:
        df.to_csv(dfpath, sep="\t", index=False)
    return df



# extract the topics of each category
def category_topics(instances, labels, lang):
    
    labelnames = list(set(labels))
    header = ["text", "category"]
    matrix = [[text, label] for text, label in zip(instances, labels)]
    df = pd.DataFrame(data=matrix, columns=header)
    df["text"] = [text.strip() for text in df.loc[:, "text"]]  # remove boundary spaces in text

    cat_group = df.groupby("category")
    
    # concatenate the member texts for each category
    for label in labelnames:
        texts = cat_group.get_group(label)
        texts = texts["text"].tolist()
        # concat_text = " ".join(texts["text"].tolist())
        
        # find the topics of the list of texts in the current category
        print("\n\n--Topics & keywords for cat ", label)
        topic_extraction_decompose.extract_topics(texts, lang,
                                                 n_features=200, n_topics=2, n_top_words=10)
        
        print("--")
        
        # record or print in console here
        '''
        for topicitem in topic_keywords.items():
            topicid, keywords = topicitem
            print(" ",topicid," : ", keywords)
        
        filename = "cat_" + label + "-topics" + ".txt"
        path = os.path.join(io_utils.ensure_dir(os.path.join(outputfolderpath, "topics")))
        '''



# extract numbers of length longer than 6 digits, which might stand for some id.
def _extract_numbers(text):
    
    number_pattern = "[a-zA-z]{,3}\d{6,}"
    numbers = re.findall(number_pattern, text)
    return numbers


# missing: month names
def _extract_dates(text):
    
    date_pattern = "\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{4}"
    dates = re.findall(date_pattern, text)
    return dates


# number types? if hizmet or tc kimlik around..
def instance_numbers_dates(df, recordpath):

    df_numbers = df.copy()
    df_numbers["numbers"] = ["-"] * len(df)
    df_numbers["dates"] = ["-"] * len(df)

    for i in df.index.values.tolist():
        text = df.loc[i, "text"]
        numbers = _extract_numbers(text)
        dates = _extract_dates(text)
        
        df_numbers.loc[i, "numbers"] = "; ".join(numbers)
        df_numbers.loc[i, "dates"] = "; ".join(dates)
    
    
    if recordpath:
        df_numbers.to_csv(recordpath, sep="\t", index=False)
        
    return df_numbers


# df = (text, category)
def spelling_correction(df, lang="tr", recordpath=""):
    
    # corrected_df = df.copy()
    spellchecker = TrSpellChecker.TrSpellChecker()
    for i in df.index.values.tolist():
        print(i, "  ", df.loc[i, "id"])
        text = df.loc[i, "text"]
        corrected_text = spellchecker.correct_text(text)
        # corrected_df.loc[i, "text"] = corrected_text
        row = "\t".join([corrected_text, df.loc[i, "category"], df.loc[i, "id"]])
        # io_utils.
        


'''
def spelling_correction(df, lang="tr", recordpath=None):
    
    corrected_df = df.copy()
    
    for i in df.index.values.tolist():
        print(i,"  ", df.loc[i, "id"])
        text = df.loc[i, "text"]
        corrected_text = spellcheck.correct_text(text, lang)
        corrected_df.loc[i, "text"] = corrected_text
    
    
    if recordpath:
        corrected_df.to_csv(recordpath, sep="\t", index=False)    
    
    return corrected_df
'''

# fix this. 
# some take df, some take instances, labels.
def analyse_corpus(dfpath, lang, outputpath):
    '''
    instances, labels = corpus_io.get_emails(dfpath)
    category_stats(instances, labels, outputpath, lang)
    category_topics(instances, labels, lang)
    '''
    df = pd.read_csv(dfpath, sep="\t")
    # instance_ner_path = os.path.join(outputpath, "instance_entities"+".csv")
    # i_ent_df = instance_entities(df, lang, instance_ner_path)
    category_ner_path = os.path.join(outputpath, "category_entities" + ".csv")
    cat_ent_df = category_entities(df, lang, category_ner_path)
    


def category_disintersection_words(df, textcol, catcol):


    preprocessor = TokenHandler.TrTokenHandler(stopword=True, stemming=False,
                                                            remove_numbers=True,
                                                            deasciify=True,
                                                            remove_punkt=True)


    cats = list(set(df[catcol].tolist()))
    cat_group = df.groupby(catcol)

    words_dict = {}
    catdf_dict = {}
    for label in cats:
        words_dict[label] = []
        catdf_dict[label] = None

    for label in cats:
        catdf = cat_group.get_group(label)
             
        texts = catdf[textcol].tolist()
        texts = [text.strip() for text in texts]
        texts = list(set(texts))
        
        words = preprocessor(" ".join(texts))
        
        words_dict[label] = words
        catdf_dict[label] = catdf


    wl = [set(i) for i in list(words_dict.values())]
    intersection = list(set.intersection(*wl))
    
    disintersects = {}
    for cat, words in words_dict.items():
        ws = [i for i in words if i not in intersection]
        ws = list(set(ws))
        disintersects[cat] = ws
        
    # most common disintersecting, intersecting
    # disintersecting
    dis_countwords = {}
    int_countwords = {}
    for cat, diswords in disintersects.items():
      
        allwords = words_dict[cat]
        counter = Counter(allwords)
        selfwordcount = [(word, counter[word]) for word in diswords]
        selfwordcount.sort(key=lambda x : x[1], reverse=True)
        dis_countwords[cat] = selfwordcount
        
        commonwordcount = [(word, counter[word]) for word in intersection]
        commonwordcount.sort(key=lambda x : x[1], reverse=True)
        int_countwords[cat] = commonwordcount
        
    
    
    # 1) intersecting words  2) disintersecting words for each cat 
    #  3) count of disintersecting words for each cat  4) count of intersecting words for each cat
    return intersection, disintersects, dis_countwords, int_countwords, cats
    



def analyse_datasets(infolder,
                     fname,
                     text_col,
                     cat_col,
                     sep,
                     lang,
                     outfolder):
    
  
    
        
    df = pd.read_csv(os.path.join(infolder, fname), sep=sep)
    
    
    
    # word dis/intersections
    
    intersection, disintersects, dis_countwords, int_countwords, cats = category_disintersection_words(df, textcol=text_col, catcol=cat_col)
    # list, dict, dict, dict
    open(os.path.join(outfolder, "intersection.csv"), "w").write("\n".join(intersection))
    for cat in cats:
        fpath = os.path.join(outfolder, cat + "_disintersecting_list.csv")
        io_utils.todisc_list(fpath, disintersects[cat])
        
        fpath = os.path.join(outfolder, cat + "_disintersecting_count.csv")
        content = "\n".join([str(i) + "\t" + str(j) for (i, j) in dis_countwords[cat]])
        open(fpath, "w").write(content)
        
        fpath = os.path.join(outfolder, cat + "_intersecting_count.csv")
        content = "\n".join([str(i) + "\t" + str(j) for (i, j) in int_countwords[cat]])
        open(fpath, "w").write(content)

    
    
    # category stats
    instances = df[text_col].tolist()
    labels = df[cat_col].tolist()
    category_stats(instances, labels, outfolder, lang, fname)




if __name__ == "__main__":
    '''
    instances, labels = corpus_io.get_csv_data(path="/home/dicle/Documents/data/ttnet_email/PROBSUMMARYM1.csv", 
                                                  delimiter=",")
    '''
    '''
    lang = "tr"
    
    #category_stats(instances, labels, outputfolderpath="/home/dicle/Documents/experiments/ttnet_email", lang=lang)
    #category_topics(instances, labels, lang)
    
    outputpath = "/home/dicle/Documents/experiments/ttnet_email"
    dfname = "data_df"
    dfpath = os.path.join(outputpath, dfname+".csv")
    
    #rawdata2df(instances, labels, dfpath)
    df = io_utils.readcsv(dfpath)
    #instance_ner_path = os.path.join(outputpath, "instance_entities"+".csv")
    #i_ent_df = instance_entities(df, lang, instance_ner_path)
    #category_ner_path = os.path.join(outputpath, "category_entities"+".csv")
    #cat_ent_df = category_entities(df, lang, category_ner_path)
    
    corrected_texts_df_path = os.path.join(outputpath, "correct_texts.csv")
    spelling_correction(df, lang, corrected_texts_df_path)
    '''
    
    '''
    print("Analyse 6-cat email data")
    lang = "tr"
    analyse_corpus(dfpath="/home/dicle/Documents/experiments/ttnet_email/data_df-6cats.csv", 
                   lang=lang, 
                   outputpath="/home/dicle/Documents/experiments/ttnet_email/clsf/normalization0_cat6")
    '''
    
    '''
    print("Extract dates & numbers in the emails")
    dfpath = "/home/dicle/Documents/experiments/ttnet_email/original_texts.csv"
    df = io_utils.readcsv(dfpath)
    recordpath = "/home/dicle/Documents/experiments/ttnet_email/data_df-numbers.csv"
    instance_numbers_dates(df, recordpath)
    '''
    
    
    '''
    # intersections
    # 13K email
    
    folderpath = "/home/dicle/Documents/data/emailset2"
    fname = "has_ariza.csv"  # "Raw_Email_Data-OriginalSender.csv"
    cat_col = "TIP"
    lang = "tr"
    
        
    df = pd.read_csv(os.path.join(folderpath, fname), sep=";")
    
    
    outfolder = "/home/dicle/Documents/data/emailset2/terms_features/"
    
    intersection, disintersects, dis_countwords, int_countwords, cats = category_disintersection_words(df, textcol="MAIL", catcol="TIP")
    # list, dict, dict, dict
    open(os.path.join(outfolder, "intersection.csv"), "w").write("\n".join(intersection))
    for cat in cats:
        fpath = os.path.join(outfolder, cat + "_disintersecting_list.csv")
        io_utils.todisc_list(fpath, disintersects[cat])
        
        fpath = os.path.join(outfolder, cat + "_disintersecting_count.csv")
        content = "\n".join([str(i) + "\t" + str(j) for (i, j) in dis_countwords[cat]])
        open(fpath, "w").write(content)
        
        fpath = os.path.join(outfolder, cat + "_intersecting_count.csv")
        content = "\n".join([str(i) + "\t" + str(j) for (i, j) in int_countwords[cat]])
        open(fpath, "w").write(content)
    

    '''
    
    # en sentiment
    infolder = "/home/dicle/Documents/data/en_sentiment"
    fname = "en_polar_10Kreviews.csv"
    text_col = "text"
    cat_col = "category"
    sep = "\t"
    outrootfolder = "/home/dicle/Documents/data/en_sentiment/dataset_analysis"
    outfolder = io_utils.ensure_dir(os.path.join(outrootfolder, fname))
    lang = "en"
    analyse_datasets(infolder, fname, text_col, cat_col, sep, lang, outfolder)
    
    '''
    df = pd.read_csv(os.path.join(folderpath, fname), sep=";")
    disintersects, intersection = category_disintersection_words(df, textcol="MAIL", catcol="TIP")
    outfolder = "/home/dicle/Documents/data/emailset2/terms_features"
    open(os.path.join(outfolder, "intersection.csv"), "w").write("\n".join(intersection))
    for i,words in enumerate(disintersects):
        open(os.path.join(outfolder, "disintersects_cat"+str(i)+".csv"), "w").write("\n".join(words))
    '''
