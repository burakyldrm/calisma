'''
Created on Feb 6, 2017

@author: dicle
'''


import sys
import csv
sys.path.append("..")


import os, random
import pandas as pd


from modules.learning.dataset import io_utils


'''
 collect the labelled texts in one csv file.

'''



####################### spam emails  #############

# mainfolder has two subfolders: normal, spam
# each folder has txt files containing emails.
#  in the email files: line_i: Kimden:.. line_j: Kime:.. line_k: Konu:.. and line_k+1: is the body
#  we record these items, as well as the category label from the folder name, in the csv as columns, per row 
def spam_mails_to_csv(mainfolder, outpath):
    
    csv_rows = []
    cats = io_utils.getfoldernames_of_dir(mainfolder)

    for cat in cats:
        
        p1 = os.path.join(mainfolder, cat)
        fnames = io_utils.getfilenames_of_dir(p1, removeextension=False)
        
        for fname in fnames:
            
            p2 = os.path.join(p1, fname)
            lines = open(p2, "r").readlines()
            items = extract_structure(lines)
            items["category"] = cat
            csv_rows.append(items)
    
    random.shuffle(csv_rows)
    df = pd.DataFrame(csv_rows)
    if outpath:
        df.to_csv(outpath, index=False, sep="\t")

    return df


# get email structure infor from the spam email dataset
#   dataset_resource : http://ceng.anadolu.edu.tr/par/

def extract_structure(lines):
    from_, _ = find_item(lines, "Kimden:")
    to, _ = find_item(lines, "Kime:")
    subject, subjindex = find_item(lines, "Konu:")
    text = "\n".join(lines[subjindex+1:])
    #return (from_, to, subject, text.strip())
    return {"from" : from_, 
            "to" : to, 
            "subject" : subject, 
            "text" : text.strip()}
 
def find_item(lines, keyword):    
    index_ = 0
    for i,line in enumerate(lines):
        if line.strip().startswith(keyword):
            index_ = i
            item = line[len(keyword):]
            return item.strip(), index_
    return "", -1




if __name__ == '__main__':
    
    
    mainfolder = "/home/dicle/Documents/data/tr_spam/TurkishEmail/char-fixed2"
    csvpath = "/home/dicle/Documents/data/tr_spam/TurkishEmail/tr_spam_800emails.csv"
    spam_mails_to_csv(mainfolder, csvpath)
    
    
    
    