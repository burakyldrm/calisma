'''
Created on Mar 21, 2017

@author: dicle
'''

import os
import re
import pandas as pd

from modules.learning.dataset import io_utils


def get_email(path):
    
    f = open(path, "r")
    content = f.read()
    p1 = "-----Original Message---"
    i1 = content.find(p1)
    i1 = i1 if i1 > -1 else 0
    mail1 = content[:i1]   # remove trailing reply
    
    from_ = re.findall("\From:(.*)\nTo", mail1, re.DOTALL)
    from_ = from_[0] if from_ else ""
    from_ = from_.strip()
    
    p2 = "X-FileName: "
    i2 = mail1.find(p2)
    i2 = i2 if i2 > -1 else 0
    text = mail1[i2+len(p2):]
    text = text.strip()
    
    return text, from_

if __name__ == '__main__':
    
    '''
    p = '/home/dicle/Documents/data/email_datasets/enron/classified/enron_with_categories/2/1825.txt'
    print(get_email(p))
    '''
    
    emails = []
    
    folder = "/home/dicle/Documents/data/email_datasets/enron/classified/enron_with_categories"
    subfolders = io_utils.getfoldernames_of_dir(folder)
    id_ = 0
    for subfolder in subfolders:
        
        p1 = os.path.join(folder, subfolder)
        fnames = io_utils.getfilenames_of_dir(p1, removeextension=False)
        txtfiles = [i for i in fnames if i.endswith(".txt")]
        print(subfolder)
        for txtfile in txtfiles:
            
            print(" Reading ", txtfile)
            p2 = os.path.join(p1, txtfile)
            
            text, from_ = get_email(p2)
            
            emails.append({"fname" : txtfile,
                           "folder" : subfolder,
                           "sender" : from_,
                           "body" : text})
            id_ += 1
    
    print("Done.")
    
    outputpath = "/home/dicle/Documents/data/email_datasets/enron/enron_csv.csv"
    df = pd.DataFrame(emails)
    df.to_csv(outputpath, sep="\t")
    
    
    