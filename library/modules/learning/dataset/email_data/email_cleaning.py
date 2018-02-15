'''
Created on Sep 7, 2016

@author: dicle
'''

import os
import re
from modules.learning.dataset import io_utils


DATE = "Sent:"
TO = "To:"
CC = "Cc:"
SUBJECT = "Subject:"

out_header = ["fileid", "From", "To", "Cc", "Date", "Subject", "len(body)"]

def readfile(path):
    
    content = io_utils.readtxtfile(path)
    return content


def parse_email(email):

    lines = email.split("\n")
    for i,line in enumerate(lines):
        print(i,"  ",line)
    


def detect_illstructured(mainfolder, outcsvpath, bodyfolder):

    fnames = []

    folders1 = io_utils.getfoldernames_of_dir(mainfolder)
    
    ngoodfiles = 0
    
    io_utils.initialize_csv_file(out_header, outcsvpath)
    
    for folder1 in folders1:
        # assuming the corpus has one more subfolder hierarchy
        p1 = os.path.join(mainfolder, folder1)
        txtfiles = io_utils.getfilenames_of_dir(p1, removeextension=False)

        # check fnames
        fnames.extend(txtfiles)

        
        for txtfile in txtfiles:
                
            fpath = os.path.join(p1, txtfile)
                
            # if line 0 or 1 has Sent:
            with open(fpath) as f:
                lines = f.readlines()
                
                date = lines[1]
                datep = re.match(r"\s*"+DATE, date)
                if datep:
                    
                    to = lines[2]
                    cc = lines[3]
                    subject = lines[4]
                    
                    date2 = extract_metadata(date, DATE)
                    to2 = extract_metadata(to, TO)
                    cc2 = extract_metadata(cc, CC)
                    subject2 = extract_metadata(subject, SUBJECT)
                    
                    bodylines = [i for i in lines[5:] if not i.isspace()]
                    body = "\n".join(bodylines)
                    body = body.strip()
                    
                    # record body aside
                    io_utils.todisc_txt(body.decode("utf-8"), os.path.join(bodyfolder, txtfile))
                    
                    items = [txtfile, "", to2, cc2, date2, subject2, str(len(body))]
                    io_utils.append_csv_cell_items(items, outcsvpath)
                
                '''
                datep = re.match(r"\s*"+DATE, txt)
                if datep:
                    if "@" in lines[0]:
                        ngoodfiles += 1
                    else:
                        print "- ", fpath
                    datestr = txt[datep.end():]
                else:
                    print fpath
                
                '''
                
            
    
    print("nfiles: ", str(len(fnames)))
    print("ngoodfiles: ", str(ngoodfiles))


def extract_metadata(rawtxt, pattern):
    
    p = re.match(r"\s*"+pattern, rawtxt)
    if p:
        txt = rawtxt[p.end():]
        return txt.strip()
    else:
        return "-"
 
 

# clean html tags, remove lines missing messages (containing only large signature texts)
def clean_texts(df, text_col):
    
    return

       

if __name__ == '__main__':
    
    
    '''
    folderpath = "/home/dicle/Documents/data/sample_emails/mail_first/015/"
    fname = "ausgabe912.txt"
    path = os.path.join(folderpath, fname)
    
    email = readfile(path)
    parse_email(email)
    '''
    
    mainfolder = "/home/dicle/Documents/data/sample_emails/mail_first/"
    outcsvpath = "/home/dicle/Documents/experiments/pattern_analysis/emails.csv"
    bodyfolder = "/home/dicle/Documents/data/sample_emails/bodies_mail_first"
    detect_illstructured(mainfolder, outcsvpath, bodyfolder)
    
    # @TODO handle ill-structured texts. clean the body.
    