# -*- coding: utf-8 -*-
'''
Created on Dec 1, 2016

@author: dicle
'''

import os
import pandas as pd

from talon.signature.bruteforce import extract_signature
import talon
from talon import signature
    



def extract_signatures_rb(emails):
    
    items = [extract_signature(email) for email in emails]
    bodies = [body for body,_ in items]
    signatures = [str(signature) for _,signature in items]
    return signatures


def extract_signatures_ml(emails, senders):
    items = [signature.extract(email, sender=sender) for email, sender in zip(emails, senders)]
    bodies = [body for body,_ in items]
    signatures = [str(signature) for _,signature in items]
    return signatures


def record_signatures(signatures, outfolder, fname1, fname2):
    
    distinct_signatures = list(set(signatures))
    
    open(os.path.join(outfolder, fname1), "w").write("\n".join(signatures))
    open(os.path.join(outfolder, fname2), "w").write("<imza>" + "</imza>\n<imza>".join(distinct_signatures) + "</imza>")

    return distinct_signatures





def tr_emails():    
    '''
    folder = "/home/dicle/Documents/data/emailset2"
    fname = "Musteri-Son-Email-DATA.csv"  
    df = pd.read_csv(os.path.join(folder, fname), sep=";")
    '''
    
    folder = "/home/dicle/Documents/data/emailset2/datalar"
    fname = "Mail_SR_part 1.xlsx"
    df = pd.read_excel(os.path.join(folder, fname))
    
    email_col = "MAIL"
    emails = df[email_col].tolist()
    signatures_rb = extract_signatures_rb(emails)
       
    outfolder = "/home/dicle/Documents/tools/email_signature_cleaner_talon/extractions"
    file_rl1 = "tr_signs_rl1.txt"
    file_rl2 = "tr_signs_rl2.txt"
    s_rb = record_signatures(signatures_rb, outfolder, file_rl1, file_rl2)
    
    
    '''
    open(os.path.join(outfolder, file_rl1), "w").write("\n".join(signatures_rb))
    open(os.path.join(outfolder, file_rl2), "w").write("<imza>" + "</imza>\n<imza>".join(s_rb) + "</imza>")
    print("rl - N: ", len(s_rb))
    '''
    talon.init()
    sender_col = "GONDEREN"
    senders = df[sender_col].tolist()
    signatures_ml = extract_signatures_ml(emails, senders)
    file_ml1 = "tr_signs_ml1.txt"
    file_ml2 = "tr_signs_ml2.txt"
    s_ml = record_signatures(signatures_ml, outfolder, file_ml1, file_ml2)
    print("rl - N: ", len(s_rb))
    print("ml - N: ", len(s_ml))



def enron_emails():    
   
    
    folder = "/home/dicle/Documents/data/email_datasets/enron/classified"
    fname = "enron_body-sender.csv"
    df = pd.read_csv(os.path.join(folder, fname), sep="\t")
    df = df.dropna()
    print(df.shape)
    email_col = "body"
    emails = df[email_col].tolist()
    #emails = emails[:20]
    signatures_rb = extract_signatures_rb(emails)
    
    outfolder = "/home/dicle/Documents/tools/email_signature_cleaner_talon/extractions/en"
    file_rl1 = "en_signs_rl1.txt"
    file_rl2 = "en_signs_rl2.txt"
    s_rb = record_signatures(signatures_rb, outfolder, file_rl1, file_rl2)
    

    talon.init()
    sender_col = "sender"
    senders = df[sender_col].tolist()
    signatures_ml = extract_signatures_ml(emails, senders)
    file_ml1 = "en_signs_ml1.txt"
    file_ml2 = "en_signs_ml2.txt"
    s_ml = record_signatures(signatures_ml, outfolder, file_ml1, file_ml2)
    print("rl - N: ", len(s_rb))
    print("ml - N: ", len(s_ml))
    
    
if __name__ == "__main__":

    enron_emails()
    

#===============================================================================
# 
# if __name__ == '__main__':
#     
#     '''
#     text = """
#      
# Merhaba,
#  
#  
#  
# Alanya mağazamızın talep formu ektedir. Bağlantı yapıldığında bilgi rica ederiz.
#  
#  
#  
# İyi çalışmalar
#  
#  
#  
#               
# Murat AKDAĞ
#  
# İdari işler Uzman Yardımcısı
#  
#  
#  
# Tel:       +90 216 325 7 325 
#  
# Cep:        +90 533 168 33 06 (50043)
#  
# Faks:     +90 216 325 0 815
#      
# Facebook   I  Twitter  I  www.e-bebek.com
#  
#  
# 
# 
#     
#     """
#     '''
#     
#     
#     text = """Wow. Awesome!
#     
#     
# --
# Bob Smith"""
# 
#     '''
# 
#     text = "Wow. Awesome! \
# -- \
# Bob Smith"
#     '''
#     #text = bytearray(text, 'utf-8')
#     #text = text.encode("utf-8")  #.decode("utf-8")
#     print(get_email_parts(text))
# 
# 
#     print(text.split("\n"))
#     
#     talon.init()
# 
#     from talon import signature
#     print(signature.extract(text, sender="smith"))
#     
#     '''
#     import pandas as pd
#     import os
#     p = "/home/dicle/Documents/data/emailset2/has_pstn2.csv"
#     df = pd.read_csv(p, sep=";")
#     #df = df.loc[:50, :]
#     outpath = "/home/dicle/Documents/data/emailset2/signature_extraction"
#     texts = df["MAIL"].tolist()
#     items = [extract_signature(text) for text in texts]
#     texts2 = [body for body,_ in items]
#     signatures = [signature for _, signature in items]
#     df2 = pd.DataFrame(data=[[old,new,sign] for old,new,sign in zip(texts, texts2, signatures)],
#                        columns=["old", "new", "sign"])
#     #df2.to_csv(outpath+"/email.csv", sep="\t")
#     
#     signs = set(signatures)
#     for s in signs:
#         print(s)               
#     
#     '''
#===============================================================================

    
    