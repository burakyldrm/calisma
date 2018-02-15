'''
Created on Dec 29, 2016

@author: dicle
'''

import os
import re


'''
Remove irrelevant text from emails. Signatures are not extracted; if found, deleted.
Only message body kept. 
Specific to KMH data.
'''

'''
some methods' parameter df is of type pandas dataframe.

'''

####

#===============================================================================
# add to markers: "şunları yazdı" if old emails are to be deleted
#===============================================================================
###


m1 = r"_{10,}\s*\n*"
m2 = r"\*{6,}\s*\n*"
m3 = "Mail-imza"
m4 = "Bu mesaj ve ekleri, mesajda gonderildigi belirtilen kisi/kisilere ozeldir ve gizlidir."
m5 = "Bu ileti hukuken korunmuş, gizli veya ifşa edilmemesi gereken bilgiler içerebilir."
m6 = "Çıktı almadan önce"
m7 = "Bu elektronik posta mesajı ve ekleri sadece gönderildiği kişi"
m8 = "YASAL UYARI:"
m9 = "GIZLILIK NOTU:"

markers = [m1, m2, m3, m4, m5, m6, m7, m8, m9]

def substr_index(pattern, text):
    obj = re.search(pattern, text, re.IGNORECASE)
    if obj:
        return obj.start()
    else:
        return -1


def sign_index(markers, text):
    indices = []
    for m in markers:
        indices.append(substr_index(m, text))
        i2 = list(filter(lambda x : x > 0, indices))
        if len(i2) > 0:
            return min(i2)
        else:
            return -1

def get_clean_text(markers, text):
    i = sign_index(markers, text)
    if i < 1:
        return text
    else:
        return text[:i]
    

def remove_html_tags2(text):
    pattern = r"(v\\:\*.+;})|(v\\:\*.+\.\.\.)"
    return re.sub(pattern, "", text).strip()

def strip_rows(df, textcol):
    
    indices = df.index.values.tolist()
    for i in indices:
        text = df.loc[i, textcol]
        df.loc[i, textcol] = text.strip()
    
    return df

isempty = lambda x : x.isspace() or len(x.strip())<1

def remove_empty_rows(df, textcol):
    
    df = strip_rows(df, textcol)
    df = df.loc[~(df[textcol].apply(isempty)), :]
    return df


isnan = lambda x : x != x
def remove_nan_cols(df, colname):
    df = df.loc[~(df[colname].apply(isnan)), :]
    return df
  
if __name__ == '__main__':
    
    print()
    
    
    