'''
Created on Oct 12, 2016

@author: dicle
'''

import os

from sklearn.datasets import fetch_20newsgroups

from . import io_utils

def get_20newsgroup():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    dataset = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)
    instances = dataset.data
    labels = dataset.target
    return instances, labels




def read_n_files(folderpath, N):
    
    filenames = io_utils.getfilenames_of_dir(folderpath, False)[:N]
    texts = []
    for fname in filenames:
        fpath = os.path.join(folderpath, fname)
        content = io_utils.readtxtfile2(fpath)
        texts.append(content)
        
    return texts


def get_csv_data(path, delimiter):
    
    categories = []
    textinstances = []
    
    with open(path) as f:
        for line in f:
            data = line.split(delimiter)
            category = str(data[0].replace(' ', ''))
            categories.append(category)
            textinstances.append(data[1])
    
    return textinstances, categories




    
    