'''
Created on Oct 19, 2016

@author: dicle
'''

'''
-- calculates the average values by normalising the denominator cols by the nominator col.
-- records the avg values in cols avg_denomninator-col 
'''

import sys
sys.path.append("..")

from itertools import permutations

import numpy as np


def get_col_avg(df, denominator_cols=[], nominator_col=""):
    
    avg_prefix = "avg_"
    for denom in denominator_cols:
        
        df[avg_prefix + denom] = df[denom] / df[nominator_col]
        df[avg_prefix + denom] = df[avg_prefix + denom].round(2)
        
    return df



######################
'''
-- normalizes a given matrix at the column level if dim=0, and at at the row level if dim=1. 
'''
def normalize_matrix(countmatrix, dim=1):
    matrix = countmatrix.copy()
    if dim == 1:
        nr, _ = matrix.shape
        for i in range(nr):
            matrix[i, :] = np.divide(matrix[i, :], float(sum(matrix[i, :])))
    
    elif dim == 0:   
        _, nc = matrix.shape
        for j in range(nc):
            matrix[:, j] = np.divide(matrix[:, j], float(sum(matrix[:, j])))
        
    return matrix
##########



########################
'''
 multiplex a dictionary where the keys have list values
'''

# get items of d listed
def _mult_single_dict(d):
    l = []
    
    for k, vals in d.items():
        for v in vals:
            l.append({k : v})
    return l


# combdicts = [{k1 : []}, {k2: []}, ..] and d is a single dict with {k : []}
# return combdicts extended with k:val pairs from d.
def _extend_mult(combdicts, d):
    a = _mult_single_dict(d)
    
    combs = []
    
    for d1 in a:
        for d2 in combdicts:
            d3 = d2.copy()
            d3.update(d1)
            combs.append(d3)
    return combs

# dicts = [{k1 : []}, {k2: []}, ..]
def mult_dict_list(dicts):
   
    if len(dicts) == 1:
        return _mult_single_dict(dicts[0])
    
    combs = _mult_single_dict(dicts[0])
    for i in range(1, len(dicts)):
        combs = _extend_mult(combs, dicts[i])
    return combs


def dict2list(d):
    dicts = []
    for k, val in d.items():
        dicts.append({k : val})
    return dicts


def multiplex_dict(d):
    dicts = dict2list(d)
    return mult_dict_list(dicts)


def printdlist(dicts):
    for d in dicts:
        print(d)

#########################################

'''
get all-size or specific-size combinations of a list
'''
        
def get_all_combs(l):
    combs = []
    for i in range(0, len(l) + 1):
        c = get_r_combs(l, i)
        combs.extend(c)
    combs = sorted(list(set(combs)), key=lambda x : len(x))
    return combs

def get_r_combs(liste, r):
    l = [tuple(sorted(i)) for i in list(permutations(liste, r))]
    l = sorted(list(set(l)))
    return l












##########################################

if __name__ == "__main__":
    
    d = {'a': [{'e': 2, 'r': 3}, {'e': 1, 'r': 5}],
         'b': [{'x': 0}, {'x': 7}],
         'c': [{'q': 9}]}

    l = multiplex_dict(d)
    
    for i, j in enumerate(l):
        print(i, " -  ", j, "....")
    
    '''
    d = {"a" : [1,2,3]}
    print(_mult_single_dict(d))
    
    
    d2 = [{"a" : [1,2,3]}, {"b" : [8,9]}, {"c" : [0,6]}]
    printdlist(mult_dict_list(d2))
    
    d3 = {'a': [91],
         'b': [97],
         'c': [21],
         'd': [94, 81],
         'e': [40, 82, 68],
         'f': [28, 72, 65, 3],
         'g': [37, 39]}
    printdlist(multiplex_dict(d3))


    choices = dict(
                    stopword_choice=(True,False),
                    spellcheck_choice=(False,),
                    stemming_choice=(True, False),
                    number_choice=(True, False),
                    deasc_choice=(True, False),
                    punct_choice=(True, False),
                    case_choice=(True, False),
                    ngramrange=((1, 1), (1, 2), (2, 2)),
                    use_idf=(True, False,),
                    nmaxfeatures=(10000, 1000, None),
                    norm=('l1', 'l2',),
                    )
    
    x = multiplex_dict(choices)
    print(type(x), len(x))
    #printdlist(x)
    
    
    '''




