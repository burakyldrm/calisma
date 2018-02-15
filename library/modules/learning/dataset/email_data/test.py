'''
Created on Nov 7, 2016

@author: dicle
'''


    
for i in range(dfa2.shape[0]):
    sid = dfa2.loc[i, "Sahibi"]
    print(sid, end=" ")
    cat = "UNKNOWN"
    try:
        cat = dfc2.loc[sid, "Bölüm"]
    except:
        pass
    dfa2.loc[i, "category"] = cat
print()
