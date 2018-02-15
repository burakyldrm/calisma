# -*- coding: utf-8 -*-
'''
Created on 8 Mar 2017

@author: cagatay.kuru
'''


import re
import json
import time

max_edit_distance = 3 
# 0: top suggestion
# 1: all suggestions of smallest edit distance
# 2: all suggestions <= max_edit_distance (slower, no early termination)

dictionary = {}
longest_word_length = 0


def get_deletes_list(w):
    '''given a word, derive strings with up to max_edit_distance characters
       deleted'''
    deletes = []
    queue = [w]
    for d in range(max_edit_distance):
        temp_queue = []
        for word in queue:
            if len(word)>1:
                for c in range(len(word)):  # character index
                    word_minus_c = word[:c] + word[c+1:]
                    if word_minus_c not in deletes:
                        deletes.append(word_minus_c)
                    if word_minus_c not in temp_queue:
                        temp_queue.append(word_minus_c)
        queue = temp_queue
        
    return deletes

def create_dictionary_entry(w):
    '''add word and its derived deletions to dictionary'''

    global longest_word_length
    new_real_word_added = False
    if w in dictionary:
        # increment count of word in corpus
        dictionary[w] = (dictionary[w][0], dictionary[w][1] + 1)  
    else:
        dictionary[w] = ([], 1)  
        longest_word_length = max(longest_word_length, len(w))
        
    if dictionary[w][1]==1:
        
        new_real_word_added = True
        deletes = get_deletes_list(w)
        for item in deletes:
            if item in dictionary:
                # add (correct) word to delete's suggested correction list 
                dictionary[item][0].append(w)
            else:
                # note frequency of word in corpus is not incremented
                dictionary[item] = ([w], 0)  
        
    return new_real_word_added

def create_dictionary(fname1, fname2):

    fname = 'corpus_and_migros.txt'
    f = open(fname, "w")
    f1 = open(fname1, "r")
    f2 = open(fname2, "r")
    f.write(f1.read())
    f.write(f2.read())

    total_word_count = 0
    unique_word_count = 0
    
    pattern = re.compile(u'[\w]+', re.UNICODE)
        
    with open(fname, encoding = "utf-8") as file:
        print ("Ktpphane olusturuluyor...")     
        ctr = 0
        for line in file:
            # separate by words by non-alphabetical characters      
            words = re.findall(pattern, line.lower())  
            for word in words:
                total_word_count += 1
                if create_dictionary_entry(word):
                    unique_word_count += 1
            
            ctr = ctr + 1
            if ctr == 4000:
                break
    
    print(("islenen kelime sayisi: %i" % total_word_count))
    print(("corpus icindeki ozgun kelime sayisi: %i" % unique_word_count))
    print(("kutuphanedeki toplam eleman sayisi: %i" % len(dictionary)))
    print(("  degisen harf sayisi(edit distance): %i" % max_edit_distance))
    print(("  corpus uzerindeki en uzun kelime: %i" % longest_word_length))
    
    print ("Json Dosyası oluşturuluyor,bekleyiniz...")

    # save to file:
    with open('my_file.json', 'w') as f:
        json.dump(dictionary, f)

    # save to file:
    with open('longest_word.txt', 'w') as f2:
        f2.write(str(longest_word_length))

    return dictionary




if __name__ == "__main__":
    
    print ("Bekleyiniz...")
    #time.sleep(2)
    start_time = time.time()
    try:
        create_dictionary('migros_categories.txt', 'corpus.txt')
    finally:
        pass
    run_time = time.time() - start_time
    print ('-----')
    print(('%.2f saniyede calisti' % run_time))
    print ('-----')