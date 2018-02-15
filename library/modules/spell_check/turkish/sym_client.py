# -*- coding: utf-8 -*-

'''
Created on 8 Mar 2017

@author: cagatay.kuru
'''
import re
import json
import time
import os
import fileinput
import string 
from django.conf import settings

BASE_DIR = os.path.join(settings.BASE_DIR, '../modules/spell_check/')

max_edit_distance = 3 
verbose = 0

dictionary = {}
longest_word_length = 0

CHARMAP = {
        "to_upper": {
            u"ı": u"I",
            u"i": u"İ",
        },
        "to_lower": {
            u"I": u"ı",
            u"İ": u"i",
        }
    }

def lower(word):
    for key, value in CHARMAP.get("to_lower").items():
        word = word.replace(key, value)
    return word.lower()

def upper(word):
    for key, value in CHARMAP.get("to_upper").items():
        word = word.replace(key, value)
    return word.upper()


def clean_document(text, lang = 'tr'):
    
    no_html = re.compile('<.*?>')
    just_word = re.compile(u'[\w]+')
    all_words = []
        
    #remove digits, including a123bc -> abc
    result = ''.join(i for i in text if not i.isdigit())

    #Remove HTML Tags
    result = re.sub(no_html,'',result)

    #Remove Punctuation
    no_punct = str.maketrans('', '', string.punctuation)
    result = result.translate(no_punct)

    #Lower case representation in Turkish
    result = lower(result)

    all_lists = re.findall(just_word, result)

    for l in all_lists:
        if len(l) != 0:
            all_words.append(l)

    #output.write(str(all_words))
    return all_words

def get_longest_word():
    with open(BASE_DIR+'turkish/longest_word.txt', 'r') as f2:
        try:
            global longest_word_length
            longest_word_length = int(f2.read())
    # if the file is empty the ValueError will be thrown
        except ValueError:
            dictionary = {}
            return 0
    #os.remove('longest_word.txt')

def get_dictionary():
    with open(BASE_DIR+ 'turkish/my_file.json', 'r') as f:
        try:
            global dictionary
            dictionary = json.load(f)
            return 1
    # if the file is empty the ValueError will be thrown
        except ValueError:
            dictionary = {}
            return 0



def dameraulevenshtein(seq1, seq2):
    
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

def get_suggestions(string, silent=False):
    '''return list of suggested corrections for potentially incorrectly
       spelled word'''
    if (len(string) - longest_word_length) > max_edit_distance:
        if not silent:
            print ("no items in dictionary within maximum edit distance")
        return []
    
    global verbose
    suggest_dict = {}
    min_suggest_len = float('inf')
    
    queue = [string]
    q_dictionary = {}  # items other than string that we've checked
    
    while len(queue)>0:
        q_item = queue[0]  # pop
        queue = queue[1:]
        
        # early exit
        if ((verbose<2) and (len(suggest_dict)>0) and 
              ((len(string)-len(q_item))>min_suggest_len)):
            break
        
        # process queue item
        if (q_item in dictionary) and (q_item not in suggest_dict):
            if (dictionary[q_item][1]>0):
                assert len(string)>=len(q_item)
                suggest_dict[q_item] = (dictionary[q_item][1], 
                                        len(string) - len(q_item))
                # early exit
                if ((verbose<2) and (len(string)==len(q_item))):
                    break
                elif (len(string) - len(q_item)) < min_suggest_len:
                    min_suggest_len = len(string) - len(q_item)
            
            for sc_item in dictionary[q_item][0]:
                if (sc_item not in suggest_dict):
                    
                    # compute edit distance
                    # suggested items should always be longer 
                    # (unless manual corrections are added)
                    assert len(sc_item)>len(q_item)

                    # q_items that are not input should be shorter 
                    # than original string 
                    # (unless manual corrections added)
                    assert len(q_item)<=len(string)

                    if len(q_item)==len(string):
                        assert q_item==string
                        item_dist = len(sc_item) - len(q_item)

                    # item in suggestions list should not be the same as 
                    # the string itself
                    assert sc_item!=string

                    # calculate edit distance using, for example, 
                    # Damerau-Levenshtein distance
                    item_dist = dameraulevenshtein(sc_item, string)
                    
                    # do not add words with greater edit distance if 
                    # verbose setting not on
                    if ((verbose<2) and (item_dist>min_suggest_len)):
                        pass
                    elif item_dist<=max_edit_distance:
                        assert sc_item in dictionary  # should already be in dictionary if in suggestion list
                        suggest_dict[sc_item] = (dictionary[sc_item][1], item_dist)
                        if item_dist < min_suggest_len:
                            min_suggest_len = item_dist
                    
                    # depending on order words are processed, some words 
                    # with different edit distances may be entered into
                    # suggestions; trim suggestion dictionary if verbose
                    # setting not on
                    if verbose<2:
                        suggest_dict = {k:v for k, v in list(suggest_dict.items()) if v[1]<=min_suggest_len}
                
        # now generate deletes (e.g. a substring of string or of a delete)
        # from the queue item
        # as additional items to check -- add to end of queue
        assert len(string)>=len(q_item)
                    
        # do not add words with greater edit distance if verbose setting 
        # is not on
        if ((verbose<2) and ((len(string)-len(q_item))>min_suggest_len)):
            pass
        elif (len(string)-len(q_item))<max_edit_distance and len(q_item)>1:
            for c in range(len(q_item)): # character index        
                word_minus_c = q_item[:c] + q_item[c+1:]
                if word_minus_c not in q_dictionary:
                    queue.append(word_minus_c)
                    q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this
                     
    # queue is now empty: convert suggestions in dictionary to 
    # list for output
    if not silent and verbose!=0:
        print(("number of possible corrections: %i" %len(suggest_dict)))
        print(("  edit distance for deletions: %i" % max_edit_distance))
    
   
    #                                  (frequency in corpus, edit distance)):
    as_list = list(suggest_dict.items())
    outlist = sorted(as_list, key=lambda term_freq_dist: (term_freq_dist[1][1], -term_freq_dist[1][0]))
    
    return outlist




def correctDocument(paramtext, language):
    #dict_okey = get_dictionary()
    #longest_okey = get_longest_word()

    serialized_output = []

    language = 'tr'
    words = clean_document(paramtext, language)

    for word in words:
        start_time = time.time()
        intermediate_dictionary = {}
        all_suggestions = get_suggestions(word)
        suggestions = []

        if len(all_suggestions) == 0:
            intermediate_dictionary = {"word":""}
            intermediate_dictionary['word'] = word

        elif len(all_suggestions) == 1:
            all_suggestions2 = all_suggestions[0]

            all_suggestions_tuple = (all_suggestions2[0], all_suggestions2[1][1])
            suggestions.append(all_suggestions_tuple)

            intermediate_dictionary = {"word":"", "suggestion": ""}
            intermediate_dictionary['word'] = word
            intermediate_dictionary['suggestion'] = suggestions[0]

        elif len(all_suggestions) > 1 and len(all_suggestions) < 5:

            for s in all_suggestions:
                all_suggestions_tuple = (s[0], s[1][1])
                suggestions.append(all_suggestions_tuple)

            intermediate_dictionary = {"word":"", "suggestion": "", "other suggestions": ""}
            intermediate_dictionary['word'] = word
            intermediate_dictionary['suggestion'] = suggestions[0]
            intermediate_dictionary['other suggestions'] = suggestions[1:len(suggestions)]

        elif len(all_suggestions) > 4:
            for s in all_suggestions:
                all_suggestions_tuple = (s[0], s[1][1])
                suggestions.append(all_suggestions_tuple)

            intermediate_dictionary = {"word":"", "suggestion": "", "other suggestions": ""}
            intermediate_dictionary['word'] = word
            intermediate_dictionary['suggestion'] = suggestions[0]
            intermediate_dictionary['other suggestions'] = suggestions[1:4]

        serialized_output.append(intermediate_dictionary)

    output = json.dumps(serialized_output, ensure_ascii=False)

    #with open('output.json', 'w') as f:
    #    f.write(output)

    return output

dict_okey = None


def correctSpell(paramtext):
    global dict_okey

    if dict_okey is None:
        dict_okey = 1
        dict_okey = get_dictionary()
        longest_okey = 1
        longest_okey = get_longest_word()

    if dict_okey == 0 and longest_okey == 0:
        print("Dictionary or longest_word is not laoded")
        print(longest_word_length)
        result="Dictionary or longest_word is not laoded"
    else:        
        language = ""
        result=correctDocument(paramtext, language)
    
    return result
    

if __name__ == "__main__":
    
    print ("Bekleyiniz...")

    start_time = time.time()

    dict_okey = 1
    dict_okey = get_dictionary()
    longest_okey = 1
    longest_okey = get_longest_word()

    if dict_okey == 0 and longest_okey == 0:
        print("Dictionary or longest_word is not laoded")
        print(longest_word_length)
        quit()

    print (" ")
    print ("Kelime duzeltme:")
    #print(longest_word_length)
    print ("---------------")


    input_text="helo i am goingt to my hame"
    #input_text = "bedim adım Ali Çağatay Kuru, 23 YAŞINDAYIM ve 2017yıtında Bilgisayar MÜhendİsliğinden mezun oldum. Html Tag'lerim aşağıdadır: <html><h1>Bu bir başlıktır </h1><h2>'Bu tırnaklı bir başlıktır'</h2></html> Bakalım nasıl 1 sonuç ovtaya çıkıcak."    
    language = ""
    print(correctDocument(input_text, language))


    run_time = time.time() - start_time
    print(run_time)

    """
    while True:
        word_in = input("input: ")
        word_in = word_in.lower()
        if len(word_in)==0:
            break
        start_time = time.time()
        print((get_suggestions(word_in)))
        run_time = time.time() - start_time
        print ('-----')
        print(('%.5f saniye surdu' % run_time))
        print ('-----')
    """

    