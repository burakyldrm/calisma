'''
Created on Apr 10, 2017

@author: dicle
'''


import os, json
import re

from dataset import io_utils

def thy_extract_qa_pairs(inpath, outpath):
    
    content = open(inpath).read()
    
    items = re.findall("\${3}\n*(.*?)\n*\*{3}\n*", content, re.DOTALL)
    print(len(items))
  
    
    # questions
    questions = [re.findall("\d{1,2}\.(.*?)\?\n", item)[0].strip()+"?" for item in items]
    nq = len(questions)
    
    
    answers = [re.findall("\?\n(.*?)$", item, re.DOTALL)[0].strip() for item in items]
    na = len(answers)
    
    if na != nq:
        print("Error!")
        return "error"
    
    pairs = []
    for q,a in zip(questions, answers):
        #print("\n", q,"\n--->",a,"\n")
        pairs.append({"question" : q, "answer" : a})
    

    with open(os.path.join(outpath), "w") as f:
        json.dump(pairs, f, ensure_ascii=False)

    return pairs



# returns list of dicts = [{question : answer}]
def read_qa_pairs(path):
    
    f = open(path, "r")
    pairs = json.load(f)

    if type(pairs) is list:
        return pairs

    else:
        error = "Reading wrong type error"
        print(error)
        return Exception


def get_database(jsonfolder):
    
    fnames = io_utils.getfilenames_of_dir(jsonfolder, removeextension=False)
    all_pairs = []
    
    for fname in fnames:
        
        p = os.path.join(jsonfolder, fname)
        pairs = read_qa_pairs(p)
        #print(pairs)
        #print(type(pairs))
        #print(type(pairs))
        all_pairs.extend(pairs)
    
    #print(all_pairs[:5])
    instances = [pair["answer"]+"\n"+pair["question"] for pair in all_pairs]
    return instances



def json_to_txt(jsonfolder, txtfolder):
    
    fnames = io_utils.getfilenames_of_dir(jsonfolder, removeextension=False)
    
        
    for fname in fnames:
        
        p = os.path.join(jsonfolder, fname)
        pairs = read_qa_pairs(p)
        
        for i,pair in enumerate(pairs):
            #print(pair)
            txtfname = str(i) + "-" + fname.replace(".json", ".txt")
            content = pair["question"] +"\n\n" + pair["answer"]
            
            txtpath = os.path.join(txtfolder, txtfname)
            outf = open(txtpath, "w")
            outf.write(content)
            outf.close()
    


def _record_thy_qa_pairs():       


    
    infolder = "/home/dicle/Documents/experiments/thy_topics/data"
    outfolder = "/home/dicle/Documents/experiments/thy_topics/jsons"
    fnames = ["online_odul_bilet", "milesNsmiles"]
    
    inext = ".txt"
    outext = ".json"
    
    for fname in fnames:
        inpath = os.path.join(infolder, fname+inext)
        outpath = os.path.join(outfolder, fname+outext)
        x = thy_extract_qa_pairs(inpath, outpath)
        print(type(x))





if __name__ == '__main__':
    
    jsonfolder = "/home/dicle/Documents/experiments/thy_topics/jsons"
    txtfolder = "/home/dicle/Documents/experiments/thy_topics/txts"
    json_to_txt(jsonfolder, txtfolder)
    
    