#!/usr/bin/python3

""" This script analyzes and disambiguates the tokens from its
    standard input. 

    Note that the disambiguation method used here does not depend 
    on the context. Each word/analysis is evaluated on its own.

    To use this script you need a 'model file'. One can be obtained 
    at www.let.rug.nl/~coltekin/trmorph/1M.m2.

    More information on disambiguation, how to train and use other
    (including context aware) models will follow (hopefully) soon.
"""

import sys
sys.path.append("..")

import json, os, fcntl, re, getopt, io
from subprocess import Popen,PIPE
from math import log

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


a_re = re.compile(r'(<|[^<]+)(<.*)')

def score_m2(model, astring):
    """ We try to estimate the joint probability of the 
        root (r) and the analysis (a). We factor the joint 
        probability as, p(r, a) = p(r|a) p(a).
        the input model contains counts of r|a, and a. 
        For unobserved quantities, we use add-one smoothing.

                          tokenC(r|a) + 1
                p(r|a) = ---------------------
                          tokenC(x|a) + typeC(x|a)

       for all roots x with analysis a. If both r and a are unknown,
       we estimate the probability from the number of times a new root
       was assigned to any analysis in the training data.

                           tokenC(a)
                p(a) = ------------------------------
                         tokenC(words) + typeC(words)
    """

    global a_re
    try:
        m = re.match(a_re, astring)
        r = m.group(1)
        a = m.group(2)
    except:
        print("no match astring: _{}_".format(astring))
        sys.exit(-1)
    ntokens = model["##tokens"]
    ntypes = len(model)

    if a in model:
        rdict = model[a]
        tokC_a = rdict['##tokens']
        typC_a = len(rdict)

        if r in rdict:
            tokC_ra = rdict[r]
        else:
            tokC_ra = 0
        p_a = (tokC_a + 1) / (ntokens + ntypes)
        p_ra = (tokC_ra + 1 ) / (tokC_a + typC_a)
        score = log(p_a) + log(p_ra)
    else:
        p_a = -log(ntokens + ntypes) #log(1 / (ntokens + ntypes))
        p_ra = log(model['##ratypes'] / ntokens)
        score = p_a + p_ra

    return score


def score_astrings(model, alist, word=None):
    slist = []
    for a in alist:
        score = score_m2(model[1], a)
        head, tail = [] , []
        for i in range(0, len(slist)):
            (sc, tmp) = slist[i]
            if sc < score:
                tail = slist[i:]
                break 
            else:
                head.append((sc,tmp))
        slist = head + [(score, a)] + tail
    return slist


def flookup_open(cmd):
    '''
    if cmd is None:
        cmd=flookup_cmd
    '''
    try:
        p = Popen(cmd, shell=True, bufsize=1,
                  stdin=PIPE, stdout=PIPE, stderr=None,
                  universal_newlines=True, close_fds=True)
    except:
        #print("Cannot start flookup with `trmorph.fst'", file=sys.stderr)
        print("Cannot start flookup with `trmorph.fst'")
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(-1)
    return p


def flookup_close(handle):
    handle.stdin.close()
    handle.stdout.close()
    #handle.stderr.close()


def get_analyses(handle, word):
    alist = []
    if len(word) == 0: return alist
    #print(word + "\n")
    handle.stdin.write(word + "\n")
    handle.stdin.flush()
    for astring in handle.stdout:
        a = astring.strip()
        if a:
            if a == '+?': continue 
            alist.append(a)
        else:
            break
    return alist


def usage():
    print("""Usage {} [options]
        This script takes tokens from the standard input, 
        analyzes each token with TRmorph (using flookup),
        scores them and prints the result to standard output

        Options:
        -h, --help          Print this help text, and exit.
        -1, --best-parse    Only print the best analysis.
        -s, --no-score      Do not print the scores, only print the ordered
                            list of analyses.
        -w, --no-word       Do not print the surface word. Useful if
                            you only analyze a single word. Can be
                            confusing if multiple words are analyzed.
        -f, --flookup-cmd   Command to run for obtaining alternative
                            analses. default="flookup -b -x ./trmorph.fst"
        -m, --model-file    The file with the trained model. default='1M.m2'
        -N, --no-newline    Suppress newline between the analyses
        """.format(sys.argv[0]))

#-- main --



   
''' 
if __name__ == "__main__":
    
    sentence = """Seher vakti habersizce girdi gara ekspres
                    kar içindeydi
                    ben paltomun yakasını kaldırmış perondaydım
                    peronda benden başka da kimseler yoktu """

    words = sentence.split()
    
    model_file = '1M.m2'
    mfile = open(model_file, 'r', encoding='utf-8')
    model = json.load(mfile, encoding='utf-8')
    flookup_cmd="../linux64/flookup -b -x ../trmorph.fst"
    trmorph = flookup_open(flookup_cmd)
    mfile.close()
    
    for word in words:
        alist = get_analyses(trmorph, word)
        
        if len(alist) == 0:
            slist = [(-1, '???')]
        else:
            slist = score_astrings(model, alist)
        
        print(alist[0])
        print(slist[0])
        
        print()
        
'''



def _get_roots():
        
    onlybest = True
    model_file = '1M.m2'
   
    flookup_cmd="../linux64/flookup -b -x ../trmorph.fst"
     
    
    try:
        mfile = open(model_file, 'r', encoding='utf-8')
    except:
        print("Cannot open the model file.")
        print("Run `{} --help' for help.".format(sys.argv[0]))
        sys.exit(-1)
    model = json.load(mfile, encoding='utf-8')
    mfile.close()
    
    trmorph = flookup_open(flookup_cmd)
    
    sentence = """Vaktinde 1234 2016 1988 hgfhdgfh habersizce girdi gara ekspres
    kar içindeydi
    ben paltomun yakasını kaldırmış perondaydım
    peronda benden başka da kimseler yoktu """
    lines = sentence.split()
    print(sentence)
    #for line in input_stream:
    roots = []
    for line in lines:
        w = line.strip()
        if len(w) == 0:
            return None
        alist = get_analyses(trmorph, w)
    
        if len(alist) == 0:
            slist = [(-1, '???')]
        else:
            slist = score_astrings(model, alist)
    
        last = len(slist)
        if onlybest:
            last = 1
        for i in range(0, last):
            (sc, a) = slist[i]
            
            print(sc, w, a)
            
            tag_pattern = r"\<(\w+:?\w*)+\>"  # "\<\w{1,4}\>"
            root = re.sub(tag_pattern, "", a)
        
            
            if root.endswith("?"):  # no root could be found, especially for NEs.
                root = w
            
            #root = root.lower()
            roots.append((w, root))
            
            
    return roots




def stem_word(word):
        
    #####onlybest = True  ######
    
    script_dir = os.path.dirname(__file__)
    #print(script_dir)
    model_file = '1M.m2'
    model_file_path = os.path.join(script_dir, model_file)
    
    morph_fst_file = "trmorph.fst"
    #flookup_cmd = "../linux64/flookup -b -x ../trmorph.fst"
    #flookup_cmd = "../linux64/flookup -b -x " + os.path.join(script_dir, morph_fst_file)
    flookup_cmd = "flookup -b -x " + os.path.join(script_dir, morph_fst_file)
    #print(flookup_cmd)
    
    try:
        mfile = open(model_file_path, 'r', encoding='utf-8')
    except:
        print("Cannot open the morphological analysis model file.")
        print("Run `{} --help' for help.".format(sys.argv[0]))
        sys.exit(-1)
    model = json.load(mfile, encoding='utf-8')
    mfile.close()
    
    trmorph = flookup_open(flookup_cmd)
    

    word = word.strip()
    
    if len(word) == 0:
        return word
    
    alist = get_analyses(trmorph, word)

    if len(alist) == 0:
        slist = [(-1, '???')]
    else:
        slist = score_astrings(model, alist)


    flookup_close(trmorph)
    
    (score, root) = slist[0]
    
    
    tag_pattern = r"\<(\w+:?\w*)+\>"  # "\<\w{1,4}\>"
    root = re.sub(tag_pattern, "", root)

    
    if root.endswith("?"):  # no root could be found, especially for NEs.
        root = word
                
    return root



def stem_word_list(words):
    
    #print(words)
    
    #####onlybest = True  ######
    
    script_dir = os.path.dirname(__file__)
    #print(script_dir)
    model_file = '1M.m2'
    model_file_path = os.path.join(script_dir, model_file)
    
    morph_fst_file = "trmorph.fst"
    #flookup_cmd = "../linux64/flookup -b -x ../trmorph.fst"
    #flookup_cmd = "../linux64/flookup -b -x " + os.path.join(script_dir, morph_fst_file)
    flookup_cmd = "flookup -b -x " + os.path.join(script_dir, morph_fst_file)
    #print(flookup_cmd)
    
    try:
        mfile = open(model_file_path, 'r', encoding='utf-8')
    except:
        print("Cannot open the morphological analysis model file.")
        print("Run `{} --help' for help.".format(sys.argv[0]))
        sys.exit(-1)
    model = json.load(mfile, encoding='utf-8')
    mfile.close()
    
    trmorph = flookup_open(flookup_cmd)
    
    
    roots = []
    for word in words:
        #print("word: ",word)
        word = word.strip()
        
        if len(word) == 0:
            root = word
        else: 
            alist = get_analyses(trmorph, word)
        
            if len(alist) == 0:
                slist = [(-1, '???')]
            else:
                slist = score_astrings(model, alist)
        
           
            
            (score, root) = slist[0]
            
            #print("root1", root)
            tag_pattern = r"\<(\w+:?\w*)+\>"  # "\<\w{1,4}\>"
            root = re.sub(tag_pattern, "", root)
            #print("root2", root)
            
            if root.endswith("?"):  # no root could be found, especially for NEs.
                root = word
        
        roots.append(root)
    
    flookup_close(trmorph)
    
    return roots


if __name__ == "__main__":
    '''
    sentence = """Seher vakti habersizce girdi gara ekspres
    kar içindeydi
    ben paltomun yakasını kaldırmış perondaydım
    peronda benden başka da kimseler yoktu """
    
    words = sentence.split()
    
    from time import time

    t0 = time()
    for w in words:
        root = stem_word(w)
        print(w, root)
    t1 = time()
    roots = stem_word_list(sentence.split())
    print(roots)
    t2 = time()
    
    print(t1-t0)
    print(t2-t1)
    '''
    a = ['türkcell', ' ', 'liraya', 'veriyor', 'boş', 'verin', 'aveayı', 'türkcellden', 'alın']
    print(stem_word_list(a))
    
         
''' 

opts, args = getopt.getopt(sys.argv[1:],"h1nsNf:m:",["help","best-parse","no-word", "no-score", 'no-newline', 'flookup-cmd', 'model-file'])

onlybest = True #False
model_file = '1M.m2'
print_score = True
print_word = True
print_newline = True
flookup_cmd="../linux64/flookup -b -x ../trmorph.fst"
for opt, arg in opts:
    if opt in ("-1", "--best-parse"):
        onlybest = True
    elif opt in ("-s", "--no-score"):
        print_score = False
    elif opt in ("-n", "--no-word"):
        print_word = False
    elif opt in ("-f", "--flookup-cmd"):
        flookup_cmd=arg
    elif opt in ("-m", "--model-file"):
        model_file=arg
    elif opt in ("-N", "--no-newline"):
        print_newline = False
    else:
        usage()
        sys.exit(-1)

a_re = re.compile(r'(<|[^<]+)(<.*)')

try:
    mfile = open(model_file, 'r', encoding='utf-8')
except:
    print("Cannot open the model file.")
    print("Run `{} --help' for help.".format(sys.argv[0]))
    sys.exit(-1)
model = json.load(mfile, encoding='utf-8')
mfile.close()

trmorph = flookup_open(flookup_cmd)

sentence = """Seher vakti habersizce girdi gara ekspres
kar içindeydi
ben paltomun yakasını kaldırmış perondaydım
peronda benden başka da kimseler yoktu """
lines = sentence.split()
print(sentence)
#for line in input_stream:
for line in lines:
    w = line.strip()
    if len(w) == 0:
        print(file=output_stream)
        continue
    alist = get_analyses(trmorph, w)

    if len(alist) == 0:
        slist = [(-1, '???')]
    else:
        slist = score_astrings(model, alist)

    last = len(slist)
    if onlybest:
        last = 1
    for i in range(0, last):
        (sc, a) = slist[i]
        if print_word:
            ww = w + " "
        else:
            ww = ""
        if print_score:
            print('{:.2f}: {}{}'.format(sc, ww, a), file=output_stream)
        else:
            print('{}{}'.format(ww, a), file=output_stream)
    if print_newline:
        print(file=output_stream)
            
            
'''         
            


    