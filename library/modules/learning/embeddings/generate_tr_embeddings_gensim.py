'''
Created on Apr 11, 2017

@author: dicle
'''
# import modules & set up logging
import gensim, logging

from time import time

import os

def txt_to_sentences(filepath):
    
    content = open(filepath, "r").read()
    
    import nltk.data
    tr_tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')
    sentences = tr_tokenizer.tokenize(content)
    sentences = [s.strip() for s in sentences]
    return sentences



def generate_embeddings(huge_txt_path, outfolder, modelname,
                        _size=50, _min_count=1):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    print("Read the text from ", huge_txt_path)
    print("\nSplitting the text into sentences..")
    i_sentences = txt_to_sentences(huge_txt_path)
    
    import nltk.tokenize as tokenizer
    sentences = []
    
    print("\nTokenizing the sentences..")
    for s in i_sentences:
        tokens = tokenizer.word_tokenize(s.encode("utf-8").decode("utf-8"), language="turkish")
        sentences.append(tokens)
    #sentences = [s.encode("utf-8").decode("utf-8").split() for s in sentences]
    
    
    print(sentences[:5])
    #sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    
    print("\nGenerating word vectors..")
    t0 = time()
    model = gensim.models.Word2Vec(sentences, size=_size, min_count=_min_count)
    t1 = time()
    print("Generating vectors finished. Took ", str(t1-t0), "sec.")
    
    outpath = os.path.join(outfolder, "s-"+str(_size)+"_min-count-"+str(_min_count)+"__"+modelname)
    print("\nSaving the vectors in ", outpath)
    model.wv.save_word2vec_format(outpath, binary=False)
    t2 = time()
    print("Recording finished. Took ", str(t2-t1), "sec.")
    #model.wv.save(outpath)

    return outpath

def load_embeddings(model_path):

    print("Loading the embeddings from ", model_path)
    t0 = time()
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf-8")  #, encoding='ISO-8859-1')
    t1 = time()
    print("Loading finished. Took ", str(t1-t0), "sec.")
    #model = gensim.models.Word2Vec.load_word2vec_format(mpath)
    words = ["şiir", "gazete", "Ankara", "Diyarbakır", "ve"]
    
    vocab = model.vocab
    print(type(vocab))
    print(list(vocab.items())[0])
    for w in words:
        print()
        if w not in vocab:
            print(w, " not in vocab")
        else:
            print(w, " --- most similar words --- : ", model.most_similar([w]))
        
    
    
    
if __name__ == "__main__":

    # GENERATE EMBEDDINGS
       
    txtfilefolder = "/home/dicle/Documents/data/tr_plain_texts/"
    txtfilename = "tr_text_compilation.txt"
    txtfilepath = os.path.join(txtfilefolder, txtfilename)
    outfolder = "/home/dicle/Documents/experiments/embeddings/25MBtrText"
    modelname = txtfilename
    
    
    modelpath = generate_embeddings(txtfilepath, outfolder, modelname, _size=200, _min_count=2)
    
    # LOAD EMBEDDINGS
    
    load_embeddings(modelpath)
    
    