'''
Created on Apr 11, 2017

@author: dicle
'''
# import modules & set up logging
import gensim, logging

import os

def txt_to_sentences(filepath):
    
    content = open(filepath, "r").read()
    
    import nltk.data
    tr_tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')
    sentences = tr_tokenizer.tokenize(content)
    sentences = [s.strip() for s in sentences]
    return sentences



def save_trial_embeddings():
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
     
    filepath = "/home/dicle/Documents/data/tr_plain_texts/tr_text_compilation.txt"
    
    
    
    i_sentences = ["Türk Sanayicileri ve İşadamları Derneği (TÜSİAD) Yönetim Kurulu Başkanı Arzuhan Doğan Yalçındağ, IMF ile yapılacak anlaşmayla sağlanacak kredi imkanının, uluslararası likiditenin kuruduğu bu dönemde, Türkiye'nin nefes almasına imkan sağlayacağını bildirdi.",
 'Yalçındağ, TÜSİAD, Dış Ekonomik İlişkiler Kurulu (DEİK) ve İstanbul Menkul Kıymetler Borsası (İMKB) işbirliği ile düzenlenen "Küresel Ekonomi ve Türkiye" konulu toplantıda yaptığı konuşmada, IMF ile yapılacak anlaşmaya değindi.',
 'Yalçındağ, "IMF ile yapılacak bir anlaşma, piyasalara ilişkin  belirsizliği ortadan kaldıracak, uygulanacak ekonomi politikalarının genel çerçevesini çizerek geleceğin öngörülebilir olmasını sağlayacak ve uluslar arası piyasalarda Türkiye\'nin kredibilitesini artıracaktır.',
 'Ayrıca, anlaşma ile sağlanacak kredi imkanı, uluslararası likiditenin kuruduğu bu dönemde bir yıl içinde yaklaşık 50 milyar doları bulan dış borç ödemesi olan Türkiye\'nin nefes almasına da imkan sağlayacaktır" diye konuştu.',
 '-GERİLEMENİN YAVAŞLAMASI SOMUT ÖNLEMLERLE MÜMKÜN-   Yalçındağ, ekonomideki gerilemenin yavaşlamasının ancak, yerinde somut önlemlerle mümkün olabileceğini kaydetti.',
 "Yalçındağ, küresel ekonominin İkinci Dünya Savaşı'ndan bu yana karşı karşıya olduğu en sıkıntılı dönemde, OECD Genel Sekreteri Angel Gurria'nın katılımıyla düzenlenen toplantıyı son derece anlamlı bulduğunu, önemsediğini söyledi.",
 "Geçen sene Temmuz ayında Amerika'da düşük kaliteli konut kredisine dayalı kağıtlarda başlayan sorunun, kısa sürede tüm bankacılık sektörüne yayıldığını, Avrupa kıtasını etkisi altına aldıktan sonra, reel sektöre sıçradığını ve daha geçen seneye kadar krizden etkilenmeyeceği düşünülen gelişmekte olan ülkelere de sirayet ettiğini anımsatan TÜSİAD Başkanı Yalçındağ, üç ay önce Lehman Brothers'ın batışının, krizin ciddiyetinin herkes tarafından daha iyi anlaşılmasını sağladığını dile getirdi.",
 'Yalçındağ, şöyle devam etti: "Bugün artık biliyoruz ki, ne kadar sağlam olursa olsun, ne kadar iyi yönetilirse yönetilsin, dünya üzerindeki tüm ülkeler, tüm sektörler, tüm şirketler için krizin bulaşma hızı ve şiddeti ciddi bir tehdit oluşturmaktadır.']
    
    i_sentences = txt_to_sentences(filepath)
    
    import nltk.tokenize as tokenizer
    sentences = []
    for s in i_sentences:
        tokens = tokenizer.word_tokenize(s.encode("utf-8").decode("utf-8"), language="turkish")
        sentences.append(tokens)
    #sentences = [s.encode("utf-8").decode("utf-8").split() for s in sentences]
    
    
    print(sentences[:5])
    #sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=1)
    
    outfolder = "/home/dicle/Documents/experiments/embeddings"
    modelname = "tr_25mb.txt"
    outpath = os.path.join(outfolder, modelname)
    model.wv.save_word2vec_format(outpath, binary=False)
    #model.wv.save(outpath)


def load_trial_embeddings():

    
    mpath = "/home/dicle/Documents/experiments/embeddings/tr_25mb.txt"
    model = gensim.models.KeyedVectors.load_word2vec_format(mpath, binary=False, encoding="utf-8")  #, encoding='ISO-8859-1')
    #model = gensim.models.Word2Vec.load_word2vec_format(mpath)
    words = ["şiir", "gazete", "Ankara", "Diyarbakır", "ve"]
    
    vocab = model.vocab
    print(type(vocab))
    print(list(vocab.items())[0])
    for w in words:
        if w not in vocab:
            print(w, " not in vocab")
        else:
            print(model.most_similar([w]))
    
    
    
if __name__ == "__main__":

    #save_trial_embeddings()

    load_trial_embeddings()
    
    
    #model.accuracy('/tmp/questions-words.txt')

    
