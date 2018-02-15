'''
Created on Mar 7, 2017

@author: dicle
'''


import nltk.stem.isri as isri



def _stem1(word):
    stemmer = isri.ISRIStemmer()
    return stemmer.stem(word)


def _stem_light(word):
    
    from tashaphyne.stemming import ArabicLightStemmer

    stemmer = ArabicLightStemmer()
    return stemmer.light_stem(word)



def stem(word):
    #return _stem_light(word)
    return _stem1(word)




if __name__ == "__main__":
    
    
    text = """
    
    عزازيل الذي صنعناه ،الكامن في أنفسنا" يذكرني يوسف زيدان بــ بورخس في استخدامه لحيلته الفنية،وخداع القاريء بأن الرواية ترجمة لمخطوط قديم. الهوامش المخترعة و اختلاق وجود مترجـِم عاد بي إلى بورخس و هوامشه و كتَّابه الوهميين. هذه أولى قراءاتي ليوسف زيدان ،وهو عبقري في السرد ويخلقُ جوَّا ساحرا متفرداً يغرقك في المتعة. هُنا يتجلى الشكُّ الراقي الممزوج بانسانية هيبا الفاتنة ربما تم تناول فكرة الرواية قبلاً ،ولكن هنا تفرداً و عذوبة لا تُقارن بنصٍ آخر كنتُ أودُّ لو صيغت النهاية بطريقة مختلفة فقد جاءت باردة لا تتناسب مع رواية خُطَّت بهذا الشغف . ولذا لا أستطيع منح الرواية خمس نجوم ،وإن كانت تجربة قرائية متفردة وممتعة. 
    
    """
    
    import gtranslate
    en_text = gtranslate.translate_textblob(text)
    print(en_text)
    
    words = text.split()
    
    for w in words:
        stem1 = stem1(w)
        stem2 = stem_light(w)
        
        en_w = gtranslate.translate_textblob(w)
        ens1 = gtranslate.translate_textblob(stem1)
        ens2 = gtranslate.translate_textblob(stem2)
        
        print(w, stem1, stem2, stem1==stem2)
        print(en_w, ens1, ens2,"\n")
    
    

