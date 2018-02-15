import modules.learning.text_categorization.prototypes.classification.prep_config as prepconfig
import sklearn.linear_model as sklinear

tr_config_object=prepconfig.FeatureChoice(
    lang    ="tr",
    weights ={
        "word_tfidf":1,
        "polyglot_value":0,
        "polyglot_count":0,
        "lexicon_count":1,
        "char_tfidft":1
    },
    stopword        =True,
    more_stopwords  =None,
    spellcheck      =False,
    stemming        =True,
    remove_numbers  =True,
    deasciify       =True,
    remove_punkt    =True,
    lowercase       =True,
    wordngramrange  =(1, 3),
    charngramrange  =(2, 2),
    nmaxfeature     =100000,
    norm            ="l2",
    use_idf         =True,
    classifier      =sklinear.SGDClassifier(
        loss        ='hinge',
        penalty     ='l2',
        alpha       =1e-10,
        n_iter      =1000,
        random_state=42
    ),
)

mapping = {
    'ilgisiz': 'neutral',
    'Positive': 'positive',
    'positive': 'positive',
    'Negative': 'negative',
    'negative': 'negative',
    'Neutral': 'neutral',
    'neutral': 'neutral',
    'Karar Veremedim': 'neutral',
    'Olumsuz': 'negative',
    'olumsuz': 'negative',
    'Olumlu': 'positive',
    'olumlu': 'positive',
    'Nötr': 'neutral',
    'nötr': 'neutral',
}