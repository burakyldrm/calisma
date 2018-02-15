'''
Created on Feb 21, 2017

@author: dicle
'''
import os
import pandas as pd

from language_tools import TokenHandler
import keyword_extraction.topic_extraction_cluster as tcluster
import keyword_extraction.topic_extraction_decompose as tdecompose
from keyword_extraction import helpers


def get_category_groups(df, cat_col):
    
    index_dict = df.groupby([cat_col]).groups  # {cat : [indices]}
    
    dfs = {}
    for cat, row_indices in index_dict.items():
        cdf = df.loc[row_indices, :]
        dfs[cat] = cdf
    
    return dfs



# all topic extractors should implement their methods in the same structure: same parameters, same return types
def topics_by_category(df, text_col, cat_col, lang="tr"):

    n_clusters = 3
    top_N_words = 30
    
    preprocessor = TokenHandler.TrTokenHandler(stopword=True, more_stopwords=None, 
                                                   stemming=False, 
                                                   remove_numbers=True,
                                                   deasciify=False, remove_punkt=True)
    
    cat_dfs = get_category_groups(df, cat_col)
    
    for cat, cdf in cat_dfs.items():
        texts = cdf[text_col].tolist()
        print(cat)
    
        print("Topics by clustering:")
        clwords = tcluster.extract_topics_kmeans(n_clusters, texts, preprocessor, 
                                                 top_N_words, n_gram_range=(1,1), reduce_dim=False)
        #clwords = tcluster.get_top_N_words(clusterer, vectorizer, n_clusters, top_N_words)
        #print(clwords)
        helpers.print_topics_words(clwords)
    
        print("Topics by LDA:")
        ldawords = tdecompose.extract_topics_lda(texts, preprocessor, n_features=None, 
                                                 n_topics=n_clusters, 
                                                 n_top_words=top_N_words, n_gram_range=(1,1), more_stopwords=None)
        #print(ldawords)
        helpers.print_topics_words(ldawords)

        print("Topics by NMF:")
        nmfwords = tdecompose.extract_topics_nmf(texts, preprocessor, n_features=None, 
                                                 n_topics=n_clusters, 
                                                 n_top_words=top_N_words, n_gram_range=(1,1), more_stopwords=None)
        #print(nmfwords)
        helpers.print_topics_words(nmfwords)

if __name__ == '__main__':
    
    # Topics for the comment data
    folder = "/home/dicle/Documents/experiments/fb_comments"
    fname = "annot-dicle_isbankasi.csv"
    df = pd.read_csv(os.path.join(folder, fname), sep="\t")
    text_col = "message"
    cat_col = "LABEL"
    topics_by_category(df, text_col, cat_col)
    
    