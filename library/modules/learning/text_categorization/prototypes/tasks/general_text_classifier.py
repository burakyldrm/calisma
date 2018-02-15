'''
Created on May 12, 2017

@author: dicle
'''


import sys
sys.path.append("..")



from sklearn.feature_extraction.text import TfidfVectorizer

from abc import abstractmethod




import sklearn.pipeline as skpipeline

import modules.learning.text_categorization.prototypes.classification.classification_task as clsf_task
import modules.learning.text_categorization.prototypes.feature_extraction.text_preprocessor as prep
import modules.learning.text_categorization.prototypes.feature_extraction.text_based_transformers as txtrans



class GeneralTextClassification(clsf_task._ClassificationTask):
    
    
    def __init__(self, feature_config, classifier, task_name="General Text Classification"):
        self.task_name = task_name
        super().__init__(feature_config, classifier, self.task_name,)
        
        
    
    
    @abstractmethod
    def _generate_feature_extraction_pipeline(self):
        
        lang = self.feature_config.lang
        final_weights = self.feature_config.weights
        prep_params = self.feature_config.prepchoice
   
                   
        # features found in the processed tokens
        token_features = []
        token_weights = {}
    
        preprocessor = prep.Preprocessor(lang=lang,
                                         stopword=prep_params.stopword, 
                                         more_stopwords=prep_params.more_stopwords,
                                         spellcheck=prep_params.spellcheck,
                                         stemming=prep_params.stemming,
                                         remove_numbers=prep_params.remove_numbers,
                                         deasciify=prep_params.deasciify,
                                         remove_punkt=prep_params.remove_punkt,
                                         lowercase=prep_params.lowercase
                                    )
        
        tfidfvect = TfidfVectorizer(tokenizer=prep.identity, preprocessor=None, lowercase=False,
                                    use_idf=prep_params.use_idf,
                                    ngram_range=prep_params.wordngramrange,
                                    max_features=prep_params.nmaxfeature)
    
        tfidfvect_name = 'word_tfidfvect'
        token_features.append((tfidfvect_name, tfidfvect))
        token_weights[tfidfvect_name] = 1
           
        
        
            # features found in the whole raw text
        text_features = []
        text_weights = {}
        # charngramvect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), lowercase=False)
                      
        
        tokenbasedpipe = skpipeline.Pipeline([('preprocessor', preprocessor),
                                              # ('nadropper', tbt.DropNATransformer()),                                       
                                              ('union1', skpipeline.FeatureUnion(
                                                    transformer_list=token_features ,
                                                    transformer_weights=token_weights                                                
                                                    )),
                                            ])
        if text_weights:
            textbasedpipe = skpipeline.Pipeline([('union2', skpipeline.FeatureUnion(
                                                    transformer_list=text_features,
                                                    transformer_weights=text_weights
                                                    ),
                                                  )
                                                ])
        
        
        
        
        '''
        features = skpipeline.FeatureUnion(transformer_list=[
                                            ('tokenbasedfeatures', tokenbasedpipe),
                                            ('textbasedfeatures', textbasedpipe),                                          
                                           ],
                                           transformer_weights=final_weights)
        '''
        #######
        # add the feature pipes to final_features if all the component weights are non-zero.
        ########
        check_zero_list = lambda x : 1 if sum(x) > 0 else 0
        #  l = [0,0,0] => check_zero(l) gives 0 and l=[0,0,1] => check_zero(l) gives 1.
        final_features_dict = {}     
                
        tkweights = list(token_weights.values())
        if(check_zero_list(tkweights) != 0):
            final_features_dict["token_based"] = tokenbasedpipe
        else:
            final_weights["token_based"] = 0
          
        txweights = list(text_weights.values())
        if(check_zero_list(txweights) != 0):
            final_features_dict["text_based"] = textbasedpipe
        else:
            final_weights["text_based"] = 0  
                                            
        final_features = list(final_features_dict.items())    
        
        fweights = list(final_weights.values())
        
        #print(final_weights)
        
        if((check_zero_list(fweights) == 0) or (len(final_features) == 0)):
            return None
        
        
        features = skpipeline.FeatureUnion(transformer_list=final_features,
                                           transformer_weights=final_weights)
        return features




        