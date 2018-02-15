'''
Created on Oct 21, 2016

@author: dicle
'''

from polyglot.text import Text


def get_named_entities(text, lang="tr"):
    ptext = Text(text, hint_language_code=lang)  # output can be re-organised
    sentences = ptext.sentences
    entities = []
    for sentence in sentences:
        lang_sentence = Text(str(sentence), hint_language_code=lang)
        s_entities = [(" ".join(entity), entity.tag) for entity in lang_sentence.entities]
        entities.extend(s_entities)
    
    return entities
