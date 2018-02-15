import re
import nltk

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

exampleArray = ['The incredibly intimidating NLP scares people away who are sissies.']


contentArray =['Starbucks is not doing very well lately.',
               'Overall, while it may seem there is already a Starbucks on every corner, Starbucks still has a lot of room to grow.',
               'They just began expansion into food products, which has been going quite well so far for them.',
               'I can attest that my own expenditure when going to Starbucks has increased, in lieu of these food products.',
               'Starbucks is also indeed expanding their number of stores as well.',
               'Starbucks still sees strong sales growth here in the united states, and intends to actually continue increasing this.',
               'Starbucks also has one of the more successful loyalty programs, which accounts for 30%  of all transactions being loyalty-program-based.',
               'As if news could not get any more positive for the company, Brazilian weather has become ideal for producing coffee beans.',
               'Brazil is the world\'s #1 coffee producer, the source of about 1/3rd of the entire world\'s supply!',
               'Given the dry weather, coffee farmers have amped up production, to take as much of an advantage as possible with the dry weather.',
               'Increase in supply... well you know the rules...',]

sents = ["WASHINGTON -- In the wake of a string of abuses by New York City police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
]

def processLanguage():
    try:
        for item in sents:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            print(tagged)

            namedEnt = nltk.ne_chunk(tagged, binary=False)
            
            print(namedEnt)

    except Exception as e:
        print(str(e))
        
def get_named_entity_tokens(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    print(chunked)
    print(len(chunked))

    continuous_chunk = []
    current_chunk = []
    current_type = []
    
    chunk_type = []  # stores (entity, type)

    for (c, chunk) in enumerate(chunked):
        #print(c, "  ->  ", chunk)
        
        if type(chunk) == Tree:
            print(c)
            entity_name = " ".join([token for token, pos in chunk.leaves()])
            current_chunk.append(entity_name) 
            current_type.append((entity_name, chunk.label()))
            print(current_chunk,"  ^ ", current_type)
            
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
        
    return continuous_chunk, current_type



def _getPerformances(text):
    #try:
    performers_list = []
    #text = re.sub(r'\W+\d+\s+.,\'"&', '', text)
    for sent in nltk.sent_tokenize(text):
        for chunk in ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            print(chunk.label())
            if hasattr(chunk, 'node'):
                if chunk.node == "PERSON":
                    performer = ' '.join(c[0] for c in chunk.leaves())
                    performers_list.append(performer)
    return performers_list
    
#processLanguage()
text = "WASHINGTON -- In the wake of a string of abuses by New York City police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement. John saw Mary at the Chicago University."

get_named_entity_tokens(text)

#print(_getPerformances(text))
