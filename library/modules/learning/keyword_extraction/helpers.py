'''
Created on Feb 28, 2017

@author: dicle
'''


# topics_words = { topic : [words] }
def print_topics_words(topics_words):

    for topic, words in topics_words.items():
        print("Topic ", topic)
        print(", ".join(words))
    print()



if __name__ == '__main__':
    
    print()