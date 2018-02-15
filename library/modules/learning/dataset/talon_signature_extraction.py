'''
Created on Mar 21, 2017

@author: dicle
'''

import talon

# machine learning method
def extract_signatures_ml(emails, senders):
    from talon import signature
    talon.init()   # required to read the classifier model
    
    items = [signature.extract(email, sender=sender) for email, sender in zip(emails, senders)]
    bodies = [body for body,_ in items]
    signatures = [str(signature) for _,signature in items]
    return bodies, signatures


# rule based method
def extract_signatures_rb(emails):
    from talon.signature.bruteforce import extract_signature

    items = [extract_signature(email) for email in emails]
    bodies = [body for body,_ in items]
    signatures = [str(signature) for _,signature in items]
    return bodies, signatures

if __name__ == '__main__':
    
    texts = ["Hello --\nEmail Sender",
             "Hi --\nEmail Sender2"]
    senders = ["sender@x.com",
               "sender@x.com"]
    
    bodies_ml, signatures_ml = extract_signatures_ml(texts, senders)
    
    bodies_rb, signatures_rb = extract_signatures_rb(texts)
    
    
    for text, s_ml, s_rb in zip(texts, signatures_ml, signatures_rb):
        print(text)
        print("Signature")
        print(" ml: ", s_ml)
        print(" rb: ", s_rb)
        print("\n\n")
        
        
        
        
        
    
    
    
    