'''
Created on Feb 7, 2017

@author: dicle
'''

class NaiveBayes():
def __init__(self,alpha=1.0):
self.name="Naive Bayes"
self.smoothing=alpha
def fit(self,X,y):
y_pos=y>0
y_neg=y<=0
self.pos_prior=y.mean()
self.neg_prior=1-y.mean()
self.pos_lik={}
for i in range(X.shape[1]):
self.pos_lik[i]=(
X[y_pos,i].sum()+self.smoothing/
(X[y_pos].sum()+self.smoothing*X.shape[1])
)
self.neg_lik={}
for i in range(X.shape[1]):
self.neg_lik[i]=(
X[y_neg,i].sum()+self.smoothing/
(X[y_neg].sum()+self.smoothing*X.shape[1])
)
def score(self,X,y):
correct=0
assert X.shape[1]==self.features
for i in range(X.shape[0]):
pos_joint=self.pos_prior
for j in range(X.shape[1]):
if X[i,j]>0:
pos_joint+=np.log(self.pos_lik[j])
neg_joint=self.neg_prior
for j in range(X.shape[1]):
if X[i,j]>0:
neg_joint+=np.log(self.neg_lik[j])
if pos_joint>neg_joint and y[i]>0:
correct+=1
elif neg_joint>=pos_joint and y[i]<=0:
correct+=1
return correct/X.shape[0]