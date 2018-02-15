'''
Created on May 8, 2017

@author: dicle
'''



class A():
    
    a = 1
    def x(self):
        print(self.a)

class B(A):
    b = 2
    def x(self):
        print(self.b)

if __name__ == '__main__':
    q = A()
    r = B()

    q.x()
    r.x()
    