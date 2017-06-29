from random import random
import random
import numpy as np
	
#Swap two elements of an array   
def Swap(ary, x, y):
    z = ary[x]
    ary[x] = ary[y]
    ary[y] = z
    
#Shuffle two arrays together in the same way
def Shuffle(ary1,ary2):
    l = ary1.shape[0]
    x = l-1
    while(x>0):
	r = random.randint(0,x)
	Swap(ary1,x,r)
	Swap(ary2,x,r)
	x = x-1

#Sign function
def sign(x):
    if x > 0:
        return 1
    return -1


#Create a perceptron classifier
class Perceptron:
    
    dim = 1
    m = np.zeros(1)
    w = np.array([0.0])
    b = 0.0
    cw = np.zeros(1)
    quartw = np.zeros(1)
    
    def __init__(self,n,rnd=False):
        self.dim = n
	self.w = np.zeros(n)
	self.m = np.zeros((n,n))
	self.cw = np.zeros((n,n,n))
	self.quartw = np.zeros((n,n,n,n))
        if rnd:
            self.b += 5*(2*random.random() -1)
	    for i in range(self.dim):
		self.w[i] += 20*(2*random.random() -1)
	    for i in range(self.dim):
                for j in range(self.dim):
		    self.m[i][j] += 10*(2*random.random() -1)
        
    def softPredict(self,x):
	xx = np.array(x)
        ret = np.dot(self.w,xx) + self.b
	ret += np.dot(xx,np.dot(self.m,xx))
	#ret += np.einsum("ijk,i,j,k->",self.cw,xx,xx,xx)
	#ret += np.einsum("ijkl,i,j,k,l->",self.quartw,xx,xx,xx,xx)
	return ret
    
    #Classify vector x   
    def hardPredict(self,x):
        return sign(self.softPredict(x))
    
    #Classify an array of vectors  
    def aryPredict(self,datax):
        ret = []
	dx = np.array(datax)
	l = dx.shape[0]
	for i in range(l):
	    ret.append(self.hardPredict(dx[i]))
	return ret
    
    #update on vector x
    def update(self,x,y):
	xx = np.array(x)
	if self.hardPredict(x) != y:
	    self.b = self.b + y
	    self.w = self.w + y*xx
	    self.m = self.m + y*np.outer(xx,xx)
	    #self.cw = self.cw + y*np.einsum("i,j,k->ijk",xx,xx,xx)
	    #self.quartw = self.quartw + y*np.einsum("i,j,k,l->ijkl",xx,xx,xx,xx)

    def train(self,datax,datay,rnd=False):
	dx = np.array(datax).copy()
	dy = np.array(datay).copy()
        if rnd:
            Shuffle(dx,dy)
	l = dx.shape[0]
	for i in range(l):
	    self.update(dx[i],dy[i])

    def pdot(p,q):
        ret = p.b*q.b + np.dot(p.w,q.w) 
        for i in range(p.dim):
            for j in range(p.dim):
                ret += p.m[i][j]*q.m[i][j]
        return ret

    def overlap(p,q):
        x = Perceptron.pdot(p,q)
        y = np.sqrt(Perceptron.pdot(p,p)*Perceptron.pdot(q,q))
        if y == 0:
            return 1
        return x/y

#Make a bunch of perceptrons and have them vote
class PerceptronHerd:

    dim = 1
    herd = []
    TProb = []
    FProb = []
    threshhold = 0.1
    fracT = 1
    
    def __init__(self,n,x):
        self.dim = n
        for i in range(x):
            self.herd.append(Perceptron(n,True))
	    self.TProb.append(0)
	    self.FProb.append(0)

    def train(self,datax,datay):
	dx = np.array(datax)
	dy = np.array(datay)

        for p in self.herd:
            p.train(dx,dy,True)

        i = 0
        j = 1
        while i < j:
            while j < len(self.herd):
                if Perceptron.overlap(self.herd[i],self.herd[j]) > self.threshhold :
                    del self.herd[j]
                else:
                    j+=1
	    i+=1

        totT = 0
        totF = 0
        for k in range(len(self.herd)):
            for  l in range(dx.shape[0]):
                y1 = self.herd[k].hardPredict(dx[l])
                y2 = dy[l]
                if y2==1:
                    totT +=1
    		    if y1 == 1:
                        self.TProb[k] += 1
                elif y2==-1:
		    totF += 1
    		    if y1 == 1:
                        self.FProb[k] += 1
        self.TProb = [a/totT for a in self.TProb]
        self.FProb = [a/totF for a in self.FProb]
        self.fracT = totT/(totT + totF)

    def vote(self,x):
        ret = 0 
        for p in self.herd:
            ret += p.hardPredict(x)
        return ret

    def softVote(self,x):
        ret = 0 
        for p in self.herd:
            ret += p.softPredict(x)
        return ret

    def prob(self,x,prior):
        prb = prior
        for k in range(len(self.herd)):
            if self.herd[k].hardPredict(x) == 1:
                top = (self.TProb[k] * prb)
		bottom = (self.TProb[k] * prb + (1-self.TProb[k])*(1-prb))
                if bottom != 0:
                    prb = top/bottom
            else:
                top = ((1-self.FProb[k]) * (1-prb))
		bottom = (self.FProb[k] * prb + (1-self.FProb[k])*(1-prb))
                if bottom != 0:
                    prb = 1-(top/bottom)
        return prb
            

    def predict(self,datax):     
	ret = []
        dx = np.array(datax)
	l = dx.shape[0]
	#for i in range(l):
	#    ret.append(sign(2*self.prob(dx[i],self.fracT) - 1))
	for i in range(l):
	    ret.append(sign(self.softVote(dx[i])))
	return ret

class PerceptronTree:

    dim = 1
    depth = 2
    me = None
    left = None
    right = None
    
    def __init__(self,n,d):
        self.dim = n
        self.depth = d
        self.me = Perceptron(n)
        if d>1:
            self.left = PerceptronTree(n,d-1)
            self.right = PerceptronTree(n,d-1) 
	


    def train(self,datax,datay):
	dx = np.array(datax)
	dy = np.array(datay)
	self.me.train(datax,datay,True)
	leftDataX = []
	leftDataY = []
	rightDataX = []
	rightDataY = []
	for i in range(dx.shape[0]):
            if self.me.hardPredict(dx[i]) == 1:
                leftDataX.append(dx[i])
                leftDataY.append(dy[i])
            else:
                rightDataX.append(dx[i])
                rightDataY.append(dy[i])
        if self.left == None:
            return 0
        self.left.train(leftDataX,leftDataY)
        if self.right == None:
            return 0
        self.right.train(rightDataX,rightDataY)
        return 0

    def predictOne(self,x):
        verdict = self.me.hardPredict(x)
        if self.left == None or self.right == None:
            return verdict
        if verdict==1:
            return self.left.predictOne(x)
        return self.right.predictOne(x)


    def predict(self,datax):
        ret = []
	dx = np.array(datax)
	l = dx.shape[0]
	for i in range(l):
	    ret.append(self.predictOne(dx[i]))
	return ret

#Make a bunch of perceptrons trees and have them vote
class PerceptronForrest:

    dim = 1
    depth = 2
    herd = []
    
    def __init__(self,n,d,x):
        self.dim = n
        for i in range(x):
            self.herd.append(PerceptronTree(n,d))

    def train(self,datax,datay):
	dx = np.array(datax)
	dy = np.array(datay)
        for p in self.herd:
            p.train(dx,dy)

    def vote(self,x):
        ret = 0
        for p in self.herd:
            ret += p.predictOne(x)
        return ret

    def predict(self,datax):
        ret = []
        dx = np.array(datax)
	l = dx.shape[0]
	for i in range(l):
	    ret.append(sign(self.vote(dx[i])))
	return ret

def Main():
    print "Hello World"
    ary = np.array([1,2,3,4])
    Shuffle(ary)
    print ary
    print

if __name__ == "__main__":
    Main()
