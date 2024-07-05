import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,y=make_blobs(n_samples=500,n_features=2,centers=3,random_state=23)

k=3
clsts={}
np.random.seed(23)
for i in range(k):
    center=2*(2*np.random.random((X.shape[1],))-1)
    points=[]
    clst={'center':center,'points':[]}
    clsts[i]=clst
def eucl(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))
def makeclst(X,clsts):
    for i in range(X.shape[0]):
        dist=[]
        curr_x=X[i]
        for j in range(k):
            d=eucl(curr_x,clsts[j]['center'])
            dist.append(d)
        curr_clst=np.argmin(dist)
        clsts[curr_clst]['points'].append(curr_x)
    return clsts
def updateclst(X,clsts):
    for i in range(k):
        points=np.array(clsts[i]['points'])
        if points.shape[0]>0:
            new_cent=points.mean(axis=0)
            clsts[i]['center']=new_cent
            clsts[i]['points']=[]
    return clsts
def predclst(X,clsts):
    pred=[]
    for i in range(X.shape[0]):
        dist=[]
        for j in range(k):
            dist.append(eucl(X[i],clsts[j]['center']))
        pred.append(np.argmin(dist))
    return pred

clsts=makeclst(X,clsts)
clsts=updateclst(X,clsts)
pred=predclst(X,clsts)

plt.scatter(X[:,0],X[:,1],c=pred)
for i in clsts:
    center=clsts[i]['center']
    plt.scatter(center[0],center[1],marker='^',c='red')
plt.show()