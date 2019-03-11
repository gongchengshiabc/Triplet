import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
def CalL2dist(S,I):
    dist=cosine_similarity(S,I)
    return dist
def CalEucdist(S,I):
    dist = cdist(S, I, metric='euclidean')
    return dist
def CalcReAcc(q, r,topk):
    num_query=q.shape[0]
    topkacc=0
    count=0
    dist=CalEucdist(q,r)
    inds=np.argsort(dist,axis=1)
    for index in range(num_query):
        ind=inds[index]
        tind=ind[0:topk]
        count +=index in tind
    topkacc=count/(num_query*1.0)
    return topkacc
def CalcReMap(q,r,ql,rl,topk):
    num_query=q.shape[0]
    topmap=0
    dist=CalEucdist(q,r)
    inds=np.argsort(dist,axis=1)
    for index in range(num_query):
        ind=inds[index]
        tind=ind[0:topk]
        count=1
        ap=0
        for i in range(topk):
            if ql[index]==rl[tind[i]]:
                ap=ap+(count*1.0)/(i+1.0)
                count=count+1
        topmap=topmap+ap/2
        print(topmap)
    topmap=topmap/num_query
    return topmap
S=np.array(([[1,3,1],[1,1,1]]))
I=np.array(([[1,0,1],[1,1,0],[0,2,2]]))
ql=[4,5]
rl=[4,5,4]

acc=CalcReMap(S,I,ql,rl,topk=3)
print(acc)