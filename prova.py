import numpy as np

class DecionTree:
    def __init__(self,maxProf=3):
        self.maxDeap=maxProf
        self.tree= None
class Node:
    def __init__(self,feat,des,sin,val,sol):
        self.feat=feat
        self.des= des
        self.sin=sin
        self.val= val
        self.sol= sol
    def _calculate_gini(self, y):
        classi,count=  np.unique(y, return_counts=True)
        prob= count/count.sum()
        return 1 - np.sum(prob ** 2)
    def _split(self,x, y,feat,sol):
        indexSin= x[:,feat]<= sol
        indexDex= x[:,feat]> sol
        return (x[indexSin],y[indexSin],x[indexDex],y[indexDex])
    