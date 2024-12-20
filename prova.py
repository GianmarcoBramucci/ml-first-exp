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
    # # calcolo del indice gini per vedere la quantita di impurita(per me: va da 0 a 1 piu basse e piu e ordinato fa piu o meno la stessa cosa della entropia solo che ha un costo minore ed anche meno preciso)
    def _calculate_gini(self, y):
        classi,count=  np.unique(y, return_counts=True)
        prob= count/count.sum()
        return 1 - np.sum(prob ** 2)
    # # divisione del dataset a partire da una soglia
    def _split(self,x, y,feat,sol):
        indexSin= x[:,feat]<= sol
        indexDex= x[:,feat]> sol
        return (x[indexSin],y[indexSin],x[indexDex],y[indexDex])
    # # trova la migliore divivisione la migliore solglia (per me: generalmente si usa all inizio per trovare il primo nodo da dove partire)
    def _find_best_split(self,x,y):
        bGini= float("inf")
        bFeat= None
        bSol= None
        for feat in range(x.shape[1]):
            solglie=  np.unique(x[:,feat])
            for sol in solglie:
                xSin,yLSin,xDex, yDex= self._split(x,y,feat,sol)
                if len(yLSin) == 0 or len(yDex) == 0: 
                    continue
                giniSin= self._calculate_gini(yLSin)
                ginDex= self._calculate_gini(yDex)
                # (per me: questa e una media ponderata degli indici di Gini dei due sottoinsiemi (sinistro e destro) che risultano dalla divisione, dove la ponderazione è data dalla dimensione di ciascun sottoinsieme. da rivedere bene come funziona)
                giniMP = (len(yLSin) * giniSin + len(yDex) * ginDex) / len(y)
                if giniMP<bGini:
                    bGini= giniMP
                    bFeat= feat
                    bSol= sol
        return bFeat, bSol	

