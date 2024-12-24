import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class DecisionTree:
    def __init__(self,maxProf=3):
        self.maxProf=maxProf
        self.tree= None
    class Node:
        def __init__(self,feat=None,dex=None,sin=None,val=None,sol=None):
            self.feat=feat
            self.dex= dex
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
    # # crea realmete il nostro albero decisionale in maniera ricorsiva 
    def _build_tree(self,x,y,prof):
        if prof>= self.maxProf or len(np.unique(y)) == 1:
            val = np.bincount(y).argmax()
            return self.Node(val=val)
        feat,sol= self._find_best_split(x,y)
        if feat is None:
            val = np.bincount(y).argmax()
            return self.Node(val=val)
        xSin,yLSin,xDex, yDex= self._split(x,y,feat,sol)
        subtreeSin= self._build_tree(xSin,yLSin,prof+1)
        subtreeDex= self._build_tree(xDex,yDex,prof+1)
        return self.Node(feat=feat, sol=sol, sin=subtreeSin, dex=subtreeDex)
    # # addestra l'albero
    def fit(self, x, y):
        self.tree=self._build_tree(x,y,prof=0)
    # # fa una predizione
    def _predict_one(self, x, node):
        if node.val is not None:  
            return node.val
        if x[node.feat] <= node.sol:
            return self._predict_one(x, node.sin)
        else:
            return self._predict_one(x, node.dex)
    # # fa predizione sul intero dataset
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

class GradientBoosting:
    def __init__(self,baseClassM,indexMT=10,badgerL=0.1):
## baseClassM: la classe del modello che vogliamo utilizzare
## indexMT: il numero di modelli che vogliamo addestrare
## badgerL: il tasso di apprendimento (learning rate)
## Questi sono i parametri inizializzati con i valori sopra indicati, e possono essere modificati successivamente o cambiare insieme al nostro modello.
        self.baseClassM= baseClassM
        self.indexMT= indexMT
        self.badgerL= badgerL
        self.models=[]
        self.lostl=[]


###################################TEST#######################################

# Carica il dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividi il dataset in training e test set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

# Crea un'istanza del tuo DecisionTree
tree = DecisionTree(maxProf=3)

# Addestra l'albero decisionale
tree.fit(XTrain, yTrain)

# Fai previsioni sul test set
yPred = tree.predict(XTest)

# Valuta il modello
accuracy = accuracy_score(yTest, yPred)
print("Accuratezza:", accuracy)

#####################RAPRESENTAZIONE GRAFICA (trovata online e aggiustata leggermente per farla funzionare)###############################################

report= classification_report(yTest, yPred)
print(report)

def plot_tree(node, depth=0, x=0.5, y=1.0, spacing=0.1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
    if node.val is not None:
        ax.text(x, y, f"Leaf: {node.val}", bbox=dict(boxstyle="round", facecolor="lightgreen"))
    else:
        ax.text(x, y, f"Feat {node.feat}\n<= {node.sol}", bbox=dict(boxstyle="round", facecolor="lightblue"))
        x_left = x - spacing / (depth + 1)
        x_right = x + spacing / (depth + 1)
        y_child = y - 0.2
        ax.plot([x, x_left], [y, y_child], 'k-') 
        ax.plot([x, x_right], [y, y_child], 'k-')
        plot_tree(node.sin, depth + 1, x_left, y_child, spacing, ax)
        plot_tree(node.dex, depth + 1, x_right, y_child, spacing, ax)

    if depth == 0:
        plt.show()
        
plot_tree(tree.tree)


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(yTest, yPred), annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Decision Tree")
plt.show()

#### CONSIDERAZIONI SUL PROGETTO ####
# Mi è piaciuto molto studiare come funziona un albero decisionale e approfondire 
# la matematica dietro il suo funzionamento (ad esempio, il coefficiente di Gini 
# e le differenze con l'entropia).  
# È stato interessante sviluppare e progettare algoritmi in maniera ricorsiva, 
# comprendere come iniziare e come scegliere la feature migliore da utilizzare 
# per costruire i vari nodi.  

# Tuttavia, devo ammettere che, dopo aver completato queste parti, mi sono leggermente 
# annoiato nella sezione dedicata alla grafica. È stata forse la parte che mi è piaciuta 
# meno, anche se comunque affascinante, perché mi ha dato modo di visualizzare e 
# verificare il lavoro svolto.  

# Come prossimi obiettivi, vorrei implementare il Gradient Boosting e, successivamente, 
# come step finale, utilizzare XGBoost.
