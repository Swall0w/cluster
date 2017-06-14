import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score

def data_split(data,num):
    X = data[:,:num]
    y = data[:,num]
    return X,y

if __name__ == "__main__":
# load dataset
#    data = np.loadtxt("seeds_dataset.txt",delimiter='\t')
#    X,y = data_split(data,7)
    digits = datasets.load_digits()
    X = digits.data 
    y = digits.target

# Standardizing data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X = X_scaled

# Clustering with K-means
    estimator = KMeans(n_clusters=10).fit(X)
#    estimator = KMeans(n_clusters=10,verbose=1).fit(X,y)

#    pred = cluster.KMeans(num_clusters=10).fit_predict(X)
#    print(pred)

# Dicreasing dimensions by PCA
    pca = PCA(n_components=2)
    X_projection = pca.fit_transform(X)

## Plot figre
    #cmap = {0:'red',1:'blue',2:'green'}
    cmap = {0:'red',1:'blue',2:'green',3:'pink',4:'salmon',5:'olivedrab',
            6:'yellow',7:'lawngreen',8:'firebrick',9:'limegreen'
    }
    fig, ax = plt.subplots()
    print(adjusted_rand_score(y,estimator.labels_))
    for (i, label) in enumerate(estimator.labels_):
        #ax.scatter(X_projection[i,0],X_projection[i,1],c=cmap[label])
        ax.scatter(X_projection[i,0],X_projection[i,1],c=cmap[label])
#        ax.annotate(str(int(y[i])),(X_projection[i,0],X_projection[i,1]),color=cmap[(int(y[i]))])
    plt.show()
