import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#from func import cluster

def data_split(data,num):
    X = data[:,:num]
    y = data[:,num]
    return X,y

if __name__ == "__main__":
# load dataset
    data = np.loadtxt("seeds_dataset.txt",delimiter='\t')
    X,y = data_split(data,7)

# Standardizing data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X = X_scaled

# sse distoribution
    distortions = []

    for i in range(1,11):
# Clustering with K-means
        KM = KMeans(n_clusters=i,n_init = 10)
        estimator = KM.fit(X)
        distortions.append(KM.inertia_)

    plt.plot(range(1,11),distortions,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum Squared Errort')
    plt.xticks(np.arange(1,11,1))
    plt.show()
# Dicreasing dimensions by PCA
#    pca = PCA(n_components=2)
#    X_projection = pca.fit_transform(X)

# Plot figre
#    cmap = {0:'red',1:'blue',2:'green'}
#    fig, ax = plt.subplots()
#    for (i, label) in enumerate(estimator.labels_):
#        ax.scatter(X_projection[i,0],X_projection[i,1],c=cmap[label])
#        ax.annotate(str(int(y[i])),(X_projection[i,0],X_projection[i,1]),color=cmap[(int(y[i]) - 1)])
#    plt.show()
