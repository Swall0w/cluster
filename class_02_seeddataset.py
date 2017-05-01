import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pylab as pl

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

# Clustering with K-means
    estimator = KMeans(n_clusters=3).fit(X)

# Dicreasing dimensions by PCA
    pca = PCA(n_components=2)
    X_projection = pca.fit_transform(X)

# Plot figre
    cmap = {0:'red',1:'blue',2:'green'}
#    plt.figure()
#    for (i,label) in enumerate(estimator.labels_):
#        #plt.scatter(X_projection[i,0],X_projection[i,1],c=cmap[label])
#        plt.text(X_projection[i,0],X_projection[i,1], str(y[i]), color=cmap[label])
#    plt.show()
    fig, ax = plt.subplots()
#    ax.scatter(X_projection[:,0],X_projection[:,1])
    for (i, label) in enumerate(estimator.labels_):
        ax.scatter(X_projection[i,0],X_projection[i,1],c=cmap[label])
        ax.annotate(str(int(y[i])),(X_projection[i,0],X_projection[i,1]),color=cmap[label])
    plt.show()
