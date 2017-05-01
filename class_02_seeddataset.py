import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def data_split(data,num):
    X = data[:,:num]
    y = data[:,num]
    return X,y

if __name__ == "__main__":
# load dataset
    data = np.loadtxt("seeds_dataset.txt",delimiter='\t')
    X,y = data_split(data,7)

# Clustering with K-means
    estimator = KMeans(n_clusters=3).fit(X)

# Dicreasing dimensions by PCA
    pca = PCA(n_components=2)
    X_projection = pca.fit_transform(X)

# Plot figre
    cmap = {0:'red',1:'blue',2:'green'}
    plt.figure()
    for (i,label) in enumerate(estimator.labels_):
        plt.scatter(X_projection[i,0],X_projection[i,1],c=cmap[label])
    plt.show()
