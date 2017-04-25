import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def data_split(data,num):
    X = data[:,:num]
    y = data[:,num]
    return X,y

if __name__ == "__main__":
    data = np.loadtxt("seeds_dataset.txt",delimiter='\t')
    X,y = data_split(data,7)
    estimator = KMeans(n_clusters=3)
    result = estimator.fit(X)
    labels = result.labels_

    for label, dat,tr in zip(labels,X,y):
        print(label,tr -1, dat)
