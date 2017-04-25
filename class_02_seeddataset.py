import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def data_split(data,num):
    X = data[:,:num]
    y = data[:,num]
    return X,y

data = np.loadtxt("seeds_dataset.txt",delimiter='\t')
X,y = data_split(data,7)
