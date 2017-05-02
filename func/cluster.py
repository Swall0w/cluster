import numpy as np

class KMeans(object):
    def __init__(self,num_clusters=3,max_itr=400)
        self.num_clusters = num_clasters
        self.max_itr = max_itr
        self.centers = None

    def _euc_distance(self,x0,x1):
        return np.sum((x0 - x1)**2)

    def fit_predict(self,X):
        feat_vecs = np.arange(len(X))
        np.random.shuffle(feat_vecs)
        initial_centroid_indexes = feat_vecs[:self.num_clusters]
        self.centers = X[initial_centroid_indexes]

        pred = np.zeros(x.shape)

        for _ in range(self.max_itr):
            new_pred = np.array([
                    np.array([
                        self._euc_distance(p, centroid)
                        for centroid in self.centers
                    ]).argmin()
                    for p in X
            ])
            if np.all(new_pred == pred):
                break
                pred = new_pred
    self.centers = np.array([X[pred == i].mean(axis=0)
            for i in range(self.num_clusters)])

        return pred
