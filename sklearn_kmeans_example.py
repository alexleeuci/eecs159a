from sklearn.cluster import KMeans
import numpy as np
X = np.array([  [0,0],[-1,0],[0,-1],[-1,-1],
                [5,5],[4,5],[5,4],[4,4]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
pred = kmeans.predict([ [0, 0.5],
                        [4, 4.5],
                        [0, -0.5],
                        [-0.5, 0],
                        [-0.2, 0.2],
                        [4, 5.2],
                        [6.2, 4.5]])
print(pred)