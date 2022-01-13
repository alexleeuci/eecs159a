from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
import numpy as np

X = [[0, 0], [0, 1], [1, 0],
     [0, 4], [0, 3], [1, 4],
     [4, 0], [3, 0], [4, 1],
     [4, 4], [3, 4], [4, 3]]

X = np.array([[2,1,5,3],
[2,1,3,4],
[3,2,1,4],
[2,1,3,4]])

# Z = ward(pdist(X))

# clusterList = fcluster(Z, t=3, criterion='distance')

print(clusterList)