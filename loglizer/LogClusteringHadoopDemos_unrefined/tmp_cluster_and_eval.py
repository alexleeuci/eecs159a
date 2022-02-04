from sklearn.cluster import KMeans
from sklearn.utils import resample
import numpy as np
from tmp_rate_clusters import rate_cluster, upsample_class

#load raw numeric data
f = open("Hadoop_EventCount","r")
x, y = len(f.readline().split()), 0
f.close()
f = open("Hadoop_EventCount","r")
for line in f:
    y+=1
f.close()
x_values, y_values = np.zeros((y,x-1)), np.zeros((y,))
f = open("Hadoop_EventCount","r")
index = 0
for line in f:
    data = line.split()
    y_values[index], x_values[index] = data[0], data[1:]
    index+=1

#upsample
x_values, y_values = upsample_class(x_values, y_values, upsample_class=0, ratio=5)
#downsample
x_values, y_values = resample(x_values, y_values, n_samples = 100)

#cluster
#get a list of cluster labels
cluster_count = 80
kmeans = KMeans(init="random", n_clusters=cluster_count, random_state=0).fit(x_values)
cluster_labels = kmeans.predict(x_values)

#evaluate
scores = rate_cluster(cluster_labels, y_values, cluster_count, online_clusters=[], online_classes=[])
print("final scores")
print(scores)