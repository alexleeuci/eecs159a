from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.utils import resample,shuffle
# this is exactly just kmeans, but we generate some random points
# the reason is that if each cluster has at least 1 data point, then
# when cluster count == data count, each data item is exactly just
# its own cluster and the accuracy should rocket up to close to 100:
# the reason it doesn't seem to happen is because of the oversampling

# so we replicate the oversampling, and try to find the number of clusters
# that "spoofs" an extremly high degree of accuracy
# then, we have demonstrated that the paper presents
# somewhat of a ill-formed problem
# (at least with publicly available data)

total_0_values = 20
total_1_values = 100
vecLen = 100
cluster_count = 200
x_values_0 = np.array([])
# 1) make some x values where y is 0
x_values_tostack_0 = []
for i in range(int(total_0_values)):
    x_values_tostack_0.append(np.array([random.randint(0,100) for j in range(vecLen)]))
x_values_0 = np.vstack(x_values_tostack_0)
print(x_values_0)
y_values_0 = np.array([0 for i in range(total_0_values)])
# 1b) resample these values
x_values_0, y_values_0 = resample(x_values_0, y_values_0, n_samples = total_1_values)
print(len(x_values_0))
# 2) make some x values where y is 1
x_values_tostack_1 = []
for i in range(int(total_1_values)):
    x_values_tostack_1.append(np.array([random.randint(0,100) for j in range(vecLen)]))
x_values_1 = np.vstack(x_values_tostack_1)
print(x_values_1)
y_values_1 = np.array([1 for i in range(total_1_values)])
# 3) combine x and y, then shuffle
x_values = np.vstack([x_values_0, x_values_1])
y_values = np.append(y_values_0, y_values_1)
print("x values, len:", len(x_values))
print(x_values)
print("y values, len:", len(y_values))
print(y_values)


cluster_to_average = dict()

kmeans = KMeans(init="random", n_clusters=cluster_count, random_state=0).fit(x_values)
pred_idx = kmeans.predict(x_values)
centers = kmeans.cluster_centers_


# we've gotten some clusters; now lets get the indicies of the data points in each cluster
# and then find the ave class of each cluster
#https://stackoverflow.com/questions/62626305/how-could-i-find-the-indexes-of-data-in-each-cluster-after-applying-the-pca-and
cluster_labels=kmeans.labels_ # get cluster label of all data
print("cluster labels of points:", cluster_labels)
# labels_ is a list such that list[index]=the cluster data item x_train[index] belongs to
# aka; if labels_ = [2,3,3,1,0,0...]
# then x_train[0] is in cluster 2
# x_train[1] is in cluster 3
# x_train[2] is in cluster 3
# x_train[3] is in cluster 1
# etc...

# get indexes of points in each cluster 
#Note: you can use these indexes in both data and data2
zero_one_threshold = 0.5
for i in range(cluster_count):
    index_cluster=np.where(cluster_labels==i)[0] # get indexes of points in cluster i
    classification_list = y_values[index_cluster] # get classification of each point in cluster i
    print("=================",i,"=================")
    print(classification_list)
    cluster_to_average[i] = 1 if classification_list.mean()>zero_one_threshold else 0
    print("")
print("cluster to average:")
print(cluster_to_average)


# fit using the dict above
y_test_prediction_class = kmeans.predict(x_values)
y_test_prediction = np.array([cluster_to_average[c] for c in y_test_prediction_class])
print("prediction of classes")
print(y_test_prediction_class)
print("prediction based on kmeans")
print(y_test_prediction)
print("actual values")
print(y_values)

print("accuracy")
total_correct = sum([1 if y_test_prediction[i]==y_values[i] else 0 for i in range(len(y_test_prediction))])
print(total_correct / len(y_test_prediction))


y_train_prediction_class = kmeans.predict(x_values)
y_train_prediction = np.array([cluster_to_average[c] for c in y_train_prediction_class])
print("prediction of classes")
print(y_test_prediction_class)
print("prediction based on kmeans")
print(y_train_prediction)
print("predictions from kmeans.clusters_")
print(kmeans.labels_)
print("actual values")
print(y_values)

print("accuracy")
total_correct = sum([1 if y_train_prediction[i]==y_values[i] else 0 for i in range(len(y_train_prediction))])
print(total_correct / len(y_train_prediction))

#now we will calculate the dunn index
#https://stackoverflow.com/questions/43784903/scikit-k-means-clustering-performance-measure
#we need a list of lists of data values
# k_list = []
# for i in range(cluster_count):
#     index_cluster=np.where(cluster_labels==i)[0] # get indexes of points in cluster i
#     k_list.append(x_train[index_cluster])
# print(k_list)
# print("dunn index:")
# print(base.dunn(k_list))
