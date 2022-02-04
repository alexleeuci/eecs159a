import numpy as np
from sklearn.utils import resample

def rate_cluster(clusters, classes, cluster_count, online_clusters=[], online_classes=[]):
    '''
    clusters = list where cluster[i] is the cluster data point i belongs to
        note: cluster names must be numbers, and they must be sequential
        eg, if there are 5 clusters, the clusters must be labeled 0, 1, 2, 3, 4.
    classes = list where classes[i] is the true classification data point i belongs to
    cluster_count = number of clusters found
    online_clusters = list of list of clusters in the same format as clusters.
        these clusters are optional and test how well the clustering algorithm
        finds patterns with new data after clustering happens
    online_classes = list of list of classes
        these are the actual classifications of each of the online clusters

    returns a list of "scores" that tell us how well each cluster discriminated
    against the known classifications, for each cluster given.
    '''
    scores = []
    cluster_to_average = dict()
    zero_one_threshold = 0.5
    for i in range(cluster_count):
        index_cluster=np.where(clusters==i)[0] # get indexes of points in cluster i
        classification_list = classes[index_cluster] # get classification of each point in cluster i
        print("=================",i,"=================")
        print(classification_list)
        cluster_to_average[i] = 1 if classification_list.mean()>zero_one_threshold else 0
        print(cluster_to_average[i])
        print("")
    print("cluster to average:")
    print(cluster_to_average)

    average_classes = np.array([cluster_to_average[c] for c in clusters])
    print("clusters")
    print(clusters)
    print("average classes of clusters")
    print(average_classes)
    print("actual values")
    print(classes)

    # print("accuracy")
    total_correct = sum([1 if average_classes[i]==classes[i] else 0 for i in range(len(average_classes))])
    score = total_correct / len(average_classes)
    # print(score)
    scores.append(score)

    for index in range(len(online_classes)):
        average_classes = np.array([cluster_to_average[c] for c in online_clusters[index]])
        # print("classifications")
        # print(classes)
        # print("average classes of clusters")
        # print(average_classes)
        # print("actual values")
        # print(classes)

        # print("accuracy")
        total_correct = sum([1 if average_classes[i]==online_classes[index][i] else 0 for i in range(len(average_classes))])
        score = total_correct / len(average_classes)
        # print(score)
        scores.append(score)

    return scores

def upsample_class(x_values, y_values, upsample_class, ratio=5):
    '''
    x_values = data points
    y_values = classifications of each data point
    upsample_class = replicate classes where y_value class = upsample_class
    ratio = how many times d you want to replicate the upsampled classes?

    returns x_values and y_values with replicated data points
    '''
    #upsample y_train/y_test data points where classification = 0
    # print(x_train)
    # print(y_train)
    zero_loc = y_values==0
    one_loc = y_values==1
    y_value_0 = y_values[zero_loc]
    x_value_0 = x_values[zero_loc]
    y_value_1 = y_values[one_loc]
    x_value_1 = x_values[one_loc]
    y_value_0, x_value_0 = resample(y_value_0, x_value_0, n_samples = len(y_value_0)*ratio)
    #import pdb; pdb.set_trace()
    x_value = np.vstack([x_value_0, x_value_1])
    y_value = np.append(y_value_0, y_value_1)
    print(x_value)
    print(y_value)
    return x_value,y_value