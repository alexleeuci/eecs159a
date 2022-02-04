#!/usr/bin/env python
# -*- coding: utf-8 -*-

# notes on this test:
# 1) instead of using the clustering algo given, I am trying new, simple cluster algos
# questions: best cluster_count
# https://stackoverflow.com/questions/65991074/how-to-find-most-optimal-number-of-clusters-with-k-means-clustering-in-python
# 2) I added resampling (oversampling) to increase the number of error data points

import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer.models import LogClusteringUnmodified
from loglizer.models import LogClusteringMulticlass, LogClustering_statsonclusters
from loglizer import dataloader,dataloader_hadoop, preprocessing
from sklearn.utils import resample,shuffle
from statistics import mean
import numpy as np
import os

#quick note for users: I wrote this on a vm and the imports are all weird
#don't modify the file structre please
sys.path.append("/mnt/c/Users/alexl/Downloads/jqm_cvi-master/jqm_cvi-master")
print(sys.path)
from jqmcvi import base

struct_log = "../data/to_process_output" # The structured log file
label_file = "../data/to_process/abnormal_label.txt" # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.1 # the threshold for anomaly detection

if __name__ == '__main__':
    print("-----------------------------TEST BEGIN-----------------------------")
    for anomaly_test in range(3,4):

        cluster_to_average = dict()
        anomaly_threshold = anomaly_test*0.1
        (x_train, y_train), (x_test, y_test) = dataloader_hadoop.load_Hadoop(struct_log,
                                                                    label_file=label_file,
                                                                    window='session', 
                                                                    train_ratio=0.5,
                                                                    split_type='uniform')
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)
        print(type(x_train))
        print(type(x_train[0]))

        #upsample y_train/y_test data points where classification = 0
        print(x_train)
        print(y_train)
        zero_loc = y_train==0
        one_loc = y_train==1
        y_train_0 = y_train[zero_loc]
        x_train_0 = x_train[zero_loc]
        y_train_1 = y_train[one_loc]
        x_train_1 = x_train[one_loc]
        y_train_0, x_train_0 = resample(y_train_0, x_train_0, n_samples = len(y_train_0)*5)
        #import pdb; pdb.set_trace()
        x_train = np.vstack([x_train_0, x_train_1])
        y_train = np.append(y_train_0, y_train_1)
        print(x_train)
        print(y_train)


        #upsample y_train/y_test data points where classification = 0
        print(x_test)
        print(y_test)
        zero_loc = y_test==0
        one_loc = y_test==1
        y_test_0 = y_test[zero_loc]
        x_test_0 = x_test[zero_loc]
        y_test_1 = y_test[one_loc]
        x_test_1 = x_test[one_loc]
        y_test_0, x_test_0 = resample(y_test_0, x_test_0, n_samples = len(y_test_0)*5)
        x_test = np.vstack([x_test_0, x_test_1])
        y_test = np.append(y_test_0, y_test_1)
        print(x_test)
        print(y_test)

        #combine train and test into one dataset
        x_values = np.vstack([x_test, x_train])
        y_values = np.append(y_test, y_train)

        from sklearn.cluster import DBSCAN
        import numpy as np
        #the bigger eps is, the fewer "-1" we have
        eps = 80
        dbscan = DBSCAN(eps=eps).fit(x_values)

        # we've gotten some clusters; now lets get the indicies of the data points in each cluster
        # and then find the ave class of each cluster
        #https://stackoverflow.com/questions/62626305/how-could-i-find-the-indexes-of-data-in-each-cluster-after-applying-the-pca-and
        cluster_labels=dbscan.labels_ # get cluster label of all data
        print("cluster labels of points:", list(cluster_labels))
        cluster_count= len(cluster_labels)
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
        for i in range(max(cluster_labels)):
            index_cluster=np.where(cluster_labels==i)[0] # get indexes of points in cluster i
            classification_list = y_values[index_cluster] # get classification of each point in cluster i
            print("=================",i,"=================")
            print(classification_list)
            cluster_to_average[i] = 1 if classification_list.mean()>zero_one_threshold else 0
            print("")
        print("cluster to average:")
        print(list(cluster_to_average))


        # fit using the dict above
        y_prediction = np.array([cluster_to_average[c] if c in cluster_to_average else -1 for c in cluster_labels])
        print("prediction based on dbscan")
        print(list(y_prediction))
        print("actual values")
        print(list(y_values))

        print("accuracy")
        total_correct = sum([1 if y_prediction[i]==y_values[i] else 0 for i in range(len(y_prediction))])
        #remove noise from prediciton
        print(total_correct / (len(y_prediction) - (y_prediction==-1).sum()))

