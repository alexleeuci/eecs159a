#!/usr/bin/env python
# -*- coding: utf-8 -*-

# notes on this test:
# instead of using the clustering algo given, I am trying new, simple cluster algos
# questions: best cluster_count
# https://stackoverflow.com/questions/65991074/how-to-find-most-optimal-number-of-clusters-with-k-means-clustering-in-python
# 

import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer.models import LogClusteringUnmodified
from loglizer.models import LogClusteringMulticlass, LogClustering_statsonclusters

from loglizer import dataloader,dataloader_hadoop, preprocessing

struct_log = "../data/to_process_output" # The structured log file
label_file = "../data/to_process/abnormal_label.txt" # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection

if __name__ == '__main__':
    print("-----------------------------TEST BEGIN-----------------------------")
    for anomaly_test in range(3,4):
        anomaly_threshold = anomaly_test*0.1
        (x_train, y_train), (x_test, y_test) = dataloader_hadoop.load_Hadoop(struct_log,
                                                                    label_file=label_file,
                                                                    window='session', 
                                                                    train_ratio=0.5,
                                                                    split_type='uniform')
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)

        from sklearn.cluster import KMeans
        import numpy as np
        cluster_count = 7
        kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(x_train)
        pred_idx = kmeans.predict(x_test)
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
        for i in range(cluster_count):
            index_cluster=np.where(cluster_labels==i)[0] # get indexes of points in cluster i
            classification_list = y_train[index_cluster] # get classification of each point in cluster i
            print("=================",i,"=================")
            print(classification_list)
            print("")

        #now we have the clusters x_test supposedly belings to:
        #since we have y_train and y_test, we can test the prediction's veracity
        print(y_test)
        print(centers)
        print(pred_idx)

        # model = LogClustering_statsonclusters(max_dist=max_dist, anomaly_threshold=anomaly_threshold, mode="online")
        # model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        # print('Train validation:')
        # precision, recall, f1 = model.evaluate(x_train, y_train)
        
        # print('Test validation:')
        # precision, recall, f1 = model.evaluate(x_test, y_test)