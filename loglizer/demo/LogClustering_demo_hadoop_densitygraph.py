#!/usr/bin/env python
# -*- coding: utf-8 -*-

# notes on this test:
# I'm getting an n^2 matrix of distances, so that I can view the distribution of distances
# This could give insight into if clustering is even possible

import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer.models import LogClusteringUnmodified
from loglizer.models import LogClusteringMulticlass, LogClustering_statsonclusters
from loglizer import dataloader,dataloader_hadoop, preprocessing

from statistics import mean
import os
import numpy as np
from numpy import linalg as LA
def _distance_metric(x1, x2):
    norm= LA.norm(x1) * LA.norm(x2)
    distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
    if distance < 1e-8:
        distance = 0
    return distance
#quick note for users: I wrote this on a vm and the imports are all weird
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

        # 1) get a matrix of train pair-wise distances
        distance_matrix = np.zeros((len(x_train)+len(x_test),len(x_train)+len(x_test)), dtype=float)
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                distance_matrix[i][j] = _distance_metric(x_train[i], x_train[j])
                print(distance_matrix[i][j])
        #print(distance_matrix)
        with open('outfile.txt','w+') as f:
            for line in distance_matrix:
                np.savetxt(f, line, fmt='%.2f')
