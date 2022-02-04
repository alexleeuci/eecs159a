#!/usr/bin/env python
# -*- coding: utf-8 -*-

# notes on this test:
# I'm getting an n^2 matrix of distances, so that I can view the distribution of distances
# This could give insight into if clustering is even possible

import matplotlib.pyplot as plt
import numpy as np

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
sys.path.append("/mnt/c/Users/alexl/Downloads/  /jqm_cvi-master")
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

        #combine x_train and x_test
        x_values = np.vstack([x_test, x_train])
        y_values = np.append(y_test, y_train)

        #split into y_value = 0 and y_value = 1
        zero_loc = y_values==0
        one_loc = y_values==1
        y_values_1 = y_values[one_loc]
        x_values_1 = x_values[one_loc]
        y_values_0 = y_values[zero_loc]
        x_values_0 = x_values[zero_loc]

        #add all the indicies in x_values
        x_values_1_distr = np.add.reduce(x_values_1)
        x_values_0_distr = np.add.reduce(x_values_0)

        #do a chi-squared analysis on the distribution of 1 vs 0 values
        table = [x_values_0_distr,x_values_1_distr]
        from scipy.stats import chi2_contingency
        from scipy.stats import chi2
        from sklearn.utils import shuffle
        from sklearn.model_selection import train_test_split
        stat, p, dof, expected = chi2_contingency(table)
        print("-----test vals-----")
        print(stat)
        print(p)
        print(dof)
        print(expected)
        print("-----")
        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

        #randomly distribute the vectors and re-do a chi-squared

        x_values = np.hstack(table)
        shuffle(x_values)
        x_values_split1, x_values_split2 = train_test_split(x_values)
        table = [x_values_split1,x_values_split2]
        from scipy.stats import chi2_contingency
        from scipy.stats import chi2
        from sklearn.utils import shuffle
        from sklearn.model_selection import train_test_split
        stat, p, dof, expected = chi2_contingency(table)
        print("-----test vals-----")
        print(stat)
        print(p)
        print(dof)
        print(expected)
        print("-----")
        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

        #graph these distributions
        import matplotlib.pyplot as plt
        import numpy as np
        plt.hist(x_values_1_distr, density=True, bins=len(x_values_1_distr))  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.savefig("distr1")

        plt.hist(x_values_0_distr, density=True, bins=len(x_values_1_distr))  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.savefig("distr0")