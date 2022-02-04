#!/usr/bin/env python
# -*- coding: utf-8 -*-

# notes on this test:
# 1) this just takes the hadoop logs and loads them into
# classification : [count_of_event_1, count_of_event_2, count_of_event_3...]
# classification : [count_of_event_1, count_of_event_2, count_of_event_3...]
# classification : [count_of_event_1, count_of_event_2, count_of_event_3...]
# 2) please put any new method of loading data here. Loading data means handeling transforming
# parsed log data (eg, a csv table with one column for each log message)
# and a label file
# to
# whatever numeric representation of input data, and its numeric classifications

import sys
sys.path.append('../')
from loglizer import dataloader,dataloader_hadoop, preprocessing
from sklearn.utils import resample,shuffle
from statistics import mean
import numpy as np
import os

struct_log = "../data/to_process_output" # The structured log file
label_file = "../data/to_process/abnormal_label.txt" # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.1 # the threshold for anomaly detection

(x_train, y_train), (x_test, y_test) = dataloader_hadoop.load_Hadoop(struct_log,
                                                            label_file=label_file,
                                                            window='session', 
                                                            train_ratio=0.5,
                                                            split_type='uniform')
feature_extractor = preprocessing.FeatureExtractor()
#x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
x_train = feature_extractor.fit_transform(x_train)
x_test = feature_extractor.transform(x_test)

f = open("Hadoop_EventCount","w+")
for i in range(len(x_train)):
    f.write(str(y_train[i]))
    f.write("\t")
    for item in x_train[i]:
        f.write(str(item))
        f.write("\t")
    f.write("\n")
f.close()

        