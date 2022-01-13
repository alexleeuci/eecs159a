#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LogClusteringMulticlass
from loglizer import dataloader, dataloader_hadoop, preprocessing
from sklearn.decomposition import PCA
import time
from sklearn.preprocessing import LabelEncoder

max_dist = 0.3 # the threshold to stop the clustering process
max_cluster = 7 #needed for many classes hadoop
anomaly_threshold = 0.3 # the threshold for anomaly detection

record_log = "./logClustering_results2"
record_file = open(record_log,"w+")

# record_file.write("1234")
# exit(0)

debugLogFile = open("debugLogFile","w+")

(x_train, y_train), (x_test, y_test) = dataloader_hadoop.load_Hadoop(
    log_dir="../data/to_process_output",
    label_file="../data/to_process/abnormal_label.txt")

# 1) turn labeled data into numeric data
le = LabelEncoder()
print(type(x_train))
x_train["EventSequence"] = x_train["EventSequence"].apply(le.fit_transform)
x_test["EventSequence"] = x_test["EventSequence"].apply(le.fit_transform)

# 2) clustering once
start = time.time()
print("dimensionality:",min(x_train[y_train == 0, :].shape[1],x_train[y_train == 0, :].shape[0]))
model = LogClusteringMulticlass(
    max_dist=max_dist,
    anomaly_threshold=anomaly_threshold,
    dimensionality=min(x_train[y_train == 0, :].shape[1],x_train[y_train == 0, :].shape[0])-4,
    #dimensionality=min(x_train.shape[1],x_train.shape[0]),
    reductionMode=-1,
    linkageMode=0,
    max_cluster = max_cluster)
model.fit(x_train[y_train == 0, :]) # Use only normal samples for training
# print(x_train.shape[0],x_train.shape[1])
# model.fit(x_train) # Use only normal samples for training
end = time.time()
timeElapsed = end-start

print('Train validation:')
precision, recall, f1 = model.evaluate(x_train, y_train)

print('Test validation:')
precision, recall, f1 = model.evaluate(x_test, y_test)

precision_for_run_matrix.append(tuple([precision,recall,f1]))

record_file.close()

