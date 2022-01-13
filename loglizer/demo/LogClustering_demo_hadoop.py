#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LogClusteringMulticlass
from loglizer import dataloader, dataloader_hadoop, preprocessing
from sklearn.decomposition import PCA
import time

struct_log_list = ['../data/HDFS/HDFS_100k.log_structured.csv'] # The structured log file
label_file_list = ['../data/HDFS/anomaly_label.csv'] # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
max_cluster = 7 #needed for many classes hadoop
anomaly_threshold = 0.3 # the threshold for anomaly detection

record_log = "./logClustering_results2"
record_file = open(record_log,"w+")

# record_file.write("1234")
# exit(0)

debugLogFile = open("debugLogFile","w+")

if __name__ == '__main__':
    for datasetIndex in range(len(struct_log_list)):
        struct_log, label_file = struct_log_list[datasetIndex], label_file_list[datasetIndex]
        (x_train, y_train), (x_test, y_test) = dataloader_hadoop.load_Hadoop(
            log_dir="../data/to_process_output",
            label_file="../data/to_process/abnormal_label.txt")
        precision_for_run_matrix = []
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)
        #vectorizer doesnt work here...
        # feature_extractor = preprocessing.Vectorizer()
        # x_train = feature_extractor.fit_transform(x_train)
        # x_test = feature_extractor.transform(x_test)
        #print(x_train[y_train == 0, :])

        #clustering search
        # maxDimensions = min(x_train[y_train == 0, :].shape[1],x_train[y_train == 0, :].shape[0])
        # for linkage in range(0,4):
        #     for mode in range(0,4):
        #         for d in range(max(0,maxDimensions-10),maxDimensions):
        #             start = time.time()
        #             model = LogClustering(
        #                 max_dist=max_dist,
        #                 anomaly_threshold=anomaly_threshold,
        #                 dimensionality=d,
        #                 reductionMode=mode,
        #                 linkageMode=linkage)
        #             model.fit(x_train[y_train == 0, :]) # Use only normal samples for training
        #             end = time.time()
        #             timeElapsed = end-start

        #             print('Train validation:')
        #             precision, recall, f1 = model.evaluate(x_train, y_train)
                    
        #             print('Test validation:')
        #             precision, recall, f1 = model.evaluate(x_test, y_test)

        #             precision_for_run_matrix.append(tuple([precision,recall,f1]))

        #             record_file.write("next\n")
        #             record_file.write(
        #             "timeElapsed="+str(timeElapsed)+" dataset="+str(datasetIndex)+" d="+str(d)+" mode="+str(mode)+" linkage="+str(linkage)+" : "+
        #             "precision="+str(precision)+" recall="+str(recall)+" f1="+str(f1)+"\n")

        # 2) clustering once
        start = time.time()
        print("dimensionality:",min(x_train[y_train == 0, :].shape[1],x_train[y_train == 0, :].shape[0]))
        model = LogClusteringMulticlass(
            max_dist=max_dist,
            anomaly_threshold=anomaly_threshold,
            mode="offline",
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

