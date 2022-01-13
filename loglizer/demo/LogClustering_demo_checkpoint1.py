#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer.models import LogClusteringOriginal
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
        (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                    label_file=label_file,
                                                                    window='session', 
                                                                    train_ratio=0.5,
                                                                    split_type='uniform')
        #hadoop loader
        # print("hadoop")
        # (x_train, y_train), (x_test, y_test) = dataloader_hadoop.load_Hadoop(
        #     log_dir="../data/to_process_output",
        #     label_file="../data/to_process/abnormal_label.txt")
        print("@@shape of x_train:")
        print(x_train.shape)
        print("@@shape of x_test:")
        print(x_test.shape)
        #hdfs loader
        # (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS("../data/HDFS/HDFS_100k.log_structured.csv",
        # label_file="../data/HDFS/anomaly_label.csv",
        # window='session',
        # train_ratio=0.5,
        # split_type='uniform',
        # save_csv=False,
        # window_size=0)

        # print("debug")
        # print("=====*****=====")
        # print("before trnsform")
        # print("x_train:")
        # print(x_train)
        # print("x_test:")
        # print(x_test)
        # print("y_train:")
        # print(y_train)
        # print("y_test:")
        # print(y_test)
        debugLogFile.write("debug\n")
        debugLogFile.write("=====*****=====\n")
        debugLogFile.write("before trnsform\n")
        debugLogFile.write("x_train:\n")
        debugLogFile.write(str(x_train))
        debugLogFile.write("\nx_test:\n")
        debugLogFile.write(str(x_test))
        debugLogFile.write("\ny_train:\n")
        debugLogFile.write(str(y_train))
        debugLogFile.write("\ny_test:\n")
        debugLogFile.write(str(y_test))

        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)

        debugLogFile.write("\n\ndebug\n")
        debugLogFile.write("=====*****=====\n")
        debugLogFile.write("x_train after fit transform:\n")
        debugLogFile.write(str(x_train))
        debugLogFile.write("y_train after fit\n")
        debugLogFile.write(str(y_train))
        debugLogFile.write("\nx_test after fit transform:\n")
        debugLogFile.write(str(x_test))
        debugLogFile.write("y_test after fit\n")
        debugLogFile.write(str(y_test))
        #exit(111)
        #record precision, recall, and f1 for these variations:
        #pca, lsi, and nmf
        #dimensions (1 to num dimensions in x)
        #linkages (single, complete, average, weighted)
        # 
        precision_for_run_matrix = []

        # for linkage in range(0,4):
        #     for mode in range(0,4):
        #         for d in range(6,len(x_train[0])):
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
        #             "dataset="+str(datasetIndex)+" d="+str(d)+" mode="+str(mode)+" linkage="+str(linkage)+" : "+
        #             "precision="+str(precision)+" recall="+str(recall)+" f1="+str(f1)+"\n")
        start = time.time()
        print("shape of x_train:")
        print(x_train.shape[0],x_train.shape[1])
        print("shape of x_test:")
        print(x_test.shape[0],x_test.shape[1])
        model = LogClusteringOriginal(
            max_dist=max_dist,
            anomaly_threshold=anomaly_threshold,
            dimensionality=min(x_train[y_train == 0, :].shape[1],x_train[y_train == 0, :].shape[0]),
            #dimensionality=min(x_train.shape[1],x_train.shape[0]),
            reductionMode=0,
            linkageMode=0,
            max_cluster = max_cluster)
        print("shape of x_train input:")
        print(x_train[y_train == 0, :].shape[0],x_train[y_train == 0, :].shape[1])
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

        record_file.write("next\n")
        record_file.write(
        "dataset="+str(datasetIndex)+" d="+str(d)+" mode="+str(mode)+" linkage="+str(linkage)+" : "+
        "precision="+str(precision)+" recall="+str(recall)+" f1="+str(f1)+"\n")
    record_file.close()

