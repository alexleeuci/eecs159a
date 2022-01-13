#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer import dataloader, preprocessing
from sklearn.decomposition import PCA
import time


struct_log_list = ['../data/HDFS/HDFS_100k.log_structured.csv'] # The structured log file
label_file_list = ['../data/HDFS/anomaly_label.csv'] # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection

record_log = "./logClustering_results2"
record_file = open(record_log,"w+")

# record_file.write("1234")
# exit(0)

if __name__ == '__main__':
    for datasetIndex in range(len(struct_log_list)):
        struct_log, label_file = struct_log_list[datasetIndex], label_file_list[datasetIndex]
        (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                    label_file=label_file,
                                                                    window='session', 
                                                                    train_ratio=0.5,
                                                                    split_type='uniform')



        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)


        #record precision, recall, and f1 for these variations:
        #pca, lsi, and nmf
        #dimensions (1 to num dimensions in x)
        #linkages (single, complete, average, weighted)
        # 
        precision_for_run_matrix = []

        for linkage in range(0,4):
            for mode in range(0,4):
                for d in range(6,len(x_train[0])):
                    start = time.time()
                    model = LogClustering(
                        max_dist=max_dist,
                        anomaly_threshold=anomaly_threshold,
                        dimensionality=d,
                        reductionMode=mode,
                        linkageMode=linkage)
                    model.fit(x_train[y_train == 0, :]) # Use only normal samples for training
                    end = time.time()
                    timeElapsed = start-end

                    print('Train validation:')
                    precision, recall, f1 = model.evaluate(x_train, y_train)
                    
                    print('Test validation:')
                    precision, recall, f1 = model.evaluate(x_test, y_test)

                    precision_for_run_matrix.append(tuple([precision,recall,f1]))

                    record_file.write("next\n")
                    record_file.write(
                    "timeElapsed="+str(timeElapsed)+" dataset="+str(datasetIndex)+" d="+str(d)+" mode="+str(mode)+" linkage="+str(linkage)+" : "+
                    "precision="+str(precision)+" recall="+str(recall)+" f1="+str(f1)+"\n")
    record_file.close()

