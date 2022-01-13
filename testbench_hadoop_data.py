#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('./loglizer')
from sklearn.decomposition import PCA
import time
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from os import walk
from collections import OrderedDict
#import clustering
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
import numpy as np

# 0) example data frame
data_df = pd.DataFrame(columns=['BlockId', 'EventSequence','Label'])
data_df.loc[0]=["a",["e1","e4","e6","e6"],1]
data_df.loc[1]=["b",["e2","e2","e1","e6","e1","e1"],0]
data_df.loc[2]=["c",["e3","e3","e4"],1]
data_df.loc[3]=["d",["e1","e1","e1","e6","e4"],1]
data_df.loc[4]=["e",["e4","e1"],0]

#0) load data frame for hadoop
log_dir="/mnt/c/Users/alexl/OneDrive/Desktop/ivv_tests/loglizer/data/to_process_output"
label_file_txt="/mnt/c/Users/alexl/OneDrive/Desktop/ivv_tests/loglizer/data/to_process/abnormal_label.txt"
vecLenAgnostic = False
labelVersion = 0
data_df = pd.DataFrame(columns=['BlockId', 'EventSequence'])
for (dirpath, dirnames, filenames) in walk(log_dir):
    for filename in filenames:
        log_file = filename
        if "abnormal_label" not in filename and "BGL" not in filename:    
            if log_file.endswith('.csv'):
                #assert window == 'session', "Only window=session is supported for HDFS dataset."
                struct_log = pd.read_csv(dirpath+"/"+log_file, engine='c',
                        na_filter=False, memory_map=True)
                data_dict = OrderedDict()
                labelDirKey = dirpath[dirpath.rindex("/")+1:]
                for idx, row in struct_log.iterrows():
                    if not labelDirKey in data_dict:
                        data_dict[labelDirKey] = []
                    data_dict[labelDirKey].append(row['EventId'])
                for key in data_dict:
                    if len(data_dict[key])<=73 or not vecLenAgnostic:
                        data_df.loc[len(data_df.index)]=[key,data_dict[key]]
# 0.2) load labels:
currentBlock = ""
application = ""
error = ""
label_dict = dict()
label_file = open(label_file_txt,"r")
for line in label_file:
    if "###" in line:
        application = line[4:-1]
    if ":" in line:
        error = line[:line.index(":")]
    currentBlock = application+"_"+error
    if "+" in line:
        label_from_file = line[line.index("+")+2:-1]
        if labelVersion==0:
            label_dict[label_from_file]=currentBlock
        else:
            label_dict[label_from_file]=(0 if "Normal" in currentBlock else 1)
newRow = []
label_dict_to_int = {}
label_next_int = -1
for index,row in data_df.iterrows():
    if not label_dict[row['BlockId']] in label_dict_to_int:
        label_next_int+=1
        label_dict_to_int[label_dict[row['BlockId']]]=label_next_int
    newRow.append(label_dict_to_int[label_dict[row['BlockId']]])
data_df.insert(data_df.shape[1], "Label", newRow, True)

# 1) split using sklearn
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(data_df['EventSequence'].values, data_df['Label'].values):
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data_df['EventSequence'].values[train_index], data_df['EventSequence'].values[test_index]
    y_train, y_test = data_df['Label'].values[train_index], data_df['Label'].values[test_index]
    print("test array")
    print("x_train:")
    print(x_train)
    print("y_train:")
    print(y_train)
    print("x_test:")
    print(x_test)
    print("y_test:")
    print(y_test)
    print("")

    # 2) convert into numeric data
    le = LabelEncoder()
    fit_transform_vec = np.vectorize(le.fit_transform, otypes=[list])
    x_train = fit_transform_vec(x_train)
    x_test = fit_transform_vec(x_test)
    # print("xtrain fit")
    # print(x_train)
    # print("xtest fit")
    # print(x_test)

    # 2b) standardization mechanism
    #SELECT ONE:
    # 2b1)
    # turn array of var len arrays into standard len arrays
    # https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array/43146354
    zeroArr = np.zeros([len(x_train),len(max(x_train,key = lambda x: len(x)))])
    for i,j in enumerate(x_train):
        zeroArr[i][0:len(j)] = j
    x_train = zeroArr
    # 2b2)
    # 

    # 3a) create pdist object and then apply fclustering
    Z = ward(pdist(x_train))
    #clusterList = fcluster(Z, t=500, criterion='distance')
    numclust = 2
    clusterList = fcluster(Z, numclust,criterion='maxclust')
    print(clusterList)
    print(y_test)
    # 3b) evaluate cluster based on input data

