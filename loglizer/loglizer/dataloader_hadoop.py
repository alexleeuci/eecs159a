"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

"""
these variables should be set to something to reflect what you want to do:
1) labelVersion = <0,1>
0: 8 different labels, based on event AND error type
1: 2 different labels, based on error or not error

2) vecLenAgnostic = <true, false>
true: do not filter out very long data points
false: filter out data points longer than 73 dimensions
"""
labelVersion=1
vecLenAgnostic = False

import pandas as pd
import os
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict
from sklearn.decomposition import PCA
from os import walk

def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    #import pdb; pdb.set_trace()
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx] 
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)

debugFileHadoopLoader = open("debugFileHadoopLoader","w+")

def load_Hadoop(log_dir=None,
label_file=None,
window='session',
train_ratio=0.5,
split_type='uniform',
save_csv=False,
window_size=0):
    filesDir = log_dir
    label_file_txt = "../data/to_process/abnormal_label.txt"
    print('====== Input data summary ======')


    #exit(111)

    data_df = pd.DataFrame(columns=['BlockId', 'EventSequence'])
    for (dirpath, dirnames, filenames) in walk(filesDir):
        #print(filenames)
        for filename in filenames:
            log_file = filename
            #print("dirpath:",dirpath)
            #print("filename:",filename)
            if "abnormal_label" not in filename and "BGL" not in filename:    
                if log_file.endswith('.csv'):
                    assert window == 'session', "Only window=session is supported for HDFS dataset."
                    #print("===============")
                    #print("Loading", dirpath+"/"+log_file)
                    struct_log = pd.read_csv(dirpath+"/"+log_file, engine='c',
                            na_filter=False, memory_map=True)
                    #print("debug1")
                    data_dict = OrderedDict()

                    # after this, data_dict must have a row 'EventId' for each identifying block
                    # 1) for hadoop, each identifying block [testtype_normal/md/nd/df] must have a list
                    # 2) so, when we process application_1445087491445_0005, we need to add the list
                    #       data_dict[application_...]=[[EventId1, EventId2,... EventIdN]]
                    #       data_dict[application_...]=[[EventId1, EventId2,... EventIdN]]
                    labelDirKey = dirpath[dirpath.rindex("/")+1:]
                    for idx, row in struct_log.iterrows():
                        if not labelDirKey in data_dict:
                            data_dict[labelDirKey] = []
                        data_dict[labelDirKey].append(row['EventId'])
                    for key in data_dict:
                        #print(data_dict[key])
                        if len(data_dict[key])<=73 or not vecLenAgnostic:
                            #print("lens:",len(data_dict[key]))
                            data_df.loc[len(data_df.index)]=[key,data_dict[key]]
    debugFileHadoopLoader.write("@@@")
    for idx,row in data_df.iterrows():
        debugFileHadoopLoader.write(str(idx)+" : "+str(row['EventSequence'])+"len="+str(len(row['EventSequence']))+"\n")
    #the dataframe format of data_df must be:
    #col:   blockId     eventSequence
    #idx1:  <id=1>      <e1,e4,e3...e_n>
    #idx2:  <id=2>      <e3,e1,e1...e_n>
    #...such that <id=1> must represent a block of log events that are labeled one thing
    #...such that e_n must represent a type of log event


    if label_file:
        # Split training and validation set in a class-uniform way
        # 1) label_dict must be a dict, with (key=BlockId:item=classification)
        # 2) data_df must have a 'Label' row, with the classification
        # label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
        # label_data = label_data.set_index('BlockId')
        # label_dict = label_data['Label'].to_dict()
        # data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
        # 1)
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
                #print(label_from_file,currentBlock)
                if labelVersion==0:
                    label_dict[label_from_file]=currentBlock
                else:
                    label_dict[label_from_file]=(0 if "Normal" in currentBlock else 1)
        #for key in label_dict:
            #print(key,":",label_dict[key])
        # 2)
        #define the dict to convert col1 into col3
        #add the new row
        newRow = []
        label_dict_to_int = {}
        label_next_int = -1
        for index,row in data_df.iterrows():
            #print(row['BlockId'])
            if not label_dict[row['BlockId']] in label_dict_to_int:
                label_next_int+=1
                label_dict_to_int[label_dict[row['BlockId']]]=label_next_int
            newRow.append(label_dict_to_int[label_dict[row['BlockId']]])
        data_df.insert(data_df.shape[1], "Label", newRow, True)
        #print(data_df)


        #print("===== datadf")
        #print(data_df)

        # Split train and test data
        (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
            data_df['Label'].values, train_ratio, split_type)

    if save_csv:
        data_df.to_csv('data_instances.csv', index=False)

#     if window_size > 0:
#         x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
#         x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
#         log = "{} {} windows ({}/{} anomaly), {}/{} normal"
#         print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
#         print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
#         return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

#     if label_file is None:
#         if split_type == 'uniform':
#             split_type = 'sequential'
#             print('Warning: Only split_type=sequential is supported \
#             if label_file=None.'.format(split_type))
#         # Split training and validation set sequentially
#         x_data = data_df['EventSequence'].values
#         (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
#         print('Total: {} instances, train: {} instances, test: {} instances'.format(
#             x_data.shape[0], x_train.shape[0], x_test.shape[0]))
#         return (x_train, None), (x_test, None), data_df
# else:
#     raise NotImplementedError('load_HDFS() only support csv and npz files!')

    # for i in y_test:
    #     print(i)
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))
    return (x_train, y_train), (x_test, y_test)

# def slice_hdfs(x, y, window_size):
#     results_data = []
#     print("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
#     for idx, sequence in enumerate(x):
#         seqlen = len(sequence)
#         i = 0
#         while (i + window_size) < seqlen:
#             slice = sequence[i: i + window_size]
#             results_data.append([idx, slice, sequence[i + window_size], y[idx]])
#             i += 1
#         else:
#             slice = sequence[i: i + window_size]
#             slice += ["#Pad"] * (window_size - len(slice))
#             results_data.append([idx, slice, "#Pad", y[idx]])
#     results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
#     print("Slicing done, {} windows generated".format(results_df.shape[0]))
#     return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]

debug=0
if debug:
    (x_train, y_train), (x_test, y_test) = load_Hadoop(
        log_dir="../data/to_process_output",
        label_file="../data/to_process/abnormal_label.txt")
