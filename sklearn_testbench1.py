#!/usr/bin/env python
# -*- coding: utf-8 -*-

#

import sys
sys.path.append('./loglizer')
from sklearn.decomposition import PCA
import time
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# 0) example data frame
data_df = pd.DataFrame(columns=['BlockId', 'EventSequence','Label'])
data_df.loc[0]=["a",["e1","e4","e6","e6"],1]
data_df.loc[1]=["b",["e2","e2","e1","e6","e1","e1"],0]
data_df.loc[2]=["c",["e3","e3","e4"],1]
data_df.loc[3]=["d",["e1","e1","e1","e6","e4"],1]
data_df.loc[4]=["e",["e4","e1"],0]
print("data_df")
print(data_df)
print("")

# 0) example array
train_ratio=0.5
split_type='sequential'
def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
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
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
            data_df['Label'].values, train_ratio, split_type)
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

# 1) split using sss sklearn
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(data_df['EventSequence'].values, data_df['Label'].values):
    print("TRAIN:", train_index, "TEST:", test_index)
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

# 2) split using sklearn
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
train_index, test_index = next(sss.split(data_df['EventSequence'].values, data_df['Label'].values))
print("TRAIN:", train_index, "TEST:", test_index)
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

# 1) turn labeled data into numeric data
le = LabelEncoder()
data_df["EventSequence"] = data_df["EventSequence"].apply(le.fit_transform)
print("after transform")
print(data_df)

# 2) turn labeled data into 

