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
print("BlockId")
print(data_df['BlockId'].values)
print("")
print("EventSequence")
print(data_df['EventSequence'].values)
print("")
print("Label")
print(data_df['Label'].values)
print("")

train_ratio = 0.5

x_data = data_df['EventSequence'].values
y_data = data_df['Label'].values
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
print("pos_idx")
print(pos_idx)
print("~pos_idx")
print(~pos_idx)
print("x_pos")
print(x_pos)
print("y_pos")
print(y_pos)
print("-----------")
print("x_train")
print(x_train)
print("x_test")
print(x_test)