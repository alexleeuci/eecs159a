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

# 1) split using sklearn
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

# 2) convert into numeric data
le = LabelEncoder()
fit_transform_vec = np.vectorize(le.fit_transform, otypes=[list])
x_train = fit_transform_vec(x_train)
x_test = fit_transform_vec(x_test)
print("xtrain fit")
print(x_train)
print("xtest fit")
print(x_test)

print(x_train[y_train == 0, :])
#

