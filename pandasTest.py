import pandas as pd
from collections import Counter
import numpy as np
#1) this test shows how to load a column (useful for label)
#thanks
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
#https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html
myframe = pd.DataFrame(columns=['col1','col2'])
myframe.loc[0]=['a',1]
myframe.loc[1]=['b',2]
myframe.loc[2]=['a',3]
print(myframe)

# #define the dict to convert col1 into col3
# myDict = {'a':4,'b':5,'c':6}
# #add the new row
# newRow = []
# for index,row in myframe.iterrows():
#     print("=====")
#     print(row['col1'],row['col2'])
#     newRow.append(myDict[row['col1']])
# myframe.insert(myframe.shape[1], "col3", newRow, True)
# print(myframe)

X_seq = pd.DataFrame(columns=['col1','col2'])
X_seq.loc[0]=['a',['e1','e1','e2','e2','e1']]
X_seq.loc[1]=['b',['e3','e3','e1','e1','e3']]
X_seq.loc[2]=['a',['e5','e5','e3','e2','e2']]
X_seq = [list(['e1','e1','e2','e2','e1']),
 list(['e3','e3','e1','e1','e3'])
]
X_seq = np.array(X_seq)
print("=====")
print(X_seq)
print("=====")

X_counts = []
for i in range(X_seq.shape[0]):
    event_counts = Counter(X_seq[i])
    X_counts.append(event_counts)
print("X_counts:",X_counts)
X_df = pd.DataFrame(X_counts)
X_df = X_df.fillna(0)
events = X_df.columns
print("events",events)
X = X_df.values
print("X",X)
#demonstrate processing step
num_instance, num_event = X.shape
#if...
df_vec = np.sum(X > 0, axis=0)
print("df_vec",df_vec)
idf_vec = np.log(num_instance / (df_vec + 1e-8))
idf_matrix = X * np.tile(idf_vec, (num_instance, 1))
print("idf_matrix",idf_matrix)
X = idf_matrix