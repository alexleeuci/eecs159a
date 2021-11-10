from numpy.random import seed
from numpy.random import randn
from numpy import ndarray
from scipy.stats import friedmanchisquare
import statistics

# compare samples
import numpy as np
import pandas as pd
# seed the random number generator
seed(1)
#generate three independent samples
# data1 = 5 * randn(100) + 50
# print(data1, type(data1))
# data2 = 5 * randn(100) + 50
# data3 = 5 * randn(100) + 52

f= open("testArray")

treatment_count = 4
dataset_count = 4
precision_data = [[[] for i in range(treatment_count)] for j in range(dataset_count)]
treatment_dict = {}
dataset_dict = {}
current_treatment_index=-1
current_dataset_index=-1
for line in f:
    if not "next" in line:
        x = line.index("dataset=")+8
        dataset = int(line[x:line.find(" ",x)])
        x = line.index("mode=")+5
        mode = int(line[x:line.find(" ",x)])
        x = line.index("linkage=")+8
        linkage = int(line[x:line.find(" ",x)])
        x = line.index("precision=")+10
        precision = float(line[x:line.find(" ",x)])
        x = line.index("recall=")+7
        recall = float(line[x:line.find(" ",x)])
        x = line.index("f1=")+3
        f1 = float(line[x:line.find(" ",x)])
        treatment = tuple([mode,linkage])
        if not treatment in treatment_dict:
            print(treatment_dict,treatment)
            current_treatment_index+=1
            treatment_dict[treatment]=current_treatment_index
            treatment_index=current_treatment_index
        else:
            treatment_index=treatment_dict[treatment]
        if not dataset in dataset_dict:
            current_dataset_index+=1
            dataset_dict[dataset]=current_dataset_index
            dataset_index=current_dataset_index
        else:
            dataset_index=dataset_dict[dataset]
        print(dataset_index,treatment_index)
        precision_data[dataset_index][treatment_index].append(precision)
print(precision_data)
for i in range(len(precision_data)):
    for j in range(len(precision_data[0])):
        precision_data[i][j]=statistics.mean(precision_data[i][j])
print(precision_data)





#create pandas DataFrame
df = pd.DataFrame()

stat, p = friedmanchisquare(*[row for index, row in df.iterrows()])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')