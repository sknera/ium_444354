import pandas as pd
import re
from sklearn import metrics
import numpy as np
import csv 
import matplotlib.pyplot as plt
f = open("pytorch/result.txt", "r")

list_result, list_predicted=[],[]
for x in f:
    data = x.split(' ')
    result =  re.findall(r'\d+', data[1])
    predicted  = re.findall(r'\d+', data[5])

    result=int(result[0])
    predicted=float('.'.join(predicted))

    list_result.append(result)
    list_predicted.append(predicted)
metrics = metrics.mean_absolute_error(list_result, list_predicted),  metrics.mean_squared_error(list_result, list_predicted),np.sqrt(metrics.mean_absolute_error(list_result, list_predicted))
print("MAE: ", metrics[0])
print("MSE: ",metrics[1])
print("RMSE: ",metrics[2])



with open('eval.csv', 'a+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow((metrics[0],metrics[1], metrics[2]))


MAE,MSE,RMSE=[],[],[]
with open('eval.csv', 'r') as r:
    for row in r:
        # row variable is a list that represents a row in csv
        row=row.split(',')
        MAE.append(float(row[0]))
        MSE.append(float(row[1]))        
        RMSE.append(float(row[2]))    

plt.xlabel('build')
plt.plot(np.arange(0, len(MAE)), MAE, label="MAE")
plt.plot(np.arange(0, len(MSE)), MSE, label="MSE") 
plt.plot(np.arange(0, len(RMSE)), RMSE, label="RMSE") 
plt.legend()
plt.savefig('metrics.png')