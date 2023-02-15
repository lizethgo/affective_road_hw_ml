# -*- coding: utf-8 -*-
"""
###############################################################################
###############################################################################

Date        : 
Author      : Lizeth Gonzalez Carabarin (@author: 888254)
File        : models.py
Company     : University of Applied Sciences Fontys
Description : This file contains a different models in the form of supervised and
              unsupervosed learning for classification and clustering.

###############################################################################
###############################################################################

"""


import numpy as np
import matplotlib.pyplot as plt
from functions import *
import matplotlib.colors as colors
from scipy import signal
import math
from read_data import * 


length = 35813 #len(dataset_eda)
seg_time = 20 # input from user 
# TODO : To determine an optimal window of time 
batch_size = int(seg_time*fs) 
num_cases = 13 #based on dataset


path = './Database/E4/{}-E4-Drv{}/Left/EDA.csv'
x_arr_scl, x_arr_scr = filtered_data(path=path, num_cases=13)

x_avg_scl, x_max_scl, x_min_scl, x_var_scl, x_kur_scl, x_avg_scr, x_max_scr, x_min_scr, x_var_scr, x_kur_scr =  time_features(num_cases, x_arr_scl, x_arr_scr, shuffle = True)    

Sxx = freq_features(data=x_arr_scr, num_cases=num_cases, plot=True)

## array - list - array
## TODO: How to convert it to an array completely?
X_train = np.array([x_avg_scl, x_max_scl, x_min_scl, x_var_scl, x_kur_scl, x_avg_scr, x_max_scr, x_min_scr, x_var_scr, x_kur_scr ])

## reading annotations



###############################################################################
###############################################################################
#### ML models
#### unsupervised approach - this shows how to cluster different data 
centers = []
for i in range(0,num_cases-2):
    X_train_1 = np.concatenate((np.reshape(x_avg_scl[i], (len(x_avg_scl[i]),1)), np.reshape(x_max_scl[i], (len(x_max_scl[i]),1)) ), axis=1)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train_1)
    color = ['b' if i > 0.5 else 'r' for i in kmeans.labels_] 
    plt.scatter(X_train_1[:,0], X_train_1[:,1], s=2, c=color)
    plt.show()
    print(kmeans.cluster_centers_)
    centers.append(kmeans.cluster_centers_)


plt.scatter(centers)

for i in range(0,num_cases-2):
    plt.scatter(centers[i][0,0],centers[i][0,1])
    plt.scatter(centers[i][1,0],centers[i][1,1])
    
###############################################################################
###############################################################################
#### ML models
#### supervised _train[0,:]
#### 
from sklearn import svm

### testing for patience 1
### separating training and testing dataset
#X_aux = X_train-> # features,data
X_train_p1 = np.zeros([10,int(0.75*len( X_train[0,0][:]))])
X_test_p1 =  np.zeros([10,int(len( X_train[0,0][:])-0.75*len( X_train[0,0][:]))])

for i in range(10):
    X_train_p1[i,:] = X_train[i,0][0:int(0.75*len( X_train[0,0][:]))]
    X_test_p1[i,:] =  X_train[i,0][0:int(len( X_train[0,0][:])-0.75*len( X_train[0,0][:]))]

h1=svm.SVR()
h2=svm.LinearSVR()


