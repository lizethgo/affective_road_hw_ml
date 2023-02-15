# -*- coding: utf-8 -*-
"""
###############################################################################
###############################################################################

Date        : 
Author      : Lizeth Gonzalez Carabarin (@author: 888254)
File        : funtions.py
Company     : University of Applied Sciences Fontys
Description : This file contains a script that defines some useful fucntions to 
              read a target dataset (cvs). Additional functions for filtering
              are also included.

              
###############################################################################
###############################################################################

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis


## defintion of functions and classes

def read_data(path):
    """This functions reads data from a cvs file and converts it into a numpy
       array """
    df_data = pd.read_csv(path)
    dataset = df_data.to_numpy()
    dataset = np.reshape(dataset, (np.shape(dataset)[0]))
    return dataset


def plotting(data, label_x, label_y, fs):
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111, )
    # Create x,y,z axis labels:
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    df = 1 / fs                         # Determine step
    fNQ = len(data) / fs                # Determine Nyquist frequency
    xaxis = np.arange(0,fNQ,df)         # Construct frequency axis
    ax.plot(xaxis, data, color='royalblue')
    plt.show()
    
def plotting_2G(data1, label1_x, label1_y, data2, label2_x, label2_y, fs):
    fig = plt.figure(figsize=(4, 2))
    fig, ax = plt.subplots(2)
    # Create x,y,z axis labels:
    ax[0].set_xlabel(label1_x)
    ax[0].set_ylabel(label1_y)
    ax[1].set_xlabel(label2_x)
    ax[1].set_ylabel(label2_y)
    ax[0].xaxis.grid(True)
    ax[0].yaxis.grid(True)
    ax[1].xaxis.grid(True)
    ax[1].yaxis.grid(True)
    df = 1 / fs                         # Determine step
    fNQ = len(data1) / fs                # 
    xaxis = np.arange(0,fNQ,df)         # 
    ax[0].plot(xaxis, data1, color='royalblue')
    fNQ = len(data2) / fs                # 
    xaxis = np.arange(0,fNQ,df)         # 
    ax[1].plot(data2, color='tomato')
    plt.show()
    
def plotting_3G(data1, label1_x, label1_y, data2, label2_x, label2_y, data3, label3_x, label3_y, fs):
    fig = plt.figure(figsize=(4, 2))
    fig, ax = plt.subplots(3)
    # Create x,y,z axis labels:
    ax[0].set_xlabel(label1_x)
    ax[0].set_ylabel(label1_y)
    ax[1].set_xlabel(label2_x)
    ax[1].set_ylabel(label2_y)
    ax[2].set_xlabel(label3_x)
    ax[2].set_ylabel(label3_y)
    ax[0].xaxis.grid(True)
    ax[0].yaxis.grid(True)
    ax[1].xaxis.grid(True)
    ax[1].yaxis.grid(True)
    ax[2].xaxis.grid(True)
    ax[2].yaxis.grid(True)
    df = 1 / fs                         # Determine step
    fNQ = len(data1) / fs                # 
    xaxis = np.arange(0,fNQ,df)         # 
    ax[0].plot(xaxis, data1, color='royalblue')
    fNQ = len(data2) / fs                # 
    xaxis = np.arange(0,fNQ,df)         # 
    ax[1].plot(data2, color='tomato')
    ax[2].plot(data3, color='green')
    plt.show()

## Filtering signals

from scipy.signal import butter, lfilter, freqz, filtfilt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def unpooling_1D(data, strides):
    return np.repeat(data, strides)

    
def normalization(data):
    return (data + abs(np.min(data)))/np.max(data + abs(np.min(data)))


def filtered_data(path, num_cases, cutoff=0.002, norm=True):
    x_arr_scr = []
    x_arr_scl = []

    #path='./Database/E4/{}-E4-Drv{}/Left/EDA.csv'

    for i in range(0, num_cases-1):
         pathd = path.format(i+1,i+1)
         print('path : ', pathd)
         x_arr=read_data(path=path.format(i+1,i+1))
         """öbtaining """
         if norm:
             x_arr_scl.append(normalization(butter_lowpass_filter(x_arr[2:], cutoff, fs, order)))
             x_arr_scr.append(normalization(butter_highpass_filter(x_arr[2:], cutoff, fs, order)))
         else:
             x_arr_scl.append(butter_lowpass_filter(x_arr[2:], cutoff, fs, order))
             x_arr_scr.append(butter_highpass_filter(x_arr[2:], cutoff, fs, order))
             
    return x_arr_scl, x_arr_scr

### testing functions  
path_eda = './Database/E4/1-E4-Drv1/Left/EDA.csv'  
path_hr = './Database/E4/1-E4-Drv1/Left/HR.csv'
dataset_eda = read_data(path_eda)
plotting(dataset_eda, label_x = 'time (s)', label_y = 'EDA', fs=4)


SR = np.shape(dataset_eda[2:])[0]/(3*3600)

# Testing filters
order = 3
fs = 4.0       # sample rate, Hz
cutoff = 0.002  # desired cutoff frequency of the filter ( Hz )
## TODO : establish cutoff frequency as a trainable parameter

# Filter the data, and plot both the original and filtered signals.
result_lowpass = butter_lowpass_filter(dataset_eda[2:], cutoff, fs, order)
plotting(result_lowpass, label_x = 'time (s)', label_y = 'EDA_SCL', fs=4)


result_highpass = butter_highpass_filter(dataset_eda[2:], cutoff, fs, order)
plotting(result_highpass, label_x = 'time (s)', label_y = 'EDA_SCR', fs=4)


def non_filtered_data(path, num_cases, norm=True):
    x_arr = []


    for i in range(0, num_cases-1):
         pathd = path.format(i+1,i+1)
         print('path : ', pathd)
         x_arr.append(read_data(path=path.format(i+1,i+1)))

            
    return x_arr





   