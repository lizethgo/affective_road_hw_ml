# -*- coding: utf-8 -*-
"""
###############################################################################
###############################################################################

Date        : 
Author      : Lizeth Gonzalez Carabarin (@author: 888254)
File        : read_data.py
Company     : University of Applied Sciences Fontys
Description : This file contains a script to read a target dataset (cvs), and
              to generate training data. It also contains functions for feature
              extraction.
               

###############################################################################
###############################################################################

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import matplotlib.colors as colors
from scipy import signal
import math
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from sklearn.utils import shuffle




############ Defining parameters #############################################

length = 35813 #len(dataset_eda)
seg_time = 20 # input from user 
# TODO : To determine an optimal window of time 

batch_size = int(seg_time*fs) 

num_cases = 13 #based on dataset

##############################################################################
### testing functions  
path_eda = './Database/E4/3-E4-Drv3/Left/EDA.csv'  
path_hr = './Database/E4/3-E4-Drv3/Left/HR.csv'
dataset_eda = read_data(path_eda)
plotting(dataset_eda, label_x = 'time (s)', label_y = 'EDA', fs=4)

dataset_hr = read_data(path_hr)
plotting(dataset_hr, label_x = 'time (s)', label_y = 'HR', fs=4)


## https://paulvangent6.rssing.com/chan-70055902/article1.html?zx=814
### Duration of recording is 3hr


# Testing functions 
order = 3
fs = 4.0       # sample rate, Hz
cutoff = 0.002  # desired cutoff frequency of the filter, Hzn 
## TODO : establish cutoff as a trainable parameter

# Filter the data, and plot both the original and filtered signals.
result_lowpass = butter_lowpass_filter(dataset_eda[2:], cutoff, fs, order)
#plotting(result_lowpass, label_x = 'time', label_y = 'EDA_SCL')


result_highpass = butter_highpass_filter(dataset_eda[2:], cutoff, fs, order)
#plotting(result_highpass, label_x = 'time', label_y = 'EDA_SCR')


plotting_2G(data1=result_lowpass, label1_x = 'time (s)', label1_y = 'EDA_SCL', 
           data2=result_highpass, label2_x = 'time (s)', label2_y = 'EDA_SCR',  fs = 4)

##############################################################################

### storing filtered data in a nested list of arrays

x_arr_scl, x_arr_scr = filtered_data(path='./Database/E4/{}-E4-Drv{}/Left/EDA.csv', num_cases=13, norm= True)
x_hr_l, x_hr_h = filtered_data(path='./Database/E4/{}-E4-Drv{}/Left/HR.csv', num_cases=13, norm= True)
x_hr = non_filtered_data(path='./Database/E4/{}-E4-Drv{}/Left/HR.csv', num_cases=13, norm= True)

for i in range(0, num_cases-1):
    plotting_2G(data1=x_arr_scl[i], label1_x = 'time (s)', label1_y = 'EDA_SCL_{}'.format(i), 
                data2=x_arr_scr[i], label2_x = 'time (s)', label2_y = 'EDA_SCR', fs = 4)

##############################################################################
#### generating annotations
#### The annotations must be upsampled and match with current measurments 

def gen_ann(path):
    y_arr_ann = [] # nested list with annotations on subjective stress meassures
    #path = './Database/Subj_metric/SM_Drv{}.csv'
    path = path
    for i in range(0, num_cases-1):
        df_data = pd.read_csv(path.format(i+1))
        dataset = df_data.to_numpy()
        dataset = np.reshape(dataset, (np.shape(dataset)[0]))
        y_arr_ann.append(read_data(path.format(i+1)))
    return  y_arr_ann

y_test = gen_ann('./Database/Subj_metric/SM_Drv{}.csv')
plotting_2G(data1=x_arr_scl[0], label1_x = 'time (s)', label1_y = 'EDA_SCL', 
            data2=y_test1[0], label2_x = 'time (s)', label2_y = 'ANN',  fs = 4)

plotting_3G(data1=x_arr_scl[7], label1_x = 'time (s)', label1_y = 'EDA_SCL', 
            data2=y_test[7], label2_x = 'time (s)', label2_y = 'ANN',
            data3=x_hr[7][2:], label3_x = 'time(s)', label3_y = 'HR', fs=4)

##############################################################################
#### #### The annotations must be upsampled and match with current measurments 

def gen_ann_timing(path, num_cases):
    mat = np.zeros([num_cases-1, 16])
    df_data = pd.read_csv(path)
    dataset = df_data.to_numpy()
    
    for i in range(0, num_cases-1):
        mat[i,:] = dataset[i,1:]
    return  mat

def gen_data_timing(path, num_cases):
    mat = np.zeros([num_cases-1, 16])
    df_data = pd.read_csv(path)
    dataset = df_data.to_numpy()
    
    for i in range(0, num_cases-1):
        mat[i,:] = dataset[i,3:19]
    return  mat

##############################################################################
#### #### Testing 

path1  = './Database/Subj_metric/Annot_Subjective_metric.csv'
time_ann = gen_ann_timing(path1, num_cases)

path2 = './Database/E4/Annot_E4_Left.csv'
time_data =  gen_data_timing(path2, num_cases)


###  SEQUENCE
# |Z_start  | Z_end   | City_s1 | City_e1 | HWY_s | HWY_e |
# | City_s2 | City_e2 | City_s3 | City_e3 | HWY_s | HWY_e |
# | City_s3 | City_e3 | Z_start | Z_end   |

###  SEQUENCE - EQUIVALENCE ON STRESS LEVEL
# L - | L  | L | H | H | M | M |
# | H | H | H | H | M | M |
# | H | H | L | L | - L

#target_length = np.shape(x_arr_scl[0])
#ratio = int(target_length/np.shape(y_test[0]))

#def upsampling(time_data, time_ann):
#for i in range (0,int(np.shape(time_data)[1]/2)):
dif1 = time_data[:,1::2]-time_data[:,::2]  ### even minus odd
dif2 = time_ann[:,1::2]-time_ann[:,::2]
dif1 = time_data[:,2::2]-time_data[:,1:-1:2]
dif2 = time_ann[:,2::2]-time_ann[:,1:-1:2]

y_test1 = np.copy(y_test)
#y_test1[0]= np.pad(y_test1[0], (int(time_data[0,0]), int(np.shape(x_arr_scl[0])[0]-time_data[0,-1])), constant_values=(0, 0))

for i in range(0, num_cases-1):
    if i != 1:
        y_test1[i]= np.pad(y_test1[i], (int(time_data[i,0]), int(np.shape(x_arr_scl[i])[0]-time_data[i,-1])), constant_values=(0, 0))




##############################################################################

# Feature extraction (time)

def time_features(num_cases, x_arr_scl, x_arr_scr, y, shuffle = False):
    x_avg_scl = []
    x_max_scl = []
    x_min_scl = []
    x_var_scl = []
    x_kur_scl = []
    
    x_avg_scr = []
    x_max_scr = []
    x_min_scr = []
    x_var_scr = []
    x_kur_scr = []
    
    y_avg = []
    
    for i in range(0, num_cases-2):
        length = len(x_arr_scl[i])
        num_seg = round(length/(seg_time*fs)) 
        x_avg_scl_arr = np.zeros(num_seg)
        x_max_scl_arr = np.zeros(num_seg)
        x_min_scl_arr = np.zeros(num_seg)
        x_var_scl_arr = np.zeros(num_seg)
        x_kur_scl_arr = np.zeros(num_seg)
    
        x_avg_scr_arr = np.zeros(num_seg)
        x_max_scr_arr = np.zeros(num_seg)
        x_min_scr_arr = np.zeros(num_seg)
        x_var_scr_arr = np.zeros(num_seg)
        x_kur_scr_arr = np.zeros(num_seg)
        
        ## annotations (average)
        y_avg_scr_arr = np.zeros(num_seg)
        #y_max_scr_arr = np.zeros(num_seg)
        #y_min_scr_arr = np.zeros(num_seg)
        #y_var_scr_arr = np.zeros(num_seg)
        #y_kur_scr_arr = np.zeros(num_seg)
        
                    
        #Y data
        #y_aux =  

        #print(np.shape(y_aux))
        
            
    
        
        for j in range(0, num_seg):
            ## X data 
            x_avg_scl_arr[j] = np.average(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            #print(np,shape(x_avg_scl))
            x_max_scl_arr[j] = np.max(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            x_min_scl_arr[j] = np.min(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            x_var_scl_arr[j] = np.var(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            x_kur_scl_arr[j] = kurtosis(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            
            x_avg_scr_arr[j] = np.average(x_arr_scr[i][j*batch_size:j*batch_size+batch_size])
            x_max_scr_arr[j] = np.max(x_arr_scr[i][j*batch_size:j*batch_size+batch_size])
            x_min_scr_arr[j] = np.min(x_arr_scr[i][j*batch_size:j*batch_size+batch_size])
            x_var_scr_arr[j] = np.var(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            x_kur_scr_arr[j] = kurtosis(x_arr_scl[i][j*batch_size:j*batch_size+batch_size])
            
            y_avg_scr_arr[j] =  np.average(y[i][j*batch_size:j*batch_size+batch_size])
            if i == 0:
                if np.isnan(y_avg_scr_arr[j]):
                    print('limit IS :', j*batch_size+batch_size)
                    print('limit IS :', [j*batch_size])
            



            
        if shuffle:
            x_avg_scl_arr, y_avg_scr_arr = random.shuffle(x_avg_scl_arr, y_avg_scr_arr)
            #np.random.shuffle(x_avg_scl_arr)

            
        x_avg_scl.append(x_avg_scl_arr)
        x_max_scl.append(x_max_scl_arr)
        x_min_scl.append(x_min_scl_arr)
        x_var_scl.append(x_var_scl_arr)
        x_kur_scl.append(x_kur_scl_arr)
        x_avg_scr.append(x_avg_scr_arr)
        x_max_scr.append(x_max_scr_arr)
        x_min_scr.append(x_min_scr_arr)
        x_var_scr.append(x_var_scr_arr)
        x_kur_scr.append(x_kur_scr_arr)
        ## annotated data        
        y_avg.append(y_avg_scr_arr )
    return y_avg, x_avg_scl, x_max_scl, x_min_scl, x_var_scl, x_kur_scl, x_avg_scr, x_max_scr, x_min_scr, x_var_scr, x_kur_scr


y_avg_all = gen_ann(path='./Database/Subj_metric/SM_Drv{}.csv')
y_avg_scr_arr, x_avg_scl, x_max_scl, x_min_scl, x_var_scl, x_kur_scl, x_avg_scr, x_max_scr, x_min_scr, x_var_scr, x_kur_scr =  time_features(num_cases, x_arr_scl, x_arr_scr, y_avg_all)    



plt.plot(np.repeat(x_avg_scl[0][20:30], batch_size))
#plt.plot(np.repeat(x_var_scl[0][20:30], batch_size))
#plt.plot(np.repeat(x_kur_scl[0,20:30], batch_size))
plt.plot(x_arr_scl[0][1600:2400])


##############################################################################
# Feature extraction (frequency)


"""
The frequency spectrum that fft() outputted was reflected about the y-axis so 
that the negative half was a mirror of the positive half. This symmetry was 
caused by inputting real numbers (not complex numbers) to the transform.
You can use this symmetry to make your Fourier transform faster by computing 
only half of it. scipy.fft implements this speed hack in the form of rfft().
"""


def freq_features(data, num_cases, plot):
    Sxx_list =  []
    
    for case in range(0,num_cases-1):
    
        N = int(len(data[case]))
        yf = rfft(data[case]-np.mean(data[case]))
        xf = rfftfreq(N, 1 / fs)
        
        # computing spectrum
        dt = 0.25
        T = N*dt
        #Sxx = (2 * dt ** 2 / T * xf * yf.conj()).real
        
        Sxx = 2 * dt ** 2 / T * (xf * np.conj(yf)) # Compute spectrum
        Sxx = Sxx[:int(len(data[case]) / 2)]  # Ignore negative frequencies1
        Sxx_list.append(Sxx)
        
        if plot:
            df = 1 / np.max(T)                         # Determine frequency resolution
            fNQ = 1 / dt / 2                           # Determine Nyquist frequency
            faxis = np.arange(0,fNQ,df)                # Construct frequency axis
            
            #plt.plot(faxis[:-1], np.real(Sxx))        # Plot spectrum vs frequency
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PS')
            plt.plot(faxis[:-1],abs(np.real(Sxx)))
            plt.show()
    return(Sxx_list)

#Sxx = freq_features(x_arr_scr, num_cases, plot=True)



def stft(data, num_cases, plot):
    for case in range(0,num_cases-2):
    
        f, t, Zxx = signal.stft(x=x_arr_scr[case], fs=4)
        fig, ax = plt.subplots(2, 1)
        pcm = ax[0].pcolor(t, f, np.abs(Zxx),
                            norm=colors.LogNorm(vmin=np.abs(Zxx).min()+0.01, vmax=np.abs(Zxx).max()+0.001),
                            cmap='PuBu_r', shading='auto')
        fig.colorbar(pcm, ax=ax[0], extend='max')
        pcm = ax[1].pcolor(t, f, np.abs(Zxx), cmap='PuBu_r', shading='auto')
        fig.colorbar(pcm, ax=ax[1], extend='max')
        plt.xlabel('time (s)')
        plt.show()


### Machine Learning methods
### supervised approach





#https://mark-kramer.github.io/Case-Studies-Python/03.html#power-spectral-density
#https://mark-kramer.github.io/Case-Studies-Python/03.html#step-4-power-spectral-density-or-spectrum
#https://m-gambera.medium.com/how-to-extract-features-from-signals-15e7db225c15

## Fourier analysis

# from scipy.fft import fft, fftfreq
# #https://docs.scipy.org/doc/scipy/tutorial/fft.html
# # Number of sample points BVP
# N = np.shape(dataset_bvp[2:])[0]
# # sample spacing
# T = 1.0 / 64.0
# x = np.linspace(0.0, N*T, N, endpoint=False)
# y = dataset_bvp#np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = fftfreq(N, T)[:N//2]
# import matplotlib.pyplot as plt
# #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# #plt.grid()
# #plt.show()


# # Number of sample points HR
# N = np.shape(dataset_hr[2:])[0]
# # sample spacing
# T = 1.0 / 4.0
# x = np.linspace(0.0, N*T, N, endpoint=False)
# y = dataset_eda#np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = fftfreq(N, T)[:N//2]
# import matplotlib.pyplot as plt
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()


### Literature
#https://pure.tue.nl/ws/portalfiles/portal/192416126/1_s2.0_S0167876021008461_main.pdf

#### Machine Learning methodologies



