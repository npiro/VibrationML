# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 07:40:39 2017

@author: npiro
"""
import pickle
import numpy as np
from SpectrumProcessor import SpectrumProcessor, MachineLearning
import matplotlib.pyplot as plt
#import peakutils
import tensorflow as tf
#%% Data files Each pickle file is a list of frames. Each frame is
# a sequence of 2048 samples with a sampling rate of 8 kHz
fileNormal1 = 'p3TimeFrames_C043_1.pkl'

fileNormal2 = 'p3TimeFrames_C044_afterAnomalyRemoved.pkl'

fileNormal3 = 'p3TimeFrames_C052_afterAnomalyRemoved.pkl'


fileAnomalySmall = 'p3TimeFrames_C052_tapeAnomaly.pkl'

fileAnomalyLarge = 'p3TimeFrames_C044_anom.pkl'

with open(fileNormal1, 'rb') as f:
    framesNor = pickle.load(f, encoding='latin1')

with open(fileNormal2, 'rb') as f:
    framesNor2 = pickle.load(f, encoding='latin1')
  
with open(fileNormal3, 'rb') as f:
    framesNor3 = pickle.load(f, encoding='latin1')
    
with open(fileAnomalySmall, 'rb') as f:
    framesAno = pickle.load(f, encoding='latin1')
    
with open(fileAnomalyLarge, 'rb') as f:
    framesAnoLarge = pickle.load(f, encoding='latin1')
    

#%% FFT
def CalculateFramesFFT(frames, rate = 8000, chunksize = 2048, avgwind = 6, plot = False):
"""
This function calculates the fft of all frames in input argument,
which should be a list.
Params:
    frames: list of frames
    rate: sampling rate. default to 8000 Hz
    chunksize: num of samples per frame. Default: 2048
    avgwind: number of fft frames to average. Default: 6
    plot: whether to plot the result. Defaut: False
"""    
    freq_vect = np.fft.rfftfreq(chunksize, 1./rate)
    fft_frames = np.array([np.fft.rfft(fr) for fr in frames])
    fft_frames /= np.abs(fft_frames).max()
    fft_frames = np.abs(fft_frames)
    #subframes = fft_frames[0:avgwind,:]
    fft_frames = np.array([np.mean(fft_frames[i:i+avgwind,:],0) for i in range(0,len(fft_frames),avgwind)])

    if plot:
        plt.plot(freq_vect,fft_frames[0])
        plt.hold(True)
        for i in range(1,len(fft_frames)):
            plt.plot(freq_vect, fft_frames[i])
        plt.hold(False)
        ax = plt.gca()
        ax.set_yscale('log')
    return (freq_vect, fft_frames)
#%% Function to compute features for model 
import pdb
def GetFeaturesFromFrames(freq_vect, fft_frames, num_features = 6, sp = None, plot = False):
"""
Calculates features from a list of fft_frames (computed with CalculateFramesFFT)
params:
    freq_vect: vector of frequencies of each fft_frame (output of CalculateFramesFFT)
    fft_frames: list of FFT frames output of CalculateFramesFFT
    num_features: number of features to compute. Default: 6
    sp: Sound processing object. If Default to None: generated inside function
    plot: Whether to plot the results: default: False
"""
    if sp is None:
        sp = SpectrumProcessor()
    peak_inds = sp.getPeaksFromSpectrogram(freq_vect,fft_frames,num_features)
    maxinds = [np.argmax(a[peak_inds[i][0]]) for i, a in enumerate(fft_frames)]

    funds = [freq_vect[peak_ind[0][maxinds[i]]] for i, peak_ind in enumerate( peak_inds)]
    xpeaks = [freq_vect[peak_ind[0]]/funds[i] for i, peak_ind in enumerate(peak_inds)]
    ypeaks = [np.array(fft_frames[i][peak_ind[0]]) for i, peak_ind in enumerate(peak_inds)]
    
    sortinds = [np.argsort(xpeak) for xpeak in xpeaks]    
    xpeaks_sorted = [xpeak[sortind] for xpeak, sortind in zip(xpeaks,sortinds)]
    ypeaks_sorted = [ypeak[sortind] for ypeak, sortind in zip(ypeaks,sortinds)]      
                     
    X = [np.hstack((xpeaks_sorted, ypeaks_sorted))][0]
         
    return (X, peak_inds)
#%% Calculate FFTs, extract features and train svm for the following cases:
# 1) framesNor: normal behaviour: fan without anomaly
# 2) framesNor2: another file same as 1
# 3) framesNor3: idem
# 4) framesAno: Small anomaly case: small piece of tape in one blade of fan
# 5) framesAnoLarge: Large anomaly case: piece of metal on fan blade

num_features = 8
avgwind_train = 5
avgwind_test = 3
(freq_vect, fft_frames_nor) = CalculateFramesFFT(framesNor, avgwind=avgwind_train)
(X_norm, pi) = GetFeaturesFromFrames(freq_vect, fft_frames_nor, num_features=num_features)
(freq_vect, fft_frames_nor2) = CalculateFramesFFT(framesNor2, avgwind=avgwind_test)
(X_norm2, _) = GetFeaturesFromFrames(freq_vect, fft_frames_nor2, num_features=num_features)
(freq_vect, fft_frames_nor3) = CalculateFramesFFT(framesNor3, avgwind=avgwind_test)
(X_norm3, _) = GetFeaturesFromFrames(freq_vect, fft_frames_nor3, num_features=num_features)
(freq_vect, fft_frames_anom) = CalculateFramesFFT(framesAno, avgwind=avgwind_test)
(X_anom, _) = GetFeaturesFromFrames(freq_vect, fft_frames_anom, num_features=num_features)
(freq_vect, fft_frames_anom_L) = CalculateFramesFFT(framesAnoLarge, avgwind=avgwind_test)
(X_anom_L, _) = GetFeaturesFromFrames(freq_vect, fft_frames_anom_L, num_features=num_features)

# Make machine learning object. In this case we
# use the SVManom model (one-class support vector machine)
ml = MachineLearning()
ml.SVManom(kernel='rbf', nu=0.05, gamma = 0.5)
# concatenate all normal data together to use a single train/test set
Xall = np.vstack((X_norm, X_norm2, X_norm3))

# Split data in train and test sets
from sklearn.model_selection import train_test_split
probFailureNormal = []
probFailureAnomaly = []
probFailureAnomalyLarge = []

# Repeat 10 times, for each data set:
# split data in train and test sets (15% in test set)
# Fit machine learning model to train set
# Predict the test set to compute number of normal and anomalous cases
# compute probability of anomaly (number of anomalous cases)/(num of total cases)
for i in range(0,10):
    Xtrain, Xtest = train_test_split(Xall, test_size=0.15, train_size=None, random_state=None)
    ml.clf.fit(Xtrain)
    predNormal = ml.clf.predict(Xtest)
    probFailureNormal.append(sum(predNormal==-1)/float(len(predNormal)))
    predAnomaly = ml.clf.predict(X_anom)
    probFailureAnomaly.append(sum(predAnomaly==-1)/float(len(predAnomaly)))
    predAnomalyLarge = ml.clf.predict(X_anom_L)
    probFailureAnomalyLarge.append(sum(predAnomalyLarge==-1)/float(len(predAnomalyLarge)))

# print results
print(np.mean(probFailureNormal))
print(np.mean(probFailureAnomaly))
print(np.mean(probFailureAnomalyLarge))


#ml.clf.score(X_norm2)

#%% Plot results
import scipy as sp
time = np.linspace(0, 2048/8000., 2048)
framesNorNP = np.array(framesNor)
framesAnNP = np.array(framesAnoLarge)
frAllNor = framesNorNP.flatten()
frAllAn = framesAnNP.flatten()
plt.subplot(211)
plt.plot(time, frAllNor[:2048])
plt.ylabel('Amplitude')
plt.title('Normal')
plt.tight_layout

plt.subplot(212)
plt.plot(time, frAllAn[:2048])
plt.subplots_adjust(hspace = 0.5)

a = plt.gca()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Anomalous')
#%%
import scipy.signal as sig
SpectrumNor = sig.periodogram(frAllNor, fs = 8000)
SpectrumAn = sig.periodogram(frAllAn, fs = 8000)

#%%

plt.subplot(121, adjustable='box', aspect = 0.3)
plt.semilogy(freq_vect, np.mean(fft_frames_nor,0))
plt.ylabel('Power specturm')
plt.title('Normal')
a = plt.gca()
plt.subplot(122)
plt.semilogy(freq_vect, np.mean(fft_frames_anom,0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power specturm')
plt.title('Anomalous')
plt.subplots_adjust(hspace = 0.5)
plt.tight_layout()
#plt.set_aspect('equal')

#%%

plt.plot(X_norm[0])