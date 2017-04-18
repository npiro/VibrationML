# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:34:18 2017

@author: npiro
"""
import time
from MicrophoneRecorder import MicrophoneRecorder
from SpectrumProcessor import SpectrumProcessor, MachineLearning
import numpy as np
import matplotlib.pyplot as plt
import pickle
#%%
mic = MicrophoneRecorder(rate=8000, chunksize=2048)

#%%
mic.start()
time.sleep(200)
frames = mic.get_frames()
mic.stop()

#%%
with open('TimeFrames_C052_tapeAnomaly.pkl', 'w') as f:
    pickle.dump(frames,f)
