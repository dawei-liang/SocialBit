# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:27:37 2020

implementation of DTW on MFCC features
adopted from https://github.com/pierre-rouanet/dtw/blob/master/examples/MFCC%20%2B%20DTW.ipynb

"""

import librosa
from matplotlib.pyplot import subplot
import librosa.display
from dtw import dtw
from numpy.linalg import norm

y1, sr1 = librosa.load('./recordings/5.wav')
y2, sr2 = librosa.load('./recordings/4.wav')

subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1, sr1)
librosa.display.specshow(mfcc1)

subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print ('Normalized distance between the two sounds:', dist) 

