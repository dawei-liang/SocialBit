#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:14:56 2020

@author: dawei
"""
"""
VAD by using voice energy subband and Transfer Learning method
"""

import librosa
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import get_labels
from vad_energy_subband import VoiceActivityDetector
import feat_extractor

#%%


def metrics(pred, truth):
    cf = confusion_matrix(truth, pred)
    tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
    acc = accuracy(truth, pred)
    return cf, tn, fp, fn, tp, acc


def framing(data, window_length, hop_length):
    """
    Convert 1D time series signals or N-Dimensional frames into a (N+1)-Dimensional array of frames.
    No zero padding, rounding at the end.
    Args:
        data: Input signals.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.
    Returns:
        np.array with as many rows as there are complete frames that can be extracted.
    """
    
    num_samples = data.shape[0]
    frame_array = data[0:window_length]
    # create a new axis as # of frames
    frame_array = frame_array[np.newaxis]  
    start = hop_length
    for _ in range(num_samples):
        end = start + window_length
        if end <= num_samples:
            # framing at the 1st axis
            frame_temp = data[start:end]
            frame_temp = frame_temp[np.newaxis]
            frame_array = np.concatenate((frame_array, frame_temp), axis=0)
        start += hop_length
    return frame_array




dir_to_load_wav = './recordings/'
folders = ['long1/voice', 'long2/voice', 'long3/voice']
sub = ['long1', 'long2', 'long3']
voice_array = {}
voice_array[sub[0]] = get_labels.load_voice(dir_to_load_wav, [folders[0]])
voice_array[sub[1]] = get_labels.load_voice(dir_to_load_wav, [folders[1]])
voice_array[sub[2]] = get_labels.load_voice(dir_to_load_wav, [folders[2]])

folders = ['long1/env', 'long2/env', 'long3/env']
env_array = {}
env_array[sub[0]] = get_labels.load_env(dir_to_load_wav, [folders[0]])
env_array[sub[1]] = get_labels.load_env(dir_to_load_wav, [folders[1]])
env_array[sub[2]] = get_labels.load_env(dir_to_load_wav, [folders[2]])

#%%

full_data = {}
full_data[sub[0]] = np.vstack((voice_array[sub[0]], env_array[sub[0]]))
full_data[sub[1]] = np.vstack((voice_array[sub[1]], env_array[sub[1]]))
full_data[sub[2]] = np.vstack((voice_array[sub[2]], env_array[sub[2]]))

# release memory
voice_array, env_array = 0, 0

#frames = framing(full_data[sub[0]], window_length=int(0.02*16000), hop_length=int(0.01*16000))


#%%
Energy_subband_pred = True
if Energy_subband_pred:
    print('Predicting with energy sub-band')
    output_windows = np.empty((0,3))
    for audio_item in sub:
        v = VoiceActivityDetector(rate=16000, data=full_data[audio_item])
        output_windows = np.vstack((output_windows, v.detect_speech()))
        # release memory
        full_data[audio_item] = 0
    pred = output_windows[:,1]
    print(accuracy(output_windows[:,2], pred))

#%%
TL_extractioin = False
if TL_extractioin:
    print('Extracting features by TL')
    # loop for each audio item
    for audio_item in sub:
        feat_array = np.empty((0, 513))   # 512 features + a label
        # loop for every 2 sec (32000 samples), save feat vectors for every item
        for seg in range(full_data[audio_item].shape[0]//32000):
            print('seg', seg)
            feat = feat_extractor.main(full_data[audio_item][seg*32000:(seg+1)*32000, 0])
            feat = np.append(feat, int(round(np.sum(full_data[audio_item][seg*32000:(seg+1)*32000, 1])/32000)))
            feat_array = np.vstack((feat_array, feat))
        np.savetxt('./recordings/csv_feat_TL/' + audio_item + ".csv", feat_array, delimiter=",")           

TL_prediction = False
if TL_prediction:
    path = './recordings/csv_feat_TL/'
    csv_list = [x for x in os.listdir(path) if x.endswith('.csv')]
    feat__training_array = np.empty((0, 513))   # 512 features + a label
    for i, item in enumerate(csv_list):
        print('item: ', item)
        # valid set, leave-one-session-out
        if i == 1:
            feat_valid = np.loadtxt(os.path.join(path, item), delimiter=',')
        # stacked array of training set
        else:
            feat_training = np.loadtxt(os.path.join(path, item), delimiter=',') 
            feat__training_array = np.vstack((feat__training_array, feat_training))
    feat_training_shuffled = shuffle(feat__training_array, random_state=0)
    feat_valid_shuffled = shuffle(feat_valid, random_state=0)
    
    model = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0, n_jobs=-1)
    model.fit(feat_training_shuffled[:, :-1], feat_training_shuffled[:, -1].astype(int))
    pred = model.predict(feat_valid_shuffled[:, :-1])
    print(accuracy(feat_valid_shuffled[:, -1].astype(int), pred))


    
