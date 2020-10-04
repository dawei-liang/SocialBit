# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 00:00:41 2020

@author: david
"""

import pandas as pd
import numpy as np

import audio_utils.audio_io as io
import check_dirs

target_file = 'outdoor'
audio_path = './field_study/field_data/p0/segments/%s.wav' % target_file # path to load the audio clip
label_path = './field_study/field_data/p0/labels/%s.csv' % target_file # path to load the csv labels
save_validation_audio = './field_study/field_data/p0/validation_audio/%s/' % target_file # path to save your segmentation

#%%
# Load wav data
sr, y = io.read_audio_data(audio_path)
y = io.audio_pre_processing(y, sr=sr, sr_new=16000)
seg_dict = {'wearer': np.empty((0,0)), 
            'others': np.empty((0,0)),
            'phone': np.empty((0,0)), 
            'tv':np.empty((0,0)), 
            'back':np.empty((0,0))} 
labels = pd.read_csv(label_path, header=None).values
for i in range(len(labels)):
    if labels[i,:] == '1':
        seg_dict['wearer'] = np.append(seg_dict['wearer'], y[i*16000:(i+1)*16000])
    elif labels[i,:] == '2':
        seg_dict['others'] = np.append(seg_dict['others'], y[i*16000:(i+1)*16000])
    elif labels[i,:] == 'p':
        seg_dict['phone'] = np.append(seg_dict['phone'], y[i*16000:(i+1)*16000])
    elif labels[i,:] == 't':
        seg_dict['tv'] = np.append(seg_dict['tv'], y[i*16000:(i+1)*16000])
    elif labels[i,:] == 'b':
        seg_dict['back'] = np.append(seg_dict['back'], y[i*16000:(i+1)*16000]) 
check_dirs.check_dir(save_validation_audio)
for seg in seg_dict.keys():
    if len(seg_dict[seg]) > 0:
        io.write_audio_data(save_validation_audio + '%s.wav' %seg, 16000, seg_dict[seg])