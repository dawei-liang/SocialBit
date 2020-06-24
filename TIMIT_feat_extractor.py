#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:00:01 2020

@author: dawei
"""

'''
citations for TIMIT: https://github.com/philipperemy/timit
file preparation: http://academictorrents.com/give/

Extrctor of fixed length mel-spectrogram from the TIMIT audio data
Features are saved in CSVs with the format of (subject id)_(sentence id)_(seg id)
'''


import librosa as lib
import numpy as np
import os, csv

import check_dirs

#%%
def feat_processor(wav_file_list, srate, n_fft, hop_length, save_dir):
    """
    args:
        wav_file_list: wav list
        srate, n_fft, hop_length
        save_dir: dir to save the csv features (training/test)
    feat naming format: (subject id)_(sentence id)_(seg id)
    """
    subject_counter = {}
    for idx, wav in enumerate(wav_file_list):
        # set up the training/test split of the upcoming csv files, 8/10 training for each sub
        sub, sentence = wav.split('/')[-2], wav.split('/')[-1]
        if sub not in subject_counter:
            subject_counter[sub] = 1
        else:
            subject_counter[sub] += 1
        if subject_counter[sub] <= 8:
            data_type = 'training'
        else:
            data_type = 'test'
        # naming format: subject_sentence
        name = sub + '_' + sentence
        # remove '.wav' in the upcoming csv names
        name = name.split('.')[0]
        
        y, sr = lib.load(wav,sr=None)
        
        if len(y.shape) > 1:
            print ('Mono Conversion') 
            y = lib.to_mono(y)
        
        if sr != srate:
            print ('Resampling to {}'.format(srate))
            y = lib.resample(y, sr, srate)
            
        mel_feat = lib.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=128)
        # transform as by Lukic et al. 2016, avoid negative scale for CNN
        mel_feat = mel_feat * 10000 + 1
        inpt = lib.power_to_db(mel_feat/10, ref=1.0)/10   # shape: [128, # of frames]
        print(name, 'in shape', inpt.shape)
        
        # snippets extraction for every (srate // hop_length) frames. In our study: 100 frames(1.05 sec)
        n = inpt.shape[1] // (srate // hop_length)
        for seg_idx in range(n):
            seg = inpt[:, seg_idx * (srate // hop_length):(seg_idx+1) * (srate // hop_length)]            
            # save mel spec feat
            feat_path = os.path.join(save_dir, data_type)
            with open(os.path.join(feat_path, name + '_' + str(seg_idx) + ".csv"), 'w', newline='') as csvfile:   # no gap between lines
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerows(seg)
            csvfile.close()
        
#%%
srate = 16000
n_fft = 1024
hop_length = 160

root = './TIMIT/TIMIT/'   # root to load wav
# obtain path to training/test wav
training_wav = [os.path.join(path, item) \
                for path,subd,f in os.walk(root+'TRAIN/') for item in f if item.endswith('.WAV')]
test_wav = [os.path.join(path, item) \
            for path,subd,f in os.walk(root+'TEST/') for item in f if item.endswith('.WAV')]
# In our implementation, we do not use the train/test split as provided
combine_wav = training_wav + test_wav

#%%
save_dir = './TIMIT_feat'
check_dirs.check_dir(os.path.join(save_dir, 'training'))
check_dirs.check_dir(os.path.join(save_dir, 'test'))
feat_processor(combine_wav, srate, n_fft, hop_length, save_dir=save_dir)