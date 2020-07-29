# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:21:18 2020

@author: david
"""

import librosa
import numpy as np

import audio_utils.audio_io as io
import audio_utils.framing_utils as framing
    


#%%

sr, y = io.read_audio_data('./field_study/pilot_edison/socialbit-et-home-072220.wav')
frame_length = 16000   # 1 sec per frame
rms_thres = 0.6
y = io.audio_pre_processing(y, sr=sr, sr_new=16000)
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=frame_length, center=False).T
rms_mean = np.mean(rms)
print('rms mean:', rms_mean)

#pitches, magnitudes = librosa.piptrack(y=y, sr=sr,
#                                       hop_length=512, n_fft=1024,
#                                       win_length=None, window='hamming')
#index = magnitudes.argmax(axis=0)
#pitch = [pitches[i, t] for t, i in enumerate(index)]

frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length).T   # final shape: [frame idx, window of samples]
frames_filtered = np.empty((0, frame_length))
rms_limit = rms_thres * rms_mean

#%%
t_gap = 0
seg_list = []
for i, e in enumerate(rms):
    if e >= rms_limit:
        print('%d / %d of total audio has been added' %(i, len(rms)))
        frames_filtered = np.vstack((frames_filtered, frames[i,:]))
        t_gap = 0
    else:
        t_gap += 1
    if t_gap == 60:
        print('filtered segment shape:', frames_filtered.shape) 
        seg_list.append(frames_filtered)
        frames_filtered = np.empty((0, frame_length))
#%%
for i in range(len(seg_list)):      
    print('saving the %dth segments' %(i+1))
    wav_filtered = framing.reconstruct_time_series(seg_list[i], hop_length_samples=frame_length)
    io.write_audio_data('./field_study/pilot_edison/coarse/%d.wav' %(i+1), rate=16000, wav_data=wav_filtered)
        
