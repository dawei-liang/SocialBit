# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:21:18 2020
@author: david
"""

import librosa
import numpy as np
import os

import audio_utils.audio_io as io
import audio_utils.framing_utils as framing
    

#%%
def get_frames(audio_file, sr_new):
    """
    """
    sr, y = io.read_audio_data(audio_file)
    y = io.audio_pre_processing(y, sr=sr, sr_new=sr_new)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=frame_length, center=False).T
    rms_mean = np.mean(rms)
    print('file %s rms mean %f:' %(str(audio_file.split('/')[-1]), rms_mean))
    
    #pitches, magnitudes = librosa.piptrack(y=y, sr=sr,
    #                                       hop_length=512, n_fft=1024,
    #                                       win_length=None, window='hamming')
    #index = magnitudes.argmax(axis=0)
    #pitch = [pitches[i, t] for t, i in enumerate(index)]
    
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length).T   # final shape: [frame idx, window of samples] 
    return frames, rms, rms_mean

def save_audio(root, seg_list, frame_length, sr_new):
    for i in range(len(seg_list)):  
        keys = list(seg_list.keys())
        print('saving the %dth / %d segments' %(i+1, len(keys)))
        wav_filtered = framing.reconstruct_time_series(seg_list[keys[i]], hop_length_samples=frame_length)
        io.write_audio_data(root + '%s.wav' %(keys[i]), rate=sr_new, wav_data=wav_filtered)
    
#%%
sr_new = 16000
frame_length = 16000   # 1 sec per frame
rms_thres_coarse = 0.6
loaded_audio_file_raw = './field_study/pilot_edison/socialbit-et-home-072220.wav'
root_save_coarse = './field_study/pilot_edison/coarse/coarse/'
#%%

frames, rms, rms_mean = get_frames(loaded_audio_file_raw, sr_new)
frames_filtered = np.empty((0, frame_length))
rms_limit_coarse = rms_thres_coarse * rms_mean

t_gap = 0
seg_list = []
for i, e in enumerate(rms):
    if e >= rms_limit_coarse:
        print('%d / %d of total audio has been added' %(i, len(rms)))
        frames_filtered = np.vstack((frames_filtered, frames[i,:]))
        t_gap = 0
    else:
        t_gap += 1
    if t_gap == 60:
        print('filtered segment shape:', frames_filtered.shape) 
        seg_list.append(frames_filtered)
        frames_filtered = np.empty((0, frame_length))
save_audio(root_save_coarse, seg_list, frame_length, sr_new)

    
#%%
valid_list = [2, 4, 7, 8, 9, 16, 22, 23, 24, 27]
rms_thres_fine = 0.2
loaded_audio_coarse_root = root_save_coarse
root_save_fine = './field_study/pilot_edison/fine/'

valid_file_list = [x for y in valid_list for x in os.listdir(loaded_audio_coarse_root) if x=='%d.wav' %y]
seg_list = {}
for item in valid_file_list:
    path = loaded_audio_coarse_root + item
    frames_filtered = np.empty((0, frame_length))
    frames, rms, rms_mean = get_frames(path, sr_new)
    rms_limit_fine = rms_thres_fine * rms_mean
    for i, e in enumerate(rms):
        if e >= rms_limit_fine:
            frames_filtered = np.vstack((frames_filtered, frames[i,:]))
        seg_list[item.strip('.wav')] = frames_filtered

save_audio(root_save_fine, seg_list, frame_length, sr_new)