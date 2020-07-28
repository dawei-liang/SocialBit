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

sr, y = io.read_audio_data('G:/Research6-Socialbit/field_study/pilot_edison/socialbit-et-home-072220.wav')
y = io.audio_pre_processing(y, sr=sr, sr_new=16000)
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512, center=False).T
rms_mean = np.mean(rms)

#pitches, magnitudes = librosa.piptrack(y=y, sr=sr,
#                                       hop_length=512, n_fft=1024,
#                                       win_length=None, window='hamming')
#index = magnitudes.argmax(axis=0)
#pitch = [pitches[i, t] for t, i in enumerate(index)]

frames = librosa.util.frame(y, frame_length=1024, hop_length=512).T
frames_filtered = np.empty((1024, 0)).T
for i, e in enumerate(rms):
    if e >= 0.2 * rms_mean:
        #frames_filtered = np.hstack((frames_filtered, frames[i,:]))
        frames_filtered = np.vstack((frames_filtered, frames[i,:]))
        
wav_filtered = framing.reconstruct_time_series(frames_filtered, hop_length_samples=512)
io.write_audio_data('G:/Research6-Socialbit/field_study/remove_0.2_rms/et-home_remove_rms.wav', rate=16000, wav_data=wav_filtered)
        
