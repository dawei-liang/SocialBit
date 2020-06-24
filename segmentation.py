# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:36:04 2020

@author: david
"""

"""
segment audio based on given voice index (in sec), then save the segments individually in wav (mono, 16k)
"""

from scipy.io import wavfile
import numpy as np
import pandas as pd
import resampy

import check_dirs

# for audio
rate, wav_data = wavfile.read('./recordings/long_test1.wav')

if rate != 16000:
  wav_data = resampy.resample(wav_data, rate, 16000)

try:
    if wav_data.shape[1] > 1:
        wav_data = np.mean(wav_data, axis=1)
except:
    pass

# for look-up table
segments_sec = pd.read_csv('./recordings/long1/voice1.csv', header=None).values
# to save
root_2_save_voice = './recordings/long1/voice/'
root_2_save_env_sounds = './recordings/long1/env/'
check_dirs.check_dir(root_2_save_voice), check_dirs.check_dir(root_2_save_env_sounds)

# segment and save voice
for i in range(len(segments_sec)):
    start_sec, end_sec = segments_sec[i,0], segments_sec[i,1]
    start_sample, end_sample = start_sec * 16000, end_sec * 16000
    segments_wav = wav_data[start_sample:end_sample]
    wavfile.write(root_2_save_voice + str(start_sec) + '_' + str(end_sec) + '.wav', 16000, segments_wav)
    print('voice %d out of %d saved' % (i, len(segments_sec)-1))
    
# segment and save env sounds
for i in range(len(segments_sec) - 1):
    start_sec, end_sec = segments_sec[i,1], segments_sec[i+1,0]
    start_sample, end_sample = start_sec * 16000, end_sec * 16000
    segments_wav = wav_data[start_sample:end_sample]
    wavfile.write(root_2_save_env_sounds + str(start_sec) + '_' + str(end_sec) + '.wav', 16000, segments_wav)
    print('env %d out of %d saved' % (i, len(segments_sec)-2))
# for the first/last seg, i is from above
if segments_sec[0,0] != 0:
    end_sec = segments_sec[0,0]
    end_sample = end_sec * 16000
    segments_wav = wav_data[:end_sample]
    wavfile.write(root_2_save_env_sounds + '_' + str(end_sec) + '.wav', 16000, segments_wav)
    print('env first extra seg saved')
if (segments_sec[i+1,1] * 16000) < len(wav_data):
    start_sec = segments_sec[i+1,1]
    start_sample = start_sec * 16000
    segments_wav = wav_data[start_sample:]
    wavfile.write(root_2_save_env_sounds + str(start_sec) + '_' + '.wav', 16000, segments_wav)
    print('env last extra seg saved')