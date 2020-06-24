# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:49:48 2020

@author: david
"""

"""
concatenate and label audio segments for voice/env
"""

from scipy.io import wavfile
import numpy as np
import os

#import check_dirs

#root_save = './recordings/csv/'
#check_dirs.check_dir(root_save)

# voice
def load_voice(dir_to_load_wav, folders):
    """
    args:
        dir_to_load_wav: root dir to of wav folders
        folders: folders of wave files
    return:
        wave data with labels, shape: [# of samples, 2]
    """
    voice_array = np.empty((0,1))
    for i, folder in enumerate(folders):
        print('loading:', folder)
        wav_list = [x for x in os.listdir(os.path.join(dir_to_load_wav, folder)) if x.endswith('.wav')]
        for wav in wav_list:
            rate, wav_data = wavfile.read(os.path.join(dir_to_load_wav, folder) + '/' + wav)
            voice_array = np.append(voice_array, wav_data)        
    # add labels
    voice_labels = np.ones(len(voice_array))
    voice_array = np.vstack((voice_array, voice_labels)).T   # shape: [# of samples, 2]
    #np.savetxt(root_save + "voice_" + str(i) + "_.csv", voice_array, delimiter=",")
    return voice_array


# env
def load_env(dir_to_load_wav, folders):
    env_array = np.empty((0,1))
    for i, folder in enumerate(folders):
        print('loading:', folder)
        wav_list = [x for x in os.listdir(os.path.join(dir_to_load_wav, folder)) if x.endswith('.wav')]
        for wav in wav_list:
            rate, wav_data = wavfile.read(os.path.join(dir_to_load_wav, folder) + '/' + wav)
            env_array = np.append(env_array, wav_data)        
    # add labels
    env_labels = np.zeros(len(env_array))
    env_array = np.vstack((env_array, env_labels)).T   # shape: [# of samples, 2]   
    #np.savetxt(root_save + "env_" + str(i) + "_.csv", env_array, delimiter=",")
    return env_array

#%%
#dir_to_load_wav = './recordings/'
#folders = ['long1/voice', 'long2/voice', 'long3/voice']
#voice_array = load_voice(dir_to_load_wav, folders)
#folders = ['long1/env', 'long2/env', 'long3/env']
#env_array = load_env(dir_to_load_wav, folders)
    