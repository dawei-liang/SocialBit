# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:58:17 2020

@author: david
"""

"""
SNR and RMS calculation follows:
https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
"""

import librosa
import numpy as np
import os

import check_dirs

#%%

ref = './recordings/data_augmentation/ref_voice/fossil_ref_voice.wav'
sig_root = './TIMIT_samples/' # root to training/test wav
noise_root = './recordings/data_augmentation/' # root to noise wav
snr_list = [3, 10]
SAMPLE_RATE = 16000


def overlay(sig, sig_ref, noise, snr):
    """
    args:
        sig: signal to be overlayed
        sig_ref: reference signals from the fossil watch
        noise: noise used for overlay
        snr: proposed snr
    return:
        mixture of the signals
    """
    # signal power  
    rms_sig = np.sqrt(np.mean(np.abs(sig)**2, axis=0, keepdims=True))
    # ref signal power 
    rms_ref = np.sqrt(np.mean(np.abs(sig_ref)**2, axis=0, keepdims=True))
    # noise power
    rms_noise = np.sqrt(np.mean(np.abs(noise)**2, axis=0, keepdims=True))
    # adjust the voice to be similar to that from the fossil watch
    sig = sig * (rms_ref/rms_sig)
    rms_sig = np.sqrt(np.mean(np.abs(sig)**2, axis=0, keepdims=True))   # should be = rms_ref
#    print('rms of ref and sig (for check):', rms_ref, rms_sig)
    # rms watts to dB
    rms_sig_squared_db = 10 * np.log10(rms_sig ** 2)
    # snr = 10(log(rms1^2)/(rms2^2)) = 10log(rms1^2) - 10log(rms2^2)
    rms_noise_squared_db = rms_sig_squared_db - snr
    # dB to watts
    rms_noise_proposed = np.sqrt(10 ** (rms_noise_squared_db / 10))
    # adjusted noise
    noise_new = noise * (rms_noise_proposed/rms_noise)
#    # calculate the noise rms in dB and check the new snr
#    rms_noise_new = np.sqrt(np.mean(np.abs(noise_new)**2, axis=0, keepdims=True))
#    rms_noise_new_squared_db = 10 * np.log10(rms_noise_new ** 2)
#    print('proposed snr, new snr: %f%f' %(snr, rms_sig_squared_db - rms_noise_new_squared_db))
    # mix the sound
    mix = sig + noise_new
    return mix
    
#%%
y_ref, sr = librosa.load(ref)

subject_list = [path.split('/')[-1] \
                for path, subd, f in os.walk(sig_root) for item in f if item.endswith('.WAV')]
subject_list = list(dict.fromkeys(subject_list))   # list of subject ids

noise_wav_list = [os.path.join(noise_root, item) for item in os.listdir(noise_root) if item.endswith('.wav')]
# Loop over each subject
for sub in subject_list:
    print('subject:', sub)
    sub_path = os.path.join(sig_root, sub)   # path to the sub
    sub_wav = [os.path.join(sub_path, item) for item in os.listdir(sub_path) if item.endswith('.WAV')]
    sig_training, sig_test = np.empty((1,0)), np.empty((1,0))
    # 8 clips for training
    for i in range(8):
        sig, sr = librosa.load(sub_wav[i])
        sig = librosa.core.resample(sig, orig_sr=sr, target_sr=SAMPLE_RATE)
        sig_training = np.append(sig_training, sig)
    # 2 clips for validation
    for i in range(8,10):
        sig, sr = librosa.load(sub_wav[i])
        sig = librosa.core.resample(sig, orig_sr=sr, target_sr=SAMPLE_RATE)
        sig_test = np.append(sig_test, sig)
    # Loop over each noise segment
    for snr in snr_list:
        print('snr:', snr)
        for noise_wav in noise_wav_list:
            noise, sr = librosa.load(noise_wav)
            noise = librosa.core.resample(noise, orig_sr=sr, target_sr=SAMPLE_RATE)
            noise_training_resized = np.resize(noise, len(sig_training))
            noise_test_resized = np.resize(noise, len(sig_test))
            training_with_noise = overlay(sig=sig_training, 
                                          sig_ref = y_ref,
                                          noise=noise_training_resized,
                                          snr=snr)
            test_with_noise = overlay(sig=sig_test, 
                                      sig_ref = y_ref,
                                      noise=noise_test_resized,
                                      snr=snr)        
            noise_idx = noise_wav.split('/')[-1].strip('.wav')
            training_root = './TIMIT_samples/' + sub + '/training/'
            test_root = './TIMIT_samples/' + sub + '/test/'
            check_dirs.check_dir(training_root)
            check_dirs.check_dir(test_root)
            librosa.output.write_wav(training_root + noise_idx + '_snr' + str(snr) + '.wav', 
                                     training_with_noise, 
                                     sr=SAMPLE_RATE, 
                                     norm=False)
            librosa.output.write_wav(test_root + noise_idx + '_snr' + str(snr) + '.wav', 
                                     test_with_noise, 
                                     sr=SAMPLE_RATE, 
                                     norm=False)


