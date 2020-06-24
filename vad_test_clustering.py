#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:06:18 2019

@author: dawei
"""
from scipy.io import wavfile
import numpy as np
import subprocess
import resampy

from python_speech_features import mfcc
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import librosa
from sklearn.metrics import silhouette_samples, silhouette_score

#%%


def read_audio_data(file):
    '''read audio, only support 16-bit depth'''
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype  # check input audio rate(int16)
    scaled_data = wav_data / 32768.0   # 16bit standardization
    return rate, scaled_data

def audio_pre_processing(data, sr, sr_new):
    # Convert to mono.
    try:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)
    except:
        pass
    # Resampling the data to specified rate
    if sr != sr_new:
      data = resampy.resample(data, sr, sr_new)
    return data

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


def transform(path_load, path_save, on = False):
    if on:
        # to change volumn:
        # + " -filter:a 'volume=0' " \
        # chop (in seconds): + " -ss 3 -to 21 " \
        command = "ffmpeg -i " + path_load \
                + " -vn -sample_fmt s16 " \
                + " -filter:a 'volume=1' " \
                + path_save
        subprocess.call(command, shell=True)
        
def mfcc_cal(audio_data, file_sampling_rate, window_length_secs, hop_length_secs, mfcc_size):
    # calculate mfcc features, shape: [# frames, mfcc size]
    mfcc_feat = librosa.feature.mfcc(audio_data, 
                                     sr=file_sampling_rate, 
                                     win_length=int(window_length_secs*file_sampling_rate), 
                                     hop_length=int(hop_length_secs*file_sampling_rate), 
                                     n_mfcc=mfcc_size).T
#    mfcc_feat = mfcc(signal=audio_data, 
#                 samplerate=file_sampling_rate,
#                 winlen=window_length_secs,
#                 winstep=hop_length_secs,
#                 numcep=mfcc_size,
#                 nfilt=26,
#                 nfft=960)   
    return mfcc_feat

def rms(audio_data, file_sampling_rate, window_length_secs, hop_length_secs):
    # rms: [# frames, 1]
    S, phase = librosa.magphase(librosa.stft(audio_data, 
                                             win_length=int(window_length_secs*file_sampling_rate), 
                                             hop_length=int(hop_length_secs*file_sampling_rate)))
    rms = librosa.feature.rms(S=S).T
    return rms
    
def pitch(audio_data, file_sampling_rate, window_length_secs, hop_length_secs):
    # pitch: [# frames, 1]
    pitches, magnitudes = librosa.piptrack(audio_data, 
                                           sr=file_sampling_rate,  
                                           win_length=int(window_length_secs*file_sampling_rate), 
                                           hop_length=int(hop_length_secs*file_sampling_rate))
    pitch_pos = np.argmax(magnitudes, axis=0)
    return pitch_pos.reshape((len(pitch_pos), 1))

def truth_labels(frame_size, start, stop):
    truth_labels = np.zeros(frame_size)
    for i in range(start, stop):
        truth_labels[i] = 1
    return truth_labels

#%%
if __name__ == '__main__':
    dir_load_audio = './preliminary_test/My_recording_3.wav'
    dir_load_audio_new = './preliminary_test/My_recording_3_new.wav'
    sr_new = 16000
    transform(dir_load_audio, dir_load_audio_new, False)
    sr, audio_data = read_audio_data(dir_load_audio_new)

    audio_data_resampled = audio_pre_processing(audio_data, sr, 16000)
    window_length = 0.05   # s*hz
    hop_length = window_length
    #framed_audio = framing(audio_data_resampled, int(window_length), int(hop_length))
    mfcc_feat = mfcc_cal(audio_data=audio_data_resampled, 
                         file_sampling_rate=sr_new, 
                         window_length_secs = window_length, 
                         hop_length_secs = hop_length, 
                         mfcc_size=15)
    rms_feat = rms(audio_data=audio_data_resampled, 
                         file_sampling_rate=sr_new, 
                         window_length_secs = window_length, 
                         hop_length_secs = hop_length)
    pitch_feat = pitch(audio_data=audio_data_resampled, 
                         file_sampling_rate=sr_new, 
                         window_length_secs = window_length, 
                         hop_length_secs = hop_length)
    feature = np.hstack((mfcc_feat, rms_feat))
    feature = np.hstack((feature, pitch_feat))
    # texture window: 1 sec(20 mfcc vec)
    feat_grouped = np.empty((0, len(feature[0])))
    i=0
    while i < len(mfcc_feat):
        feat_grouped = np.vstack((feat_grouped, np.mean(feature[i:i+20, :], axis=0)))
        i+=20
    #truth_labels = truth_labels(len(mfcc_feat), start=460, stop=640)
#%% 
#    range_n_clusters = [2, 3, 4, 5, 6]
#    for n_clusters in range_n_clusters:
#        cluster = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0)
#        labels = cluster.fit_predict(feat_grouped)
#        silhouette_avg = silhouette_score(feat_grouped, labels)
#        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        
    cluster = KMeans(n_clusters=3, n_jobs=-1, random_state=0)
    labels = cluster.fit_predict(feat_grouped)

#    clf = RandomForestClassifier(n_estimators=100, random_state=0, verbose=1, n_jobs=-1)
#    clf.fit(mfcc_feat, truth_labels)
#    
##%%
#    dir_load_audio_test = './preliminary_test/My_recording_5.wav'
#    dir_load_audio_new_test = './preliminary_test/My_recording_5_new.wav'
#    transform(dir_load_audio_test, dir_load_audio_new_test, False)
#    sr_test, audio_data_test = read_audio_data(dir_load_audio_new_test)
#
#    audio_data_resampled_test = audio_pre_processing(audio_data_test, sr_test, 16000)
#    #framed_audio = framing(audio_data_resampled, int(window_length), int(hop_length))
#    mfcc_feat_test = mfcc_cal(audio_data=audio_data_resampled_test, 
#                         file_sampling_rate=sr_new, 
#                         window_length_secs = 0.05, 
#                         hop_length_secs = 0.05, 
#                         mfcc_size=15)
#    pred = clf.predict(mfcc_feat_test)
    