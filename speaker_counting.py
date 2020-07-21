# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:06:20 2020

@author: david
"""

"""
spkear clustering based on different criterion, with the crowd++ method and NN embedding/MFCCs
"""

import audio_utils.audio_io as io
import audio_utils.framing_utils as framing

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

#%%
# Compute BIC distance between two MFCC features
def cluter_on_bic(mfcc_s1, mfcc_s2):
    mfcc_s = np.concatenate((mfcc_s1, mfcc_s2), axis=1)

    m, n = mfcc_s.shape
    m, n1 = mfcc_s1.shape
    m, n2 = mfcc_s2.shape

    sigma0 = np.cov(mfcc_s).diagonal()
    eps = np.spacing(1)
    realmin = np.finfo(np.double).tiny
    det0 = max(np.prod(np.maximum(sigma0,eps)),realmin)

    part1 = mfcc_s1
    part2 = mfcc_s2

    sigma1 = np.cov(part1).diagonal()
    sigma2 = np.cov(part2).diagonal()

    det1 = max(np.prod(np.maximum(sigma1, eps)), realmin)
    det2 = max(np.prod(np.maximum(sigma2, eps)), realmin)

    BIC = 0.5 * (n * np.log(det0) - n1 * np.log(det1) - n2 * np.log(det2)) - 0.5 * (m + 0.5 * m * (m + 1)) * np.log(n)
    return BIC


def cluter_on_cos(mfcc_s1, mfcc_s2):
    mfcc_s1 = np.mean(mfcc_s1, axis=1).reshape((mfcc_s1.shape[0], 1)).T
    mfcc_s2 = np.mean(mfcc_s2, axis=1).reshape((mfcc_s2.shape[0], 1)).T
    dis = cosine_distances(mfcc_s1, mfcc_s2)

    return dis * 1000

def pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr,
                                       hop_length=160, n_fft=1024,
                                       win_length=None, window='hamming')
    index = magnitudes.argmax(axis=0)
    pitch = [pitches[i, t] for t, i in enumerate(index)]
    return pitch
#%%
hop_length, frame_size, sr_new = 160, 1024, 16000

sr, y = io.read_audio_data('G:/Research6-Socialbit/recordings/remove_0.2_rms/sarnab_remove_rms.wav')
y = io.audio_pre_processing(y, sr=sr, sr_new=sr_new)
mfcc = librosa.feature.mfcc(y=y, sr=sr_new, n_mfcc=19, hop_length=hop_length, n_fft=frame_size)
pitch_value = pitch(y, sr_new)
mfcc = np.vstack((mfcc, pitch_value))
# mfcc mean for every 1 sec

n = int(mfcc.shape[1] // (sr_new // hop_length))
mfcc_mean = []
for seg_idx in range(n):
    mfcc_seg = mfcc[:, seg_idx * int((sr_new // hop_length)):(seg_idx+1) * int((sr_new // hop_length))]
    mfcc_mean.append(mfcc_seg)
mfcc_mean = np.asarray(mfcc_mean)

clusters = {0: mfcc_mean[0]}
counter = {0: 1}   # counting the summed elements for each item in clusters, used for mean calculation
for i in range(1, mfcc_mean.shape[0]):
    max_thres, min_thres = 0, 0   # 12往右，3往左
    
    min_dis, max_dis = np.float('inf'), np.float('-inf')
    # mark the nearest and farthest clusters
    for cluster in list(clusters):
        dis = cluter_on_cos(mfcc_mean[i], clusters[cluster])
        if min(min_dis, dis) == dis:
                min_dis, min_index = dis, cluster
        if max(max_dis, dis) == dis:
                max_dis = dis
    print(i, 'dis:', dis, 'max:', max_dis, 'min:', min_dis)
    # an uncertain point
    if max_dis <= max_thres and min_dis >= min_thres:
        continue
    # a point similar to an existing cluster
    if max_dis < min_thres:
        # 更新均值
        index_sum = clusters[min_index] * counter[min_index]
        counter[min_index] += 1
        clusters[min_index] = (index_sum + mfcc_mean[i]) / counter[min_index]       
    # a new cluster
    if min_dis > max_thres:
        clusters[i] = mfcc_mean[i]
        counter[i] = 1
print(list(clusters))

#%%
mfcc_mean_grouped = []
for i in range(len(mfcc_mean)):
    mfcc_mean_grouped.append(np.mean(mfcc_mean[i], axis=1))
mfcc_mean_grouped = np.asarray(mfcc_mean_grouped)   # [sec, 20]

#%%
#from scipy.spatial.distance import cosine
#print(cluter_on_cos(np.array([1,2,3,4,1]).reshape((1,5)), np.array([2,2,3,4,8]).reshape((1,5))))   
#print(cluter_on_cos(np.array([1,2,3,4,5,2]).reshape((1,6)), np.array([1,2,3,4,5,4]).reshape((1,6))))
#print(cosine(np.array([1,2,3,4,5,2]).reshape((1,6)), np.array([1,2,3,4,5,4]).reshape((1,6))))
