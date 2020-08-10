#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:22:43 2020

@author: dawei
"""

"""
supervised method for speech type recognition
"""

import os
import pandas as pd
from numpy.random import seed
import numpy as np
import csv
import librosa.display

import check_dirs

#%%
def select_segment(segment_length, duration):
    """
    randomly select a start and an end point within an audio clip for segmentation
    args:
        segment_length: temporal size of the segment
        duration: desired duration for segmentation
    return:
        start and end points for segmentation
    """
    start = np.random.randint(low=0, high=segment_length - duration, size=1)[0]
    end = start + duration
    return start, end


def expand_segment(segment, desired_size):
    """
    Given a clip of audio, chunk it for a desired size (with or without expansion)
    
    args:
        segment: the audio clip that may need to be expanded, shape: [630, temporal size]
        desired_size: the desired temporal size
    return:
        audio segment of the desired size, shape: [630, desired_size]
    """
    if segment.shape[1] > desired_size:
        s, e = select_segment(segment.shape[1], duration = desired_size)   
    else:
        diff_times = int((desired_size) // segment.shape[1]) + 1
        segment = np.repeat(segment, repeats=diff_times, axis=1)
        #print(segment.shape, desired_size, diff_times)
        s, e = select_segment(segment.shape[1], duration = desired_size)
    return segment[:, s:e]
        
#%%
feat_path = './TIMIT_feat/TIMIT_embedding_from_CNN'
save_path_speech = './conversational_feat/speech/'
save_path_dyad = './conversational_feat/dyad/'
save_path_group = './conversational_feat/group/'
save_path_real = './conversational_feat/real_data/'
check_dirs.check_dir(save_path_speech)
check_dirs.check_dir(save_path_dyad)
check_dirs.check_dir(save_path_group)
check_dirs.check_dir(save_path_real)

feat_list = [os.path.join(feat_path, sub) for sub in os.listdir(feat_path) if sub.endswith('csv')]
sub_feat_dict, sub_labels = {}, []
for feat in feat_list:
    sub_feat = pd.read_csv(feat, header=None).values.T
    sub_label = feat.split('/')[-1].split('.')[0]
    sub_feat_dict[sub_label] = sub_feat
    sub_labels.append(sub_label)
    
temp=float('inf')
for i in sub_feat_dict:
    temp = min(temp, sub_feat_dict[i].shape[1])
print('check the smallest audio clip size:', temp)

#%%
t = 30   # desired temporal size of an interaction unit

''' feat for speech '''
for sub_i, sub in enumerate(sub_labels):
    print('selected subject: ', sub_i, sub)
 
    speech_feat_temp = expand_segment(segment=sub_feat_dict[sub], desired_size=t)
    print('speech_feat shape:', speech_feat_temp.shape)   # [size of embedding, 60]
    
    with open(save_path_speech + sub + "_speech.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(speech_feat_temp)
    csvfile.close()
    

''' feat for dyad conversation '''
for sub_i, sub in enumerate(sub_labels):
    print('selected subject: ', sub_i, sub)
    # the other selected subject
    sub2 = np.random.choice(sub_labels, size=1)[0]
    
    duration_sub1 = np.random.randint(low=3,high=24,size=1)[0]
    sub1_feat = expand_segment(segment=sub_feat_dict[sub], desired_size=duration_sub1)
    
    duration_sub2 = np.random.randint(low=3,high=t-duration_sub1-3,size=1)[0]
    sub2_feat = expand_segment(segment=sub_feat_dict[sub2], desired_size=duration_sub2)
    dyad_feat_temp = np.hstack((sub1_feat, sub2_feat))
    
    sub3 = [sub, sub2][np.random.randint(low=0,high=2,size=1)[0]]     
    duration_sub3 = t - duration_sub1 - duration_sub2
    sub3_feat = expand_segment(segment=sub_feat_dict[sub3], desired_size=duration_sub3)  
    dyad_feat_temp = np.hstack((dyad_feat_temp, sub3_feat))
    
    
#    while len(dyad_feat_temp[0]) < t:
#        sub_id = [sub, sub2][np.random.randint(low=0,high=2,size=1)[0]]     
#        duration_sub1_1 = t - duration_sub1 - duration_sub2
#        duration_sub1_1 = np.random.randint(low=5,high=10,size=1)[0]        
#    duration_sub2_1 = np.random.randint(low=5,high=10,size=1)
#    duration_sub1_2 = np.random.randint(low=5,high=10,size=1)
#    duration_sub2_1 = 5 - duration_sub1_1

    # reshape from (size of embedding,) to [size of embedding, 1]
#        if len(sub_feat_dict[sub_id][0]) > duration_sub1_1:
#            s, e = select_segment(len(sub_feat_dict[sub_id][0]), duration_sub1_1)
#            sub1_feat = sub_feat_dict[sub_id][:, s:e]  
#            dyad_feat_temp = np.hstack((dyad_feat_temp, sub1_feat))
#        else:
#            dyad_feat_temp = np.hstack((dyad_feat_temp, sub_feat_dict[sub_id][:, :]))
        
    #s, e = select_segment(len(sub_feat_dict[sub2][0]), duration_sub2_1)
    #sub2_feat = sub_feat_dict[sub2][:, s:e]
    #dyad_feat_temp = dyad_feat_temp[:, :t]
    # expand the mean features for each subject
#    sub1_1_feat_expanded = np.repeat(sub1_feat, repeats=duration_sub1_1, axis=1)
#    sub1_2_feat_expanded = np.repeat(sub1_feat, repeats=duration_sub1_2, axis=1)    
#    sub2_1_feat_expanded = np.repeat(sub2_feat, repeats=duration_sub2_1, axis=1)
#    sub2_2_feat_expanded = np.repeat(sub2_feat, repeats=duration_sub2_2, axis=1)

    #dyad_feat_temp = np.hstack((sub1_feat, sub2_feat))
    #dyad_feat_temp = np.hstack((dyad_feat_temp, sub2_1_feat_expanded))
    #dyad_feat_temp = np.hstack((dyad_feat_temp, sub2_2_feat_expanded))
    print('dyad_feat shape:', dyad_feat_temp.shape)   # [size of embedding, 60]
    
    with open(save_path_dyad + sub + '_' + sub2 + '_' + sub3 + "_dyad.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(dyad_feat_temp)
    csvfile.close()
        

''' feat for group conversation '''
for sub_i, sub in enumerate(sub_labels):
    print('selected subject: ', sub_i, sub)
    # the other selected subjects
    sub2 = np.random.choice(sub_labels, size=1)[0]
    sub3 = np.random.choice(sub_labels, size=1)[0]
    while sub3 == sub or sub3 == sub2:
        sub3 = np.random.choice(sub_labels, size=1)[0]
    
    duration_sub1 = np.random.randint(low=3,high=24,size=1)[0]
    sub1_feat = expand_segment(segment=sub_feat_dict[sub], desired_size=duration_sub1)
    
    duration_sub2 = np.random.randint(low=3, high = t-duration_sub1-3,size=1)[0]
    sub2_feat = expand_segment(segment=sub_feat_dict[sub2], desired_size=duration_sub2)
    group_feat_temp = np.hstack((sub1_feat, sub2_feat))
    
    duration_sub3 = t - duration_sub1 - duration_sub2
    sub3_feat = expand_segment(segment=sub_feat_dict[sub3], desired_size=duration_sub3)
    group_feat_temp = np.hstack((group_feat_temp, sub3_feat))
    
    with open(save_path_group + sub + '_' + sub2 + '_' + sub3 + "_group.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(dyad_feat_temp)
    csvfile.close()

#%%
#import matplotlib.pyplot as plt
#import seaborn as sns
##plt.figure(figsize=(12, 8))
#a=librosa.display.cmap(speech_feat_temp) 
#ax = sns.heatmap(dyad_feat_temp[0:200], linewidth=0.5)
#plt.show()

