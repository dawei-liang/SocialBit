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
import check_dirs
    

#%%
def get_frames(audio_file, sr_new):
    """
    Do framing on given audio and return the frames, rms of frames and mean rms (non-overlapping)
    args:
        audio_file: given audio
        sr_new: proposed re-sampling rate
    return:
        frames: array of frames, shape [time axis, # of samples per frame]
        rms: rms of frames, shape [time axis, 1]
        rms_mean: mean rms along time axis
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

#%%
def save_audio(dirc, seg_list, frame_length, sr_new):
    """
    save audio segmented audio files
    args:
        dirc: dir to save the audio
        seg_list: hash of audio seg
        frame_length: # of samples per frame, used for reconstruction
        sr_new: proposed saving rate
    return:
        none
    """
    for i in range(len(seg_list)):  
        keys = list(seg_list.keys())
        print('saving %d / %d segments' %(i+1, len(keys)))
        wav_filtered = framing.reconstruct_time_series(seg_list[keys[i]], hop_length_samples=frame_length)
        io.write_audio_data(dirc + '%s.wav' %(keys[i]), rate=sr_new, wav_data=wav_filtered)
    
#%%
'''main'''
sr_new = 16000
frame_length = 16000   # 1 sec per frame
loaded_audio_file_raw = './field_study/pilot_edison/socialbit-et-home-072220.wav'   # raw audio
dir_save_coarse = './field_study/pilot_edison/coarse/temp/'   # dir to save coarse segs
dir_loaded_audio_coarse = dir_save_coarse
valid_list = [8]   # coarse segs used for fine filtering
dir_save_fine = './field_study/pilot_edison/fine/temp/'   # dir to save fine segs
mode = 'coarse'
rms_thres_coarse, rms_thres_fine = 0.6, 0.2   # energy threshold for noise removal
t_thres_coarse, t_thres_fine = 60, 20   # max gaps between segs

if mode == 'coarse':
    frames, rms, rms_mean = get_frames(loaded_audio_file_raw, sr_new)
    frames_filtered, frame_noise = np.empty((0, frame_length)), np.empty((0, frame_length))
    rms_limit_coarse = rms_thres_coarse * rms_mean   
    t_gap = 0
    seg_list, noise = {}, {}   # list of kept/noise segs
    intra_seg_idx = []   # list of the segment indices
    for i, e in enumerate(rms):
        # if over rms threshold, keep the frame
        if e >= rms_limit_coarse:
            print('%d / %d of total audio has been filtered' %(i, len(rms)))
            frames_filtered = np.vstack((frames_filtered, frames[i,:]))
            # mark the current idx
            intra_seg_idx.append(i)
            t_gap = 0
        else:
            frame_noise = np.vstack((frame_noise, frames[i,:]))
            t_gap += 1
        # if over temporal threshold, cut a segment
        if t_gap == t_thres_coarse:
            print('filtered segment shape:', frames_filtered.shape)
            seg_name = str(min(intra_seg_idx)) + '-' + str(max(intra_seg_idx))
            seg_list[seg_name] = frames_filtered           
            frames_filtered = np.empty((0, frame_length))
            intra_seg_idx = []
    # save segs
    check_dirs.check_dir(dir_save_coarse)
    save_audio(dir_save_coarse, seg_list, frame_length, sr_new)
    # save noise
    noise['noise_coarse'] = frame_noise
    save_audio(dir_save_coarse, noise, frame_length, sr_new)
 
elif mode == 'fine':
    valid_file_list = [x for y in valid_list for x in os.listdir(dir_loaded_audio_coarse) if x=='%s.wav' %y]
    seg_list, noise = {}, {}
    intra_seg_idx = []   # list of the segment indices
    seg_name = []
    frame_noise = np.empty((0, frame_length))
    for item in valid_file_list:
        path = dir_loaded_audio_coarse + item
        frames_filtered = np.empty((0, frame_length))
        frames, rms, rms_mean = get_frames(path, sr_new)
        rms_limit_fine = rms_thres_fine * rms_mean
        for i, e in enumerate(rms):
            if e >= rms_limit_fine:
                frames_filtered = np.vstack((frames_filtered, frames[i,:]))
                intra_seg_idx.append(i)
            else:
                frame_noise = np.vstack((frame_noise, frames[i,:]))
            off_set = int(item.split('-')[0])   # offset time points of the current segment
            seg_name = str(off_set + min(intra_seg_idx)) + '-' + str(off_set + max(intra_seg_idx))
            seg_list[seg_name] = frames_filtered
        intra_seg_idx = []
    # save noise
    noise['noise_fine'] = frame_noise
    save_audio(dir_save_coarse, noise, frame_length, sr_new)
    
    # save segs
    check_dirs.check_dir(dir_save_fine)
    save_audio(dir_save_fine, seg_list, frame_length, sr_new)
    
