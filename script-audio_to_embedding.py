# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:35:54 2020

@author: david
"""

import librosa
import csv
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy

#import torch
#import torch.nn as nn

import audio_utils.audio_io as io
import check_dirs
import model_Lukic

target_file = 'reading'
audio_path = './field_study/field_data/p0/segments/%s.wav' % target_file # path of the audio clip
label_path = './field_study/field_data/p0/labels/%s.csv' % target_file # path of the csv
save_csv_dir = './field_study/field_data/p0/spectrogram_1s/%s/' % target_file # dir to save segments of spectrogram

weight_path = './CNN_feat_extractor-epoch_20-val_0.3729.hdf5' # path to load weights
torch_path = './CNN_feat_extractor-epoch_4-trainloss_0.0230669.pt'

# path and file names to save the computed embedding/mfcc features
save_embedding_path = './field_study/field_data/p0/embedding_1s/'
save_mfcc_path = './field_study/field_data/p0/mfcc_1s/'

embedding_layer = 'L9' # output embedding layer, can be 9, 11, 13
extract_spec = False   # Whether to extract and save the spectrogram first
num_mfcc = 14   # # of mfccs to use, 12 or 120


#%%

def load_csv(csv_list):
    """
    Obtain feat and labels from list of csv spectrogram
    
    args:
        list of csv paths
    return:
        data and labels in two lists 
    """
    feat, labels = [], []
    for csv_item in sorted(csv_list):
        spectrogram = pd.read_csv(csv_item, header=None).values
        # naming format of the csv: /../sub_seg.csv
        label = csv_item.split('/')[-1].split('_')[0]
        print('loaded spectrogram shape:', spectrogram.shape, 'label:', label)
        feat.append(spectrogram)
        labels.append(label)
    return feat, labels

#%%
    
def model_compiler(model_architecture, load_weight=True, weight_path=None, embedding_layer=None):
    """
    Compile the model
    
    arg:
        model: model architecture
        load_weight: whether to load pre-trained weights
        weight_path: path of network weights
        embedding_layer: layer for embedding outputs
    return:
        compiled network model
    """
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loss = categorical_crossentropy
    
    
    if load_weight:
        model_architecture.load_weights(weight_path)
        intermediate_layer_model = Model(inputs=model_architecture.input,
                                         outputs=model_architecture.get_layer(embedding_layer).output)
        intermediate_layer_model.compile(loss=loss,
                                         optimizer=opt,
                                         metrics=['accuracy'])   
        return intermediate_layer_model
    
    model_architecture.compile(loss=loss,
          optimizer=opt,
          metrics=['accuracy'])
      
    return model_architecture

#%%
"""
Main function
"""
# Load wav data
sr, y = io.read_audio_data(audio_path)
y = io.audio_pre_processing(y, sr=sr, sr_new=16000)

#%%
# Mel feat computation
mel_feat = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=160, n_mels=128)
mel_feat = mel_feat * 10000 + 1
inpt = librosa.power_to_db(mel_feat/10, ref=1.0)/10
print('mel-spectrogram in shape', inpt.shape)
# Acoustic feat extraction
acoustic_mean = np.empty((num_mfcc*3+6+1,0)) # 14 mfccs + 28 deltas + 6 others + label
mfcc_feat = librosa.feature.mfcc(y=y, 
                                 sr=16000, 
                                 n_mfcc=num_mfcc, 
                                 n_fft=1024, 
                                 hop_length=160, 
                                 n_mels=128)
mfcc_delta_1 = librosa.feature.delta(mfcc_feat)
mfcc_delta_2 = librosa.feature.delta(mfcc_delta_1)
rms = librosa.feature.rms(y=y, 
                          frame_length=1024, 
                          hop_length=160)
rms_delta_1 = librosa.feature.delta(rms)
zcr = librosa.feature.zero_crossing_rate(y=y, 
                                         frame_length=1024, 
                                         hop_length=160)
zcr_delta_1 = librosa.feature.delta(zcr)
f0 = librosa.yin(y=y, 
                  fmin=librosa.note_to_hz('C2'), 
                  fmax=librosa.note_to_hz('C7'), 
                  sr=16000, 
                  frame_length=1024, 
                  hop_length=160)
f0_delta_1 = librosa.feature.delta(f0)
f0, f0_delta_1 = f0.reshape((1, len(f0))), f0_delta_1.reshape((1, len(f0_delta_1)))

#%%
# Load labels
if target_file != 'noise':
    labels = pd.read_csv(label_path, header=None).values
else:
    labels = np.zeros((inpt.shape[1], 1), dtype=int)   # noise label: 0
# snippets segmentation for every (srate // hop_length) seg, here it is 1 sec per segment
srate, hop_length = 16000, 160
n = inpt.shape[1] // (srate // hop_length)

# save segments of mel-spectrogram in individual csv files
check_dirs.check_dir(save_csv_dir)
for seg_idx in range(n):
    label = labels[seg_idx][0]
    if extract_spec:
        seg = inpt[:, seg_idx * (srate // hop_length):(seg_idx+1) * (srate // hop_length)]
    # compute mean acoustic features
    mean = np.empty((1,0))
    for feat_type in [mfcc_feat, mfcc_delta_1, mfcc_delta_2, rms, rms_delta_1, zcr, zcr_delta_1, f0, f0_delta_1]:
        mean = np.append(mean, np.mean(feat_type[:, seg_idx * (srate // hop_length):(seg_idx+1) * (srate // hop_length)], axis=1))    
    mean = np.append(mean, label)
    acoustic_mean = np.hstack((acoustic_mean, mean.reshape((num_mfcc*3+6+1, 1))))
    # naming format: ./label_segx.csv, x is the segment idx      
    if extract_spec:    
        with open(save_csv_dir + str(label) + '_seg' + str(seg_idx) + ".csv", 'w', newline='') as csvfile:   # no gap in lines
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerows(seg)
        csvfile.close()

acoustic_mean = acoustic_mean.T

#%%    
# load mel-spectrogram and labels from the csv files
csv_list = [os.path.join(save_csv_dir, item) \
            for item in os.listdir(save_csv_dir) if item.endswith('.csv')]
# It is noted that the input feat and labels are ordered when being read again
feat, labels = load_csv(csv_list)
feat = np.asarray(feat)

#%%
#import model_Lukic

# embedding extraction
sne_feat_to_cnn = np.expand_dims(feat, axis=3)
model = model_Lukic.model_keras(SHAPE=(128, 100, 1), num_test_speakers = 100)
model_arch = model.architecture()
    
    #activation = {}
    #def get_activation(name):
    #    def hook(model, input, output):
    #        activation[name] = output.detach()
    #    return hook
    #
    #model = model_Lukic.model_torch(num_test_speakers=100)
    #model.fc3.register_forward_hook(get_activation('fc3'))
    #model = nn.DataParallel(model)
    #model.to('cpu')
    #model.load_state_dict(torch.load(torch_path, map_location=torch.device('cpu')))
    #model.eval()
    #sne_feat_to_cnn = torch.from_numpy(sne_feat_to_cnn.reshape((len(sne_feat_to_cnn), 1, 128, 100))).float()
    #embeddings = model(sne_feat_to_cnn).data.numpy()
    
intermediate_layer_model = model_compiler(model_arch, 
                                          load_weight=True, 
                                          weight_path=weight_path, 
                                          embedding_layer=embedding_layer)
embeddings = intermediate_layer_model.predict(sne_feat_to_cnn)

#%%
# feature visualization: compute the t-SNE scores
#if target_file != 'noise':
#    tsne = TSNE(n_components=2, random_state=0)
#    if mfcc:
#        tsne_feat = tsne.fit_transform(acoustic_mean[:, :48])
#    else:
#        tsne_feat = tsne.fit_transform(embeddings)
#    # scatter plot of tSNE
#    plt.figure(figsize=(8,5))
#    sns.scatterplot(
#        x=tsne_feat[:,0], y=tsne_feat[:,1],
#        hue=labels_in_str,
#        palette=sns.color_palette("hls", 3),
#        legend="full",
#        alpha=0.8)

#%%
# save embedding and mfcc features
check_dirs.check_dir(save_embedding_path)
check_dirs.check_dir(save_mfcc_path)

labels_to_save = np.asarray(labels).reshape((len(labels), 1))
embeddings_to_save = np.hstack((embeddings, labels_to_save))
with open(save_embedding_path + target_file + '.csv', 'w', newline='') as csvfile:   # no gap in lines
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerows(embeddings_to_save)
csvfile.close()
    
with open(save_mfcc_path + target_file + '.csv', 'w', newline='') as csvfile:   # no gap in lines
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerows(acoustic_mean)
csvfile.close()