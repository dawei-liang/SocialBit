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

import audio_utils.audio_io as io
import check_dirs
import model_Lukic

target_file = '8_1'
audio_path = './aaa/fine/%s.wav' % target_file # path of the audio clip
label_path = './aaa/fine/%s.csv' % target_file # path of the csv
save_csv_dir = './aaa/spectrogram_1s/%s/' % target_file # dir to save segments of spectrogram

weight_path = './CNN_feat_extractor-epoch_20-val_0.3729.hdf5' # path to load weights

# path and file names to save the computed embedding/mfcc features
save_embedding_path = './aaa/embedding_1s/'
save_mfcc_path = './aaa/mfcc_1s/'

embedding_layer = 'L9' # output embedding layer, can be 9, 11, 13
mfcc = True # whether to plot and save mfcc


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
# Mel feat computation
mel_feat = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=160, n_mels=128)
mel_feat = mel_feat * 10000 + 1
inpt = librosa.power_to_db(mel_feat/10, ref=1.0)/10
print('mel-spectrogram in shape', inpt.shape)
if mfcc:
    mfcc_mean = np.empty((13,0)) # 12 mfccs + label
    mfcc_feat = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=12, n_fft=1024, hop_length=160, n_mels=128)
# Load labels
labels = pd.read_csv(label_path, header=None).values
# snippets segmentation for every (srate // hop_length) seg, here it is 1 sec per segment
srate, hop_length = 16000, 160
n = inpt.shape[1] // (srate // hop_length)
# save segments of mel-spectrogram in individual csv files
check_dirs.check_dir(save_csv_dir)
for seg_idx in range(n):
    label = labels[seg_idx][0]
    seg = inpt[:, seg_idx * (srate // hop_length):(seg_idx+1) * (srate // hop_length)]
    if mfcc:
        mean = np.mean(mfcc_feat[:, seg_idx * (srate // hop_length):(seg_idx+1) * (srate // hop_length)], axis=1)
        mean = np.append(mean, label)
        mfcc_mean = np.hstack((mfcc_mean, mean.reshape((13, 1))))
    # naming format: ./label_segx.csv, x is the segment idx          
    with open(save_csv_dir + str(label) + '_seg' + str(seg_idx) + ".csv", 'w', newline='') as csvfile:   # no gap in lines
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(seg)
    csvfile.close()
if mfcc:
    mfcc_mean = mfcc_mean.T

#%%    
# load mel-spectrogram and labels from the csv files
csv_list = [os.path.join(save_csv_dir, item) \
            for item in os.listdir(save_csv_dir) if item.endswith('.csv')]
# It is noted that the input feat and labels are ordered when being read again
feat, labels = load_csv(csv_list)
labels_in_str = []
if mfcc:
    labels_to_be_converted = mfcc_mean[:, -1]
else:
    labels_to_be_converted = labels
for i in range(len(labels_to_be_converted)):
    if labels_to_be_converted[i] == '0':
        labels_in_str.append('irrelevant')
    else:
        labels_in_str.append('speaker' + str(labels_to_be_converted[i]))
feat = np.asarray(feat)

#%%
# embedding extraction
sne_feat_to_cnn = np.expand_dims(feat, axis=3)
model = model_Lukic.model_keras(SHAPE=(128, 100, 1), num_test_speakers = 100)
model_arch = model.architecture()
intermediate_layer_model = model_compiler(model_arch, 
                                          load_weight=True, 
                                          weight_path=weight_path, 
                                          embedding_layer=embedding_layer)
embeddings = intermediate_layer_model.predict(sne_feat_to_cnn)

#%%
# feature visualization: compute the t-SNE scores
tsne = TSNE(n_components=3, random_state=0)
if mfcc:
    tsne_feat = tsne.fit_transform(mfcc_mean)
else:
    tsne_feat = tsne.fit_transform(embeddings)
# scatter plot of tSNE
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=tsne_feat[:,0], y=tsne_feat[:,1],
    hue=labels_in_str,
    palette=sns.color_palette("hls", 4),
    legend="full",
    alpha=0.8)

#%%
# save embedding and mfcc features
check_dirs.check_dir(save_embedding_path)
check_dirs.check_dir(save_mfcc_path)
if not mfcc:
    labels_to_save = np.asarray(labels).reshape((len(labels), 1))
    embeddings_to_save = np.hstack((embeddings, labels_to_save))
    with open(save_embedding_path + target_file + '.csv', 'w', newline='') as csvfile:   # no gap in lines
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(embeddings_to_save)
    csvfile.close()
else:
    mfcc_to_save = mfcc_mean[:, :-1]
    
    with open(save_mfcc_path + target_file + '.csv', 'w', newline='') as csvfile:   # no gap in lines
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(mfcc_mean)
    csvfile.close()