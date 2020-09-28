#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:41:33 2020

@author: dawei
"""

"""
Foreground voice detection.

2 classes: voice vs background; 3 classes: wearer's voice vs non-wearers' voice vs background

"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.random import set_seed
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

import check_dirs

seed(0)
# keras seed
set_seed(0)

#%%
class model_2class():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def model(self):    
        model = Sequential()
        model.add(L.Conv1D(32, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same'))
        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
        model.add(L.Flatten())
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(1, activation='sigmoid'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=binary_crossentropy,
                      optimizer=opt,
                      metrics=['binary_accuracy'])
        return model
    
class model_3class():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def model(self):    
        model = Sequential()
        model.add(L.Conv1D(32, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same'))
        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
        model.add(L.Flatten())
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(3, activation='softmax'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
        return model
#%%  
"""
main func
"""
feature = 'mfcc'   # embedding or mfcc
mode = '3class'

sound_dir = './field_study/field_data/p0/mfcc_1s/120MFCC/'
sound_list = [os.path.join(sound_dir, item) \
            for item in os.listdir(sound_dir) if item.endswith('.csv')]
sound_data = []
for item in sound_list:
    audio_seg = pd.read_csv(item, delimiter=',', header=None).values
    sound_data.append(audio_seg)
sound_data = np.concatenate(sound_data, axis = 0)
if feature == 'embedding':
    feat_size = 1000
elif feature == 'mfcc':
    feat_size = 12
features, labels = sound_data[:, :feat_size], sound_data[:, -1]
# remove invalid test sounds
idx_valid_voice = np.where(labels != 'm')
labels = labels[idx_valid_voice]
features = features[idx_valid_voice]

# labels transform for consistent format
idx_wearer_voice = np.where(labels == '1')   # wearer
labels[idx_wearer_voice] = 1
idx_2 = np.where(labels == '2')   # non-wearer (physical and virtual)
labels[idx_2] = 2
idx_p = np.where(labels == 'p')
labels[idx_p] = 2
idx_t = np.where(labels == 't')
labels[idx_t] = 2
idx_noise = np.where(labels == 'b')   # non-vocal background
idx_noise_old = np.where(labels == 0)   # old labels of noise
# wearer: 1, non-wearer: 2, back:0
if mode == '3class':
    labels[idx_noise] = 0
    labels = labels.astype(int)
# wearer: 1, all others: 2
elif mode == '2class':
    labels[idx_noise] = 2
    labels[idx_noise_old] = 2
    labels = labels.astype(int)        

# min-max standization
features = (features - np.min(features)) / np.max(features)
features = np.expand_dims(features, axis=-1).astype(float)
labels = np.reshape(labels, (len(labels), 1))            
    
#%%
"""
training and validation

please leave only the best models of each fold in the folder before validation, 
so that they can be read in order and without error
"""

predict = True
save_model_path = './field_study/field_data/p0/models/'
check_dirs.check_dir(save_model_path)
batch_size = 128
kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
fold_no = 1
f1_per_fold, acc_per_fold, pre_per_fold, rec_per_fold = [], [], [], []
# training
if not predict:
    for train, test in kfold.split(features, labels):
          feat_train, labels_train = features[train], labels[train]
          feat_test, labels_test = features[test], labels[test]
          if mode == '2class':
              model_fold = model_2class(input_shape=(feat_size, 1)).model()
          elif mode == '3class':              
              model_fold = model_3class(input_shape=(feat_size, 1)).model()
              labels_train = tf.keras.utils.to_categorical(labels_train, num_classes=3, dtype=int)
              labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=3, dtype=int)
          # Fit data to model, only models with the best val acc (unbalanced) are saved
          model_fold.fit(feat_train, labels_train,
                      batch_size=batch_size,
                      epochs=20,
                      validation_data=(feat_test, labels_test),
                      verbose=2,
                      callbacks=[K.callbacks.ModelCheckpoint(save_model_path+"fold%01d-epoch_{epoch:02d}-acc_{val_accuracy:.4f}.h5" %fold_no, 
                                                             monitor='val_accuracy', 
                                                             verbose=0, 
                                                             save_best_only=True, 
                                                             save_weights_only=True, 
                                                             mode='auto', 
                                                             save_freq='epoch')])
          del model_fold
          fold_no = fold_no + 1
# validation                     
if predict:
      models = [os.path.join(save_model_path, item) \
                for item in os.listdir(save_model_path) if item.endswith('.h5')]
      for train, test in kfold.split(features, labels):
          feat_train, labels_train = features[train], labels[train]
          feat_test, labels_test = features[test], labels[test]
          if mode == '2class':
              model_pred = model_2class(input_shape=(feat_size, 1)).model()
          elif mode == '3class':
              model_pred = model_3class(input_shape=(feat_size, 1)).model()
              labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=3, dtype=int)
          model_pred.load_weights(models[fold_no - 1])
          # prediction
          pred = model_pred.predict(feat_test)
          pred = np.argmax(pred, axis=1)
          labels_test = np.argmax(labels_test, axis=1)
          # set threshold for binary classification
          if mode == '2class':
              idx_p = np.where(pred > 0.5)
              idx_n = np.where(pred <= 0.5)
              pred[idx_p] = 1
              pred[idx_n] = 2
          elif mode == '3class':
              pass
          f1_per_fold.append(f1_score(labels_test, pred, average = 'weighted') * 100)
          acc_per_fold.append(balanced_accuracy_score(labels_test, pred) * 100)
          pre_per_fold.append(precision_score(labels_test, pred, average = 'weighted'))
          rec_per_fold.append(recall_score(labels_test, pred, average = 'weighted'))  
          # initialize
          del model_pred
          fold_no = fold_no + 1
      f1 = np.mean(f1_per_fold)
      acc = np.mean(acc_per_fold)
      precision = np.mean(pre_per_fold)
      recall = np.mean(rec_per_fold)
      print('f1: ', f1, '\n acc: ', acc,
            '\n precision: ', precision, '\n recall: ', recall)