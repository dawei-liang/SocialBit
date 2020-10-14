#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:41:33 2020

@author: dawei
"""

"""
Foreground voice detection.

2 classes: voice vs background; 
3 classes: wearer's voice vs non-wearers' voice vs background

"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

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
    
class model_2class_2D():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def model(self):    
        model = Sequential()
        model.add(L.Conv2D(32, strides=(2, 2), kernel_size=(4,4), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(L.Conv2D(64, strides=(2, 2), kernel_size=(4,4), activation='relu', padding='same'))
        model.add(L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(L.Conv2D(128, strides=(2, 2), kernel_size=(4,4), activation='relu', padding='same'))
        model.add(L.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(L.Flatten())
        model.add(L.Dense(512, activation='relu'))
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
        # naming format of the csv: /../activity\label_seg.csv
        label = csv_item.split('/')[-1].split('_')[0][-1]
        print('loaded spectrogram shape:', spectrogram.shape, 'label:', label)
        feat.append(spectrogram)
        labels.append(label)
    return feat, labels

#%%  
"""
main func
"""
feature = 'mfcc'   # embedding, mfcc, contextual
mode = '3class'
# obtain features and labels for embedding/mfcc features (1D)
if feature != 'contextual':
    sound_dir0 = './field_study/field_data/p0/%s_1s/' % feature
    sound_list0 = [os.path.join(sound_dir0, item) \
                for item in os.listdir(sound_dir0) if item.endswith('.csv')]
    sound_dir = './field_study/field_data/P2/%s_1s/' % feature
    sound_list = [os.path.join(sound_dir, item) \
                for item in os.listdir(sound_dir) if item.endswith('.csv')]
    sound_list.extend(sound_list0)
    
    sound_data = []
    for item in sound_list:
        audio_seg = pd.read_csv(item, delimiter=',', header=None).values
        sound_data.append(audio_seg)
    sound_data = np.concatenate(sound_data, axis = 0)
    if feature == 'embedding':
        feat_size = 1000
    elif feature == 'mfcc':
        feat_size = 46   # original 48D, we did not use f0 and delta f0 as they do not help
    features, labels = sound_data[:, :feat_size], sound_data[:, -1]

# obtain features and labels for contextual mfcc features (2D)
else:
    csv_list = [os.path.join(root, item) \
                for root, dirs, files in os.walk('./field_study/field_data/P2/%s_1s/' % feature) for item in files if item.endswith('.csv')]
    # It is noted that the input feat and labels are sorted when being read again
    feat, labels = load_csv(csv_list)
    features, labels = np.asarray(feat), np.asarray(labels)

#%%
# remove invalid test sounds
idx_valid_voice_m = np.where(labels != 'm')
idx_valid_voice_x = np.where(labels != 'x')
idx_valid_voice = np.intersect1d(idx_valid_voice_m, idx_valid_voice_x)
labels = labels[idx_valid_voice]
features = features[idx_valid_voice]

# labels transform for consistent format
# 2class: 1 for wearer, 0 for all other sounds;
# 3class: 1 for wearer, 2 for non-wearer voice; 0 for backgrounds
idx_wearer_voice = np.where(labels == '1')   # wearer
labels[idx_wearer_voice] = 1
idx_2 = np.where(labels == '2')   # non-wearer (physical and virtual)
labels[idx_2] = 2 if mode == '3class' else 0
idx_p = np.where(labels == 'p')
labels[idx_p] = 2 if mode == '3class' else 0
idx_t = np.where(labels == 't')   # tv
labels[idx_t] = 0
idx_noise = np.where(labels == 'b')   # non-vocal background
idx_noise_old = np.where(labels == 0)   # old labels of noise, not used in new data
# wearer: 1, non-wearer: 2, back:0
if mode == '3class':
    labels[idx_noise] = 0
    labels = labels.astype(int)
# wearer: 1, all others: 2
elif mode == '2class':
    labels[idx_noise] = 0
    labels[idx_noise_old] = 0
    labels = labels.astype(int)        

# min-max standization
features = (features - np.min(features)) / np.max(features)
# reshape for nn
features = np.expand_dims(features, axis=-1).astype(float)
labels = np.reshape(labels, (len(labels), 1))            
    
#%%
"""
training and validation (NN)

please leave only the best models of each fold in the folder before prediction/validation, 
so that they can be read in order and without error
"""

predict = True
save_model_path = './field_study/field_data/P2/models/'
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
              #model_fold = model_2class(input_shape=(feat_size, 1)).model()
              model_fold = model_2class_2D(input_shape=(48,100, 1)).model()
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
                      callbacks=[K.callbacks.ModelCheckpoint(save_model_path+"fold%01d-epoch_{epoch:02d}-acc_{val_binary_accuracy:.4f}.h5" %fold_no, 
                                                             monitor='val_binary_accuracy', 
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
                for item in sorted(os.listdir(save_model_path)) if item.endswith('.h5')]
      for train, test in kfold.split(features, labels):
          if fold_no>1: break
          feat_train, labels_train = features[train], labels[train]
          feat_test, labels_test = features[test], labels[test]
          if mode == '2class':
              #model_pred = model_2class(input_shape=(feat_size, 1)).model()
              model_pred = model_2class_2D(input_shape=(48,100, 1)).model()
          elif mode == '3class':
              model_pred = model_3class(input_shape=(feat_size, 1)).model()
              labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=3, dtype=int)
          model_pred.load_weights(models[fold_no - 1])
          # prediction
          pred = model_pred.predict(feat_test)
          if mode == '3class':
              pred = np.argmax(pred, axis=1)
              labels_test = np.argmax(labels_test, axis=1)
          # set threshold for binary classification
          if mode == '2class':
              idx_p = np.where(pred > 0.31)
              idx_n = np.where(pred <= 0.31)
              pred[idx_p] = 1
              pred[idx_n] = 0
          elif mode == '3class':
              pass
          acc_per_fold.append(balanced_accuracy_score(labels_test, pred) * 100)
          f1_per_fold.append(f1_score(labels_test, pred, average = 'macro') * 100)
          pre_per_fold.append(precision_score(labels_test, pred, average = 'macro') * 100)
          rec_per_fold.append(recall_score(labels_test, pred, average = 'macro') * 100)  
          # initialize
          del model_pred
          fold_no = fold_no + 1
      f1 = np.mean(f1_per_fold)
      acc = np.mean(acc_per_fold)
      precision = np.mean(pre_per_fold)
      recall = np.mean(rec_per_fold)
      print('f1: ', f1, '\n acc: ', acc,
            '\n precision: ', precision, '\n recall: ', recall)   

#%%
"""
training and validation with RF

"""
RF = False
if RF == True:
    seed(0)
    predict = True
    save_model_path = './field_study/field_data/P2/models/RF/'
    check_dirs.check_dir(save_model_path)
    kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    fold_no = 1
    f1_per_fold, acc_per_fold, pre_per_fold, rec_per_fold = [], [], [], []
    features = np.reshape(features, (features.shape[0], features.shape[1]))
    # training
    if not predict:
        for train, test in kfold.split(features, labels):
              feat_train, labels_train = features[train], labels[train]
              feat_test, labels_test = features[test], labels[test]
              clf = RandomForestClassifier(n_estimators=50, random_state = 0, n_jobs=-1)
              # Fit data to model, then save models
              clf.fit(feat_train, labels_train)
              para = clf.get_params()
              # name: RandomForestClassifier_fold%d_estimators_%d
              filename =  str(clf).split('(')[0] + '_fold%d_estimators_%d' %(fold_no+1, para['n_estimators'])
              joblib.dump(clf, save_model_path + filename)
              fold_no = fold_no + 1
    # validation                     
    if predict:
          models = [os.path.join(save_model_path, item) \
                    for item in sorted(os.listdir(save_model_path))]
          for train, test in kfold.split(features, labels):
              feat_train, labels_train = features[train], labels[train]
              feat_test, labels_test = features[test], labels[test]
              model_pred = joblib.load(models[fold_no - 1])
              # prediction
              pred = model_pred.predict(feat_test)
              # set threshold for binary classification
              if mode == '2class':
                  idx_p = np.where(pred > 0.5)
                  idx_n = np.where(pred <= 0.5)
                  pred[idx_p] = 1
                  pred[idx_n] = 2
              elif mode == '3class':
                  pass
              acc_per_fold.append(balanced_accuracy_score(labels_test, pred) * 100)
              f1_per_fold.append(f1_score(labels_test, pred, average = 'macro') * 100)
              pre_per_fold.append(precision_score(labels_test, pred, average = 'macro') * 100)
              rec_per_fold.append(recall_score(labels_test, pred, average = 'macro') * 100)  
              # initialize
              del model_pred
              fold_no = fold_no + 1
          f1 = np.mean(f1_per_fold)
          acc = np.mean(acc_per_fold)
          precision = np.mean(pre_per_fold)
          recall = np.mean(rec_per_fold)
          print('f1: ', f1, '\n acc: ', acc,
                '\n precision: ', precision, '\n recall: ', recall)