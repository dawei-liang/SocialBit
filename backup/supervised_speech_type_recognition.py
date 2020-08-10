#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:41:49 2020

@author: dawei
"""

import os
import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

import keras as K
from keras.utils import np_utils
from tensorflow import set_random_seed
import keras.layers as L
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.losses import binary_crossentropy, categorical_crossentropy

import check_dirs

seed(0)
set_random_seed(0)

#%%
def reshape_data_labels(data, labels, class_size):
    """
    args:
        data, labels: data in array, labels in list
    return:
        reshape data and one-hot labels
    """
    # Reshape training data as (#,feature_size,1) for CNN
    data = np.expand_dims(data, axis=3)   
    # One-hot encoding (# of instances, # of distinct classes), not needed for binary class
    labels = np_utils.to_categorical(labels, class_size)
    return data, labels

#%%
def architecture(SHAPE, num_class_type):
    x_inputs = L.Input(shape=SHAPE)    
    x = x_inputs #inputs is used by the line "Model(inputs, ... )" below

    x = L.Conv2D(32, (4,4), strides=(2,2), activation='relu', padding='same', name='L1')(x)
    x = L.Dropout(0.1)(x)
    print("L1: ", x.shape)
    
    x = L.GlobalAveragePooling2D()(x)
     
    #x = L.Flatten()(x)
    
    x = L.Dense(units=64, activation='relu', name='L2')(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(units=32, activation='relu', name='L3')(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(units=32, activation='relu', name='L4')(x)
    x = L.Dropout(0.2)(x)
    
    #x_output = L.Dense(1, activation='sigmoid', name='L5')(x)    
    x_output = L.Dense(3, activation='softmax', name='L5')(x)
    print ("L5: ", x_output.shape)
    
    model = Model(inputs=x_inputs, outputs=x_output)

    return model

#%%

def model_compiler(model, load_weight=False, weight_path=None):
    """
    arg:
        model: model architecture
        load_weight: whether to load pre-trained weights
        weight_path: path of network weights
        embedding_layer: layer for embedding outputs
    return:
        compiled network model
    """
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #loss = binary_crossentropy
    loss = categorical_crossentropy

    
    if load_weight:
        model.load_weights(weight_path)
    model = Model(inputs=model.input, outputs=model.output)
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])   
    return model
#%%
speech_path = './conversational_feat/speech'
dyad_path = './conversational_feat/dyad'
group_path = './conversational_feat/group'
save_model_path = './conversational_feat/models'
check_dirs.check_dir(save_model_path)

mode = 'training'

speech_list = [os.path.join(speech_path, item) for item in os.listdir(speech_path)]
dyad_list = [os.path.join(dyad_path, item) for item in os.listdir(dyad_path)]
group_list = [os.path.join(group_path, item) for item in os.listdir(group_path)]

speech_feat, speech_labels = [], []
for item in speech_list:
    speech_feat_temp = pd.read_csv(item, header=None).values
    speech_feat.append(speech_feat_temp)
    speech_labels.append(0)
speech_feat = np.asarray(speech_feat)

dyad_feat, dyad_labels = [], []
for item in dyad_list:
    dyad_feat_temp = pd.read_csv(item, header=None).values
    dyad_feat.append(dyad_feat_temp)
    dyad_labels.append(1)
dyad_feat = np.asarray(dyad_feat)

group_feat, group_labels = [], []
for item in group_list:
    group_feat_temp = pd.read_csv(item, header=None).values
    group_feat.append(group_feat_temp)
    group_labels.append(2)
group_feat = np.asarray(group_feat)

#%%
feat, labels = np.vstack((speech_feat, dyad_feat)), speech_labels + dyad_labels
feat, labels = np.vstack((feat, group_feat)), labels + group_labels
training_data, test_data, training_labels, test_labels = train_test_split(feat, labels, test_size=0.2, random_state=0)
# reshpe data/labels for CNN
training_feat_to_CNN, training_labels_to_CNN = reshape_data_labels(training_data, training_labels, class_size=len(set(labels)))
test_feat_to_CNN, test_labels_to_CNN = reshape_data_labels(test_data, test_labels, class_size=len(set(labels)))
#training_data, test_data, training_labels, test_labels = 0,0,0,0


if mode == 'training':
    model = architecture(SHAPE=(630, None, 1), num_class_type = len(set(labels)))
    model = model_compiler(model, load_weight=False)
    model.fit(training_feat_to_CNN, training_labels_to_CNN,   
    batch_size=128,
    epochs=100,
    verbose=2,
    validation_data = (test_feat_to_CNN, test_labels_to_CNN),
    shuffle=True,
    callbacks=[EarlyStopping(monitor='val_acc', patience=20, mode='auto'),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr = 0),
            K.callbacks.ModelCheckpoint(save_model_path+"/CNN_feat_extractor-epoch_{epoch:02d}-val_{val_acc:.4f}.hdf5", 
                         monitor='val_acc', 
                         verbose=0, 
                         save_best_only=True, 
                         save_weights_only=True, 
                         mode='auto', 
                         period=1)])
    
elif mode == 'test':
    model = architecture(SHAPE=(630, 30, 1), num_class_type = len(set(labels)))
    model = model_compiler(model, load_weight=True, 
                           weight_path = save_model_path + '/CNN_630sub(3,24)(3,24)(3,24)30-epoch_24-val_0.9418.hdf5') 
    pred = np.argmax(model.predict(test_feat_to_CNN), axis = 1)
    # add threshold to the predictions
#    for i in range(len(pred)):
#        if pred[i] >= 0.5:
#            pred[i] = 1
#        else:
#            pred[i] = 0
    print(accuracy_score(test_labels, pred))
    print(recall_score(test_labels, pred, average='weighted'))
    print(precision_score(test_labels, pred, average='weighted'))
