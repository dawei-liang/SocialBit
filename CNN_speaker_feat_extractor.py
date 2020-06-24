#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:47:36 2020

@author: dawei

CNN-based extractor of speaker representations (embedding).
when 'test': input format = csv mel-spec(1s) as (subject id)_(sentence id)_(seg id)
"""

import os
import pandas as pd
import numpy as np
import csv
from numpy.random import seed
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

import keras as K
from keras.utils import np_utils
from tensorflow import set_random_seed
import keras.layers as L
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

import check_dirs

seed(0)
set_random_seed(0)
#%%
def load_csv(csv_list):
    """
    args:
        list of csv path
    return:
        data and labels of two lists 
    """
    feat, labels = [], []
    for csv_item in sorted(csv_list):
        spectrogram = pd.read_csv(csv_item, header=None).values
        # naming format: /../sub_sentence_seg.csv
        sub = csv_item.split('/')[-1].split('_')[0]
        print('loaded spectrogram shape:', spectrogram.shape, 'sub:', sub)
        feat.append(spectrogram)
        labels.append(sub)
    return feat, labels

#%%
def str_to_int(training_labels, test_labels):
    """
    args:
        training_labels, test_labels: list of training/test labels in string
    return:
        list of training/test labels in integer
    """
    labels = training_labels + test_labels
    seperate_point = len(training_labels)
    str_int_mapping, label_in_int = {}, []
    class_idx = 0   # index of distinct classes
    for label in labels:
        # if class not in str_int_mapping, use a new class_idx and add it to dict
        if label not in str_int_mapping:
            label_in_int.append(class_idx)
            str_int_mapping[label] = class_idx
            class_idx += 1
        # if class already in str_int_mapping, simply re-use the class_idx
        else:
            label_in_int.append(str_int_mapping[label])
            
    return label_in_int[:seperate_point], label_in_int[seperate_point:]   # training, test

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
    # One-hot encoding (# of instances, # of distinct classes)
    labels = np_utils.to_categorical(labels, class_size)
    return data, labels

#%%
def architecture(SHAPE, num_test_speakers):
    x_inputs = L.Input(shape=SHAPE)    
    x = x_inputs #inputs is used by the line "Model(inputs, ... )" below

    x = L.Conv2D(32, (4,4), strides=(2,2), activation='relu', padding='same', name='L1')(x)
    x = L.Dropout(0.3, name='L2')(x)
    x = L.MaxPooling2D(pool_size=(4,4), strides=(2,2), name='L3')(x) 
    
    print("L3:", x.shape)
    
    x = L.Conv2D(128, (4,4), strides=(2,2), activation='relu', padding='same', name='L4')(x)    
    x = L.Dropout(0.3, name='L5')(x)
    x = L.MaxPooling2D(pool_size=(4,4), strides=(2,2), name='L6')(x)
    
    print("L6:", x.shape)
    
    x = L.Conv2D(128, (4,4), strides=(2,2), activation='relu', padding='same', name='L7')(x)    
    x = L.Dropout(0.3, name='L8')(x)
    print("L8:", x.shape)
    
    
    x = L.Flatten()(x)
    x = L.Dense(units=int(10*num_test_speakers), activation='relu', name='L9')(x)
    x = L.Dropout(0.5, name='L10')(x)
    x = L.Dense(units=int(5*num_test_speakers), activation='relu', name='L11')(x)
    x = L.Dropout(0.5, name='L12')(x)
    x = L.Dense(units=num_test_speakers, activation='linear', name='L13')(x)
    print ("L13: ", x.shape)
    # Output layer
    x_output = L.Softmax(name='L14')(x)
    
    model = Model(inputs=x_inputs, outputs=x_output)

    return model

#%%
def model_compiler(model, load_weight=False, weight_path=None, embedding_layer=None):
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
    loss = categorical_crossentropy
    
    if load_weight:
        model.load_weights(weight_path)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(embedding_layer).output)
        intermediate_layer_model.compile(loss=loss,
                                         optimizer=opt,
                                         metrics=['accuracy'])   
        return intermediate_layer_model
    
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])   
    return model

            
#%%
training_path, test_path = './TIMIT_feat/training', './TIMIT_feat/test'   # path to load tspectrograms
# obtain path to the training/test csv features
#sub_list = os.listdir(training_path)[:400]
#training_wav = [os.path.join(training_path, item) \
 #               for item in os.listdir(training_path) for x in sub_list if x.split('/')[-1].split('_')[0] in item]
#test_wav = [os.path.join(test_path, item) \
 #           for item in os.listdir(test_path) for x in sub_list if x.split('/')[-1].split('_')[0] in item]

training_wav = [os.path.join(training_path, item) \
                for item in os.listdir(training_path)]
test_wav = [os.path.join(test_path, item) \
            for item in os.listdir(test_path)] 

# Load training/test features and subject labels
training_feat, training_labels = load_csv(training_wav)
test_feat, test_labels = load_csv(test_wav)
training_feat, test_feat = np.asarray(training_feat), np.asarray(test_feat)
# convert labels from str to int
training_labels_int, test_labels_int = str_to_int(training_labels, test_labels)
print('data shape:', training_feat[0].shape, test_feat[0].shape)
distinct_speakers = set(training_labels + test_labels)
print('# of distinct speakers: %d' %len(distinct_speakers))
# reshpe data/labels for CNN
training_feat_to_CNN, training_labels_to_CNN = reshape_data_labels(training_feat, training_labels_int, class_size=len(set(training_labels_int)))
test_feat_to_CNN, test_labels_to_CNN = reshape_data_labels(test_feat, test_labels_int, class_size=len(set(test_labels_int)))
print('class size with on-hot, training:', training_labels_to_CNN.shape[1])
print('class size with on-hot, test:', test_labels_to_CNN.shape[1])


#%%
mode = 'test'
save_model_path = './TIMIT_feat/models/'
save_embedding_path = './TIMIT_feat/TIMIT_embedding_from_CNN/'
embedding_of_real_path = './embedding_of_real/'
weight_path = './TIMIT_feat/models/630sub-CNN_feat_extractor-epoch_110-val_0.5372.hdf5'
embedding_layer = 'L13'
check_dirs.check_dir(save_model_path)
check_dirs.check_dir(embedding_of_real_path)
check_dirs.check_dir(save_embedding_path)

if mode == 'training':
    model = architecture(SHAPE=(128, 100, 1), num_test_speakers = len(distinct_speakers))
    model = model_compiler(model, load_weight=False)
    model.fit(training_feat_to_CNN, training_labels_to_CNN,   
    batch_size=128,
    epochs=500,
    verbose=2,
    validation_data = (test_feat_to_CNN, test_labels_to_CNN),
    shuffle=True,
    callbacks=[EarlyStopping(monitor='val_acc', patience=50, mode='auto'),
            K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr = 0),
            K.callbacks.ModelCheckpoint(save_model_path+"CNN_feat_extractor-epoch_{epoch:02d}-val_{val_acc:.4f}.hdf5", 
                         monitor='val_acc', 
                         verbose=0, 
                         save_best_only=True, 
                         save_weights_only=False, 
                         mode='auto', 
                         period=1)])

elif mode == 'test':
    '''plot t-SNE feature distributions for given test data'''
    ###
    training_path = './recordings/preliminary/conv_quite'
    sne_wav = [os.path.join(training_path, item) \
            for item in os.listdir(training_path) for x in ['0', '1', '2'] if item.startswith(x)]
    ###
    #sne_wav = [os.path.join(training_path, item) \
     #       for item in os.listdir(training_path) for x in ['FADG0', 'FAEM0', 'FAJW0'] if x in item]   
    sne_feat, sne_labels = load_csv(sne_wav)
    for i in range(len(sne_labels)):
        if sne_labels[i] == '0':
            sne_labels[i] = 'transition'
        else:
            sne_labels[i] = 'speaker' + sne_labels[i]
    sne_feat = np.asarray(sne_feat)
    
    sne_feat_to_cnn = np.expand_dims(sne_feat, axis=3)
    model = architecture(SHAPE=(128, 100, 1), num_test_speakers = len(distinct_speakers))
    intermediate_layer_model = model_compiler(model, 
                                              load_weight=True, 
                                              weight_path=weight_path, 
                                              embedding_layer=embedding_layer)
    embeddings = intermediate_layer_model.predict(sne_feat_to_cnn)
    # compute t-SNE scores
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_new = tsne.fit_transform(embeddings)
    # scatter plot of tSNE
    plt.figure(figsize=(8,5))
    sns.scatterplot(
        x=embeddings_new[:,0], y=embeddings_new[:,1],
        hue=sne_labels,
        palette=sns.color_palette("hls", 3),
        legend="full",
        alpha=0.8)

elif mode == 'TIMIT_extraction':
    '''Extract and save embedding features for the TIMIT data (for later supervised usage)'''
    # use all subjects from training/test
    feat_to_be_extracted = np.vstack((training_feat_to_CNN, test_feat_to_CNN))
    feat_labels = training_labels + test_labels
    
    model = architecture(SHAPE=(128, 100, 1), num_test_speakers = len(distinct_speakers))
    intermediate_layer_model = model_compiler(model, 
                                              load_weight=True, 
                                              weight_path=weight_path, 
                                              embedding_layer=embedding_layer)
    pred = intermediate_layer_model.predict(feat_to_be_extracted)
    # save csv for each subject
    for sub in distinct_speakers:
        sub_embedding = []
        for i in range(len(feat_labels)):
            if feat_labels[i] == sub:
                sub_embedding.append(pred[i])
        sub_embedding = np.asarray(sub_embedding)
        embedding_path = os.path.join(save_embedding_path + sub)
        with open(embedding_path + ".csv", 'w', newline='') as csvfile:   # no gap between lines
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerows(sub_embedding)
        csvfile.close()

elif mode == 'preliminary_extraction':
    '''Extract and save embedding features for the preliminary data (for later supervised usage)'''
    # use all subjects from training/test
    training_path = './recordings/preliminary/mono_outdoor'
    sne_wav = [os.path.join(training_path, item) for item in os.listdir(training_path)]
    ###
    #sne_wav = [os.path.join(training_path, item) \
     #       for item in os.listdir(training_path) for x in ['FADG0', 'FAEM0', 'FAJW0'] if x in item]   
    pre_feat, pre_labels = load_csv(sne_wav)
    pre_feat = np.asarray(pre_feat)
    pre_feat_to_cnn = np.expand_dims(pre_feat, axis=3)
    
    model = architecture(SHAPE=(128, 100, 1), num_test_speakers = len(distinct_speakers))
    intermediate_layer_model = model_compiler(model, 
                                              load_weight=True, 
                                              weight_path=weight_path, 
                                              embedding_layer=embedding_layer)
    pred = intermediate_layer_model.predict(pre_feat_to_cnn)
    # save csv for each the session
    with open(embedding_of_real_path + "mono_outdoor0.csv", 'w', newline='') as csvfile:   # no gap between lines
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(pred)
    csvfile.close()
#%%
#for temp in training_wav:
#    if temp.split('/')[-1].split('_')[0] in ['FADG0', 'FAEM0', 'FAJW0']:
#        print(True)