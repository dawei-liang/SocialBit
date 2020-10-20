# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:57:36 2020

@author: zifan; modifier: dawei
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, classification_report
from matplotlib import pyplot as plt
import seaborn as sn
import itertools

#%%
class distance_based_classifier_two_class():

    def __init__(self, distance_metric = 'cosine_similarity', threshold = 0.7):
        self.distance_metric = distance_metric
        self.threshold = threshold

    def fit(self, vecs):
        '''Take the average of all the wearer speech vector as characteristic vector
        args:
            vecs -- [np.ndarray] numpy array in shape (num_vector, len_vector)
        '''
        self.chara_vec = np.sum(vecs, axis = 0)/vecs.shape[0]

    def predict(self, vecs, threshold = None):

        if self.distance_metric == 'cosine_similarity':
            dis = cosine_similarity(vecs, np.expand_dims(self.chara_vec, 0)).reshape(-1)
        else:
            raise KeyError('distance metric: %s does not exist' %(self.distance_metric))

        pred_labels = np.ones(len(dis))
        if not threshold:
            idx_os_ns =  np.where(dis <= self.threshold)
            pred_labels[idx_os_ns] = 2
            return pred_labels
        else:
            idx_os_ns =  np.where(dis <= threshold)
            pred_labels[idx_os_ns] = 2
            return pred_labels
    def compute_distance(self, vecs):

        if self.distance_metric == 'cosine_similarity':
            dis = cosine_similarity(vecs, np.expand_dims(self.chara_vec, 0)).reshape(-1)
        else:
            raise KeyError('distance metric: %s does not exist' %(self.distance_metric))

        return dis

class distance_based_classifier_three_class():
    """
    Classification by using cosine distance metric
    """
    
    def __init__(self, distance_metric = 'cosine_similarity', threshold = [0.25, 0.2], feat_type = None):
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.feat_type = feat_type

    def fit(self, vecs):
        self.chara_vec = np.sum(vecs, axis = 0)/vecs.shape[0]

    def predict(self, vecs, threshold = None):

        if self.distance_metric == 'cosine_similarity':
            dis = cosine_similarity(vecs, np.expand_dims(self.chara_vec, 0)).reshape(-1)
        else:
            raise KeyError('distance metric: %s does not exist' %(self.distance_metric))

        pred_labels = np.zeros(len(dis))
        if not threshold:
            if feat_type == 'embedding':
                idx_ws =  np.where(dis > self.threshold[0])
                idx_os =  np.where(dis < self.threshold[1])
                pred_labels[idx_ws] = 1
                pred_labels[idx_os] = 2
            elif feat_type == 'mfcc':
                # the initial array for mfcc is [2,...,2]
                pred_labels = pred_labels + 2
                idx_ws =  np.where(dis > self.threshold[0])
                idx_b =  np.where(dis < self.threshold[1])
                pred_labels[idx_ws] = 1
                pred_labels[idx_b] = 0
            return pred_labels
        else:
            if feat_type == 'embedding':
                idx_ws =  np.where(dis > threshold[0])
                idx_os =  np.where(dis < threshold[1])
                pred_labels[idx_ws] = 1
                pred_labels[idx_os] = 2
            elif feat_type == 'mfcc':
                pred_labels = pred_labels + 2
                idx_ws =  np.where(dis > self.threshold[0])
                idx_os =  np.where(dis < self.threshold[1])
                pred_labels[idx_ws] = 1
                pred_labels[idx_b] = 0
            return pred_labels
    def compute_distance(self, vecs):

        if self.distance_metric == 'cosine_similarity':
            dis = cosine_similarity(vecs, np.expand_dims(self.chara_vec, 0)).reshape(-1)
        else:
            raise KeyError('distance metric: %s does not exist' %(self.distance_metric))

        return dis

def load_features(feat_path, feat_type):
    """
    Func to load the features as reference voice
    Args:
        feat_path
        feat_type: 'embedding' or 'mfcc'
    return:
        features: [# samples, feat dimension]
        labels: [# samples]
    """
    if feat_type == 'embedding':
        feat_size = 1000
    else:
        feat_size = 46   # original 48D, we did not use f0 and delta f0 as they do not help
    a = pd.read_csv(feat_path, delimiter=',', header=None).values

    return a[:, :feat_size], a[:, -1]

def load_multiple_features(feat_path, feat_type):
    """
    Same func as above, except it is to load test features and can loop for multiple files
    """
    if feat_type == 'embedding':
        feat_size = 1000
    else:
        feat_size = 46
    f = []
    for fp in feat_path:
        a = pd.read_csv(fp, delimiter=',', header=None).values
        f.append(a)
    f = np.concatenate(f, axis = 0)

    return f[:, :feat_size], f[:, -1]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Func to plot confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground True Sounds')
    plt.xlabel('Predicted Sounds')
#%%
    
'''Main'''

distsance_metric = 'cosine_similarity'
feat_type = 'embedding'   # use embedding or mfcc
classes = 2   # 2 or 3


train_feat_path = './field_study/field_data/P2/%s_1s/reading.csv' %feat_type   # ref voice
test_feat_path = ['./field_study/field_data/P2/%s_1s/call.csv' %feat_type,
                 './field_study/field_data/P2/%s_1s/dinner.csv' %feat_type,
                 './field_study/field_data/P2/%s_1s/game.csv' %feat_type,
                 './field_study/field_data/P2/%s_1s/outdoor.csv' %feat_type,
                 './field_study/field_data/P2/%s_1s/TV.csv' %feat_type]   # test voice  
train_embedding, train_labels = load_features(train_feat_path, feat_type = feat_type)
test_embedding, test_labels = load_multiple_features(test_feat_path, feat_type = feat_type)

# remove invalid training sounds
idx_valid_voice = np.where(train_labels == '1')
train_labels_filtered = train_labels[idx_valid_voice]
train_vec_ws = train_embedding[idx_valid_voice]
# remove invalid test sounds
idx_valid_voice_m = np.where(test_labels != 'm')
idx_valid_voice_x = np.where(test_labels != 'x')
idx_valid_voice = np.intersect1d(idx_valid_voice_m, idx_valid_voice_x)
test_labels_filtered = test_labels[idx_valid_voice]
test_embedding_filtered = test_embedding[idx_valid_voice]

#%%
# plot distributions of the embedding values 
# select embedding for each class
idx_1 = np.where(test_labels_filtered == '1')
embedding_1 = test_embedding_filtered[idx_1]
embedding_1_mean = np.mean(embedding_1, axis=0).astype(float)
embedding_1_max = np.max(embedding_1, axis=0).astype(float)

idx_2 = np.where(test_labels_filtered == '2')
embedding_2 = test_embedding_filtered[idx_2]
embedding_2_mean = np.mean(embedding_2, axis=0).astype(float)
embedding_2_max = np.max(embedding_2, axis=0).astype(float)

idx_p = np.where(test_labels_filtered == 'p')
embedding_p = test_embedding_filtered[idx_p]
idx_t = np.where(test_labels_filtered == 't')
embedding_t = test_embedding_filtered[idx_t]
embedding_p_t = np.vstack((embedding_p, embedding_t))
embedding_p_t_mean = np.mean(embedding_p_t, axis=0).astype(float)
embedding_p_t_max = np.max(embedding_p_t, axis=0).astype(float)

idx_b = np.where(test_labels_filtered == 'b')
embedding_b = test_embedding_filtered[idx_b]
idx_n = np.where(test_labels_filtered == 0)
embedding_n = test_embedding_filtered[idx_n]
embedding_b_n = np.vstack((embedding_b, embedding_n))
embedding_b_n_mean = np.mean(embedding_b_n, axis=0).astype(float)
embedding_b_n_max = np.mean(embedding_b_n, axis=0).astype(float)

plt.figure(1)
plt.hist(embedding_1_mean, bins=50, range=(0,1), density=True)
plt.xlabel('Embedding values')
plt.ylabel('Density')
plt.title('Wearer')
plt.show()

plt.figure(2)
plt.hist(embedding_2_mean, bins=50, range=(0,1), density=True)
plt.xlabel('Embedding values')
plt.ylabel('Density')
plt.title('Non-wearer (Human)')
plt.show()

plt.figure(3)
plt.hist(embedding_p_t_mean, bins=50, range=(0,1), density=True)
plt.xlabel('Embedding values')
plt.ylabel('Density')
plt.title('Non-wearer (Virtual)')
plt.show()

plt.figure(4)
plt.hist(embedding_b_n_mean, bins=50, range=(0,1), density=True)
plt.xlabel('Embedding values')
plt.title('Background')
plt.ylabel('Density')
plt.show()

plt.figure(5)
plt.hist([embedding_1_max, embedding_2_max, embedding_p_t_max, embedding_b_n_max], bins=50, range=(0,7), density=True)
plt.legend(["Wearer", "Other human voice", "Other virtual voice", "Background"])
plt.xlabel('Max embedding values')
plt.ylabel('Density')
plt.show()

#%%
# Change training sound labels to a consistent format
for i in range(len(train_labels_filtered)):
    train_labels_filtered[i] = 1 
train_labels_filtered = train_labels_filtered.astype(int)
# Change test sound labels to a consistent format
idx_wearer_voice = np.where(test_labels_filtered == '1')   # wearer
test_labels_filtered[idx_wearer_voice] = 1
idx_other_voice = np.where(test_labels_filtered == '2')   # non-wearer (physical and virtual)
test_labels_filtered[idx_other_voice] = 2
idx_other_voice = np.where(test_labels_filtered == 'p')
test_labels_filtered[idx_other_voice] = 2
idx_other_voice = np.where(test_labels_filtered == 't')
test_labels_filtered[idx_other_voice] = 0
idx_noise = np.where(test_labels_filtered == 'b')   # non-vocal background
idx_noise_old = np.where(test_labels_filtered == 0)   # old labels of noise
if classes == 3:
    test_labels_filtered[idx_noise] = 0
elif classes == 2:
    test_labels_filtered[idx_noise] = 2
    test_labels_filtered[idx_noise_old] = 2
test_labels_filtered = test_labels_filtered.astype(int)   

#%%
if feat_type == 'embedding':
    if classes == 2:
        threshold = 0.2276   # P0: 0.3222; P2: 0.1545
    elif classes == 3:
        threshold = [0.2458, 0.1213]   # P0: [0.3221, 0.2905]; P2: [0.1725, 0.1030]
elif feat_type == 'mfcc':
    if classes == 2:
        threshold = 0.9888   # P5: 0.9888 
    elif classes == 3:
        threshold = [0.9922, 0.9819] # P5: [0.9922, 0.9819] 
    
# fit and predict
if classes == 2:  
    clf = distance_based_classifier_two_class(distance_metric = 'cosine_similarity', 
                                                threshold = threshold)   
elif classes == 3:
    clf = distance_based_classifier_three_class(distance_metric = 'cosine_similarity', 
                                                threshold = threshold,
                                                feat_type = feat_type)
clf.fit(train_vec_ws[:, :])
pred_labels = clf.predict(test_embedding_filtered[:,:])
acc = balanced_accuracy_score(test_labels_filtered, pred_labels)
f1 = f1_score(test_labels_filtered, pred_labels, average = 'macro')

# print classification result
if classes == 2:
    print('Using threshold %.4f: balanced accuracy %f macro f1 %f' %(threshold,  
                                                                        acc, f1))
elif classes == 3:
    print('Using threshold %.4f, %.4f: balanced accuracy %f macro f1 %f' %(threshold[0], 
                                                                              threshold[1], 
                                                                              acc, f1))
plt.figure(6, figsize = (6,4))
a = confusion_matrix(test_labels_filtered, pred_labels)

if classes == 2:
    plot_confusion_matrix(a, classes=["Back + Other voice", "Wearer"], normalize=True,
                          title='Classification Results')
    plt.title('Confusion matrix, 2-Class (threshold %.3f)' %(threshold))
elif classes == 3:
    plot_confusion_matrix(a, classes=["Background", "Wearer", "Other voice"], normalize=True,
                          title='Classification Results')
    plt.title('Confusion matrix, 3-Class (thresholds %.3f, %.3f)' %(threshold[0], threshold[1]))
plt.show()

# Plot class-wise feature distributions
idx_ws = np.where(test_labels_filtered == 1)
idx_os = np.where(test_labels_filtered == 2)
idx_ns = np.where(test_labels_filtered == 0)

vec_ws = test_embedding_filtered[idx_ws]
vec_os = test_embedding_filtered[idx_os]
vec_ns = test_embedding_filtered[idx_ns]

dis_ws = clf.compute_distance(vec_ws)
dis_os = clf.compute_distance(vec_os)
if classes == 3:
    dis_ns = clf.compute_distance(vec_ns)


plt.figure(7)
if classes == 2:
    plt.hist([dis_ws, dis_os], bins=50, density  = True)
    plt.legend(["Wearer", "Background + Other voice"])
    plt.xlabel('Cosine similarity')
    plt.ylabel('Density')
    plt.show()
elif classes == 3:
    plt.hist([dis_ws, dis_os,dis_ns], bins=50, density  = True)
    plt.legend(["Wearer", "Other voice", "Background"])
    plt.xlabel('Cosine similarity')
    plt.ylabel('Density')
    plt.show()   
    