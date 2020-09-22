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

class distance_based_classifier_three_class():
    """
    Classification by using cosine distance metric
    """
    
    def __init__(self, distance_metric = 'cosine_similarity', threshold = [0.25, 0.2]):
        self.distance_metric = distance_metric
        self.threshold = threshold

    def fit(self, vecs):
        self.chara_vec = np.sum(vecs, axis = 0)/vecs.shape[0]

    def predict(self, vecs, threshold = None):

        if self.distance_metric == 'cosine_similarity':
            dis = cosine_similarity(vecs, np.expand_dims(self.chara_vec, 0)).reshape(-1)
        else:
            raise KeyError('distance metric: %s does not exist' %(self.distance_metric))

        pred_labels = np.zeros(len(dis))
        if not threshold:
            idx_ws =  np.where(dis > self.threshold[0])
            idx_os =  np.where(dis < self.threshold[1])
            pred_labels[idx_ws] = 1
            pred_labels[idx_os] = 2
            return pred_labels
        else:
            idx_ws =  np.where(dis > threshold[0])
            idx_os =  np.where(dis < threshold[1])
            pred_labels[idx_ws] = 1
            pred_labels[idx_os] = 2
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
        feat_size = 12
    a = pd.read_csv(feat_path, delimiter=',').values

    return a[:, :feat_size], a[:, -1]

def load_multiple_features(feat_path, feat_type):
    """
    Same func as above, except it is to load test features and can loop for multiple files
    """
    if feat_type == 'embedding':
        feat_size = 1000
    else:
        feat_size = 12
    f = []
    for fp in feat_path:
        a = pd.read_csv(fp, delimiter=',').values
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


if feat_type == 'embedding':
    threshold = [0.33, 0.278]
elif feat_type == 'mfcc':
    threshold = [0.996891, 0.994805]

train_feat_path = './field_study/field_data/p0/%s_1s/reading.csv' %feat_type   # ref voice
test_feat_path = ['./field_study/field_data/p0/%s_1s/call.csv' %feat_type,
                 './field_study/field_data/p0/%s_1s/dinner.csv' %feat_type,
                 './field_study/field_data/p0/%s_1s/game.csv' %feat_type]   # test voice  
train_embedding, train_labels = load_features(train_feat_path, feat_type = feat_type)
test_embedding, test_labels = load_multiple_features(test_feat_path, feat_type = feat_type)

# select reference (training) voice
train_idx_ws = np.where(train_labels == '1')
train_labels[train_idx_ws] = 1
train_vec_ws = train_embedding[train_idx_ws]

# remove invalid test sounds
idx_valid_voice = np.where(test_labels != 'm')
test_labels_filtered = test_labels[idx_valid_voice]
test_embedding_filtered = test_embedding[idx_valid_voice]
# Change test sound labels to a consistent format
idx_wearer_voice = np.where(test_labels_filtered == '1')   # wearer
test_labels_filtered[idx_wearer_voice] = 1
idx_other_voice = np.where(test_labels_filtered == '2')   # non-wearer (physical and virtual)
test_labels_filtered[idx_other_voice] = 2
idx_other_voice = np.where(test_labels_filtered == 'p')
test_labels_filtered[idx_other_voice] = 2
idx_other_voice = np.where(test_labels_filtered == 't')
test_labels_filtered[idx_other_voice] = 2
idx_noise = np.where(test_labels_filtered == 'b')   # non-vocal background
test_labels_filtered[idx_noise] = 0
true_labels = test_labels_filtered.copy().astype(int)

#%%
# fit and predict
clf = distance_based_classifier_three_class(distance_metric = 'cosine_similarity', threshold = threshold)
clf.fit(train_vec_ws)
pred_labels = clf.predict(test_embedding_filtered)

# print classification result
print('Using threshold %.4f, %.4f: balanced accuracy %f weighted f1 %f' %(threshold[0], threshold[1], 
                                                                          balanced_accuracy_score(true_labels, pred_labels),
                                                                          f1_score(true_labels, pred_labels, average = 'weighted')))
plt.figure(1, figsize = (6,4))
a = confusion_matrix(true_labels, pred_labels)
#df_cm = pd.DataFrame(a, index = ,
#                        columns = ["background", "wearer", "other voice"])
#sn.heatmap(df_cm, annot=True, cmap = plt.cm.Blues)
plot_confusion_matrix(a, classes=["background", "wearer", "other voice"], normalize=True,
                      title='Classification Results')
plt.title('confusion matrix (threshold %.3f, %.3f)' %(threshold[0], threshold[1]))
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
dis_ns = clf.compute_distance(vec_ns)

plt.figure(2)
plt.hist([dis_ws, dis_os,dis_ns], density  = True)
plt.legend(["wearer", "other voice", "background"])
plt.xlabel('cosine similarity')
plt.ylabel('density')
plt.show()
