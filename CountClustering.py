#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
from scipy.spatial import distance
import os


# In[79]:


# Import data

filename_1 = "pre_outdoor_embedding.csv "
filename_2 = "pre_outdoor_mfcc.csv "
filename_3 = "pre_TV_embedding.csv"
filename_4 = "pre_TV_mfcc.csv"
data_1 = np.array(pd.read_csv(filename_1, header = None))
data_2 = np.array(pd.read_csv(filename_2, header = None))
data_3 = np.array(pd.read_csv(filename_3, header = None))
data_4 = np.array(pd.read_csv(filename_4, header = None))
X_1 = data_1[:, :-1]
Y_1 = data_1[:, -1]
X_2 = data_2[:, :-1]
Y_2 = data_2[:, -1]
X_3 = data_3[:, :-1]
Y_3 = data_3[:, -1]
X_4 = data_4[:, :-1]
Y_4 = data_4[:, -1]


# In[80]:


def getDistance(seg_1, seg_2):
#    return np.linalg.norm((seg_1 - seg_2), ord = 1)
    return scipy.spatial.distance.cosine(seg_1, seg_2)
    
def unsupervisedalgo(mfcc, Constants_mfcc_dist_diff_un):

    new_mfcc = np.zeros((mfcc.shape[0], mfcc.shape[1]))
    new_in_count = np.zeros((mfcc.shape[0], ))
    new_mfcc[0, :] = mfcc[0, :]
    speaker_count = 1
    new_in_count[0] = 1

    for i in range(1, mfcc.shape[0], 1):
        diff_count = 0
        for j in range(speaker_count):
            mfcc_dist = getDistance(mfcc[i], new_mfcc[j])
            if (mfcc_dist >= Constants_mfcc_dist_diff_un): 
                diff_count = diff_count + 1;
            else:
                new_mfcc[j] = (new_mfcc[j]*new_in_count[j]+mfcc[i])/(new_in_count[j]+1)
                new_in_count[j] += 1
                break
        if (diff_count == speaker_count):
            new_mfcc[speaker_count, :] = mfcc[i, :]
            new_in_count[speaker_count] += 1
            speaker_count = speaker_count + 1
    return speaker_count, new_in_count


# In[81]:


print(unsupervisedalgo(X_1, 0.5))
print(unsupervisedalgo(X_2, 0.05))

print(unsupervisedalgo(X_3, 0.5))
print(unsupervisedalgo(X_4, 0.05))


# In[ ]:




