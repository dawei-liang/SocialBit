import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot as plt
import seaborn as sn

class binary_cluster_classifier_three_class():

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

def load_features(feat_path):
    a = np.genfromtxt(feat_path, delimiter=',')

    return a[:, :1000], a[:, -1]

def load_multiple_features(feat_path):
    f = []
    for fp in feat_path:
        a = np.genfromtxt(fp, delimiter=',')
        f.append(a)
    f = np.concatenate(f, axis = 0)

    return f[:, :1000], f[:, -1]

## Three classes
## embedding_1s_7 as reference
## embedding_1s_8_1 + embedding_1s_8_2 + embedding_1s_noise

train_feat_path = './recordings/remove_0.2_rms/preliminary/features/embedding_1s/7.csv'
test_feat_path = ['./recordings/remove_0.2_rms/preliminary/features/embedding_1s/8_1.csv',
                 './recordings/remove_0.2_rms/preliminary/features/embedding_1s/8_2.csv',
                 './recordings/remove_0.2_rms/preliminary/features/embedding_1s/noise.csv',]
threshold = [0.25, 0.2]
distsance_metric = 'cosine_similarity'

train_embedding, train_labels = load_features(train_feat_path)
test_embedding, test_labels = load_multiple_features(test_feat_path)
# Change label 3 to 2, use 2 as non-wearer label
idx_3 = np.where(test_labels == 3)
test_labels[idx_3] = 2

train_idx_ws = np.where(train_labels == 1)
train_vec_ws = train_embedding[train_idx_ws]

true_labels = test_labels.copy()

clf = binary_cluster_classifier_three_class(distance_metric = 'cosine_similarity', threshold = threshold)
clf.fit(train_vec_ws)
pred_labels = clf.predict(test_embedding)

# print classification result
print('Using threshold %.4f, %.4f: accuracy %f' %(threshold[0], threshold[1], accuracy_score(true_labels, pred_labels)))
print(classification_report(true_labels, pred_labels, target_names = ["noise", "wearer", "non-wearer"]))

a = confusion_matrix(true_labels, pred_labels)
df_cm = pd.DataFrame(a, index = ["noise", "wearer", "non-wearer"],
                        columns = ["noise", "wearer", "non-wearer"])
plt.figure(figsize = (6,4))
sn.heatmap(df_cm, annot=True, cmap = plt.cm.Blues)
plt.title('confusion matrix (threshold %.3f, %.3f)' %(threshold[0], threshold[1]))
plt.show()


# Plot distribution
idx_ws = np.where(test_labels == 1)
idx_os = np.where(test_labels == 2)
idx_ns = np.where(test_labels == 0)

vec_ws = test_embedding[idx_ws]
vec_os = test_embedding[idx_os]
vec_ns = test_embedding[idx_ns]

dis_ws = clf.compute_distance(vec_ws)
dis_os = clf.compute_distance(vec_os)
dis_ns = clf.compute_distance(vec_ns)

plt.hist([dis_ws, dis_os,dis_ns], density  = True)
plt.legend(["wear", "non-wearer", "noise"])
plt.xlabel('cosine similarity')
plt.ylabel('density')
plt.show()
