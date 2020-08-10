import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot as plt
import seaborn as sn

train_feat_path = './recordings/remove_0.2_rms/preliminary/features/train_test_split/pre_outdoor_embedding/embedding_train_r0.20_s43.csv'
test_feat_path = './recordings/remove_0.2_rms/preliminary/features/train_test_split/pre_outdoor_embedding/embedding_test_r0.20_s43.csv'
threshold = 0.25
distsance_metric = 'cosine_similarity'

class binary_cluster_classifier():

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

        if not threshold:
            return dis > self.threshold
        else:
            return dis > threshold

def load_features(feat_path):
    a = np.genfromtxt(feat_path, delimiter=',')

    return a[:, :1000], a[:, -1]

if __name__ == "__main__":

    train_embedding, train_labels = load_features(train_feat_path)
    test_embedding, test_labels = load_features(test_feat_path)

    # get all the wearer speech vectors in training set
    train_idx_ws = np.where(train_labels == 1)
    train_vec_ws = train_embedding[train_idx_ws]

    # process test data labels: wearer speech -> True; non-wearer speech -> False
    idx_sp = np.where(test_labels != 0)
    vec_sp = test_embedding[idx_sp]
    true_labels = test_labels[idx_sp] == 1

    # fit and predit
    clf = binary_cluster_classifier(distance_metric = 'cosine_similarity', threshold = threshold)
    clf.fit(train_vec_ws)
    pred_labels = clf.predict(vec_sp)

    # print classification result
    print('Using threshold %.2f: accuracy %f' %(threshold, accuracy_score(true_labels, pred_labels)))
    print(classification_report(true_labels, pred_labels, target_names = ["other's speech", "wearer's speech"]))

    # print confusion matrix
    a = confusion_matrix(true_labels, pred_labels)
    df_cm = pd.DataFrame(a, index = ["other's speech", "wearer's speech"],
                        columns = ["other's speech", "wearer's speech"])
    plt.figure(figsize = (6,4))
    sn.heatmap(df_cm, annot=True)
    plt.title('confusion matrix (threshold %.2f)' %(threshold))
    plt.show()



