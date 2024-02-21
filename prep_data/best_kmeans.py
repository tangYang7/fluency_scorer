import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
import warnings
import pickle
import os
import joblib

def load_feature(dataLoader, dataset_type):
    extract_feat_list = []
    file_path = f'../data/{dataset_type}_feats.pkl'

    with open(file_path, 'rb') as file:
        saved_tensor_dict = pickle.load(file)

    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            feats = saved_tensor_dict[path]
            extract_feat_list.append(feats.cpu())

    extract_feat_tensor = torch.concat(extract_feat_list, dim=0)
    print(extract_feat_tensor.shape)
    return extract_feat_tensor, saved_tensor_dict

def cluster_pred(dataLoader, saved_tensor_dict, dataset_type):
    cluster_pred_dict = {}
    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            feat_tensor = saved_tensor_dict[path]
            # print(feat_tensor.shape)
            cluster_pred = cluster.predict(feat_tensor.cpu().numpy())
            cluster_pred_tensor = torch.tensor(cluster_pred)
            # print(cluster_pred_tensor)
            if path not in cluster_pred_dict:
                cluster_pred_dict[path] = cluster_pred_tensor

    with open(f'../data/{dataset_type}_cluster_index.pkl', 'wb') as file:
        pickle.dump(cluster_pred_dict, file)

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

class fluDataset(Dataset):
    def __init__(self, set):
        paths = load_file(f'../speechocean762/{set}/wav.scp')
        for i in range(paths.shape[0]):
            paths[i] = paths[i].split('\t')[1]
        if set == 'train':
            self.utt_label = torch.tensor(np.load('../data/tr_label_utt.npy'), dtype=torch.float)
        elif set == 'test':
            self.utt_label = torch.tensor(np.load('../data/te_label_utt.npy'), dtype=torch.float)
        self.paths = paths

    def __len__(self):
        return self.utt_label.size(0)

    def __getitem__(self, idx):
        # audio, utt_label
        return self.paths[idx], self.utt_label[idx, :]

batch_size = 1

tr_dataset = fluDataset('train')
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
te_dataset = fluDataset('test')
te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

k_values = [
            20, 23, 25, 28, 30, 
            33, 35, 38, 40, 45, 
            50, 70, 100
            ]
Davies_bouldin_score = {}
DistanceList = {}
for k in k_values:
    print(f'Start k-means fit for k={k}...')

    max_iter, num_clusters, bs, n_init = 100, k, 10000, 20
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # cluster = KMeans(n_clusters=num_clusters)
    cluster = MiniBatchKMeans(n_clusters=num_clusters,
                            max_iter=max_iter,
                            batch_size=bs,
                            n_init=n_init,
                            random_state=42
                            )

    extract_feat_tensor, saved_tensor_dict = load_feature(tr_dataloader, 'tr')

    print('Start k-means fit...')
    cluster.fit(extract_feat_tensor.numpy())
    extract_feat_tensor, saved_tensor_dict = load_feature(te_dataloader, 'te')
    labels = cluster.predict(extract_feat_tensor.numpy())
    print('\033[1;34mMission Completed: cluster_fit X.\033[0m')

    Davies_bouldin_score[k] = davies_bouldin_score(extract_feat_tensor.numpy(), labels)
    DistanceList[k] = (cluster.inertia_)

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(list(Davies_bouldin_score.keys()), list(Davies_bouldin_score.values()), marker='o')
plt.title('Davies_bouldin_scores for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies_bouldin_score')
plt.grid(True)
plt.savefig('kmeans.jpg')

# Plot the DistanceList scores
plt.figure(figsize=(10, 6))
plt.plot(list(DistanceList.keys()), list(DistanceList.values()), marker='o')
plt.title('DistanceList for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distance')
plt.grid(True)
plt.savefig('kmeans2.jpg')
