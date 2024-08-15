import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import warnings
import pickle
import os
import joblib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("SO762_dir", type=str, default='../speechocean762')
parser.add_argument("--feat_dir", type=str, default='../data')
parser.add_argument("--output_dir", type=str, default='../exp/kmeans')
args = parser.parse_args()

def load_feature(dataLoader, dataset_type):
    extract_feat_list = []
    file_path = f'{args.feat_dir}/{dataset_type}_feats.pkl'

    with open(file_path, 'rb') as file:
        saved_tensor_dict = pickle.load(file)

    for j, (paths, _) in enumerate(dataLoader):
        for path in paths:
            feats = saved_tensor_dict[path]
            extract_feat_list.append(feats.cpu())

    extract_feat_tensor = torch.concat(extract_feat_list, dim=0)
    print(extract_feat_tensor.shape)
    return extract_feat_tensor, saved_tensor_dict

def cluster_pred(dataLoader, saved_tensor_dict, dataset_type):
    cluster_pred_dict = {}
    for j, (paths, _) in enumerate(dataLoader):
        for path in paths:
            feat_tensor = saved_tensor_dict[path]
            # print(feat_tensor.shape)
            cluster_pred = cluster.predict(feat_tensor.cpu().numpy())
            cluster_pred_tensor = torch.tensor(cluster_pred)
            # print(cluster_pred_tensor)
            if path not in cluster_pred_dict:
                cluster_pred_dict[path] = cluster_pred_tensor

    with open(f'{args.feat_dir}/{dataset_type}_cluster_index.pkl', 'wb') as file:
        pickle.dump(cluster_pred_dict, file)

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

class fluDataset(Dataset):
    def __init__(self, set):
        paths = load_file(f'{args.SO762_dir}/{set}/wav.scp')
        for i in range(paths.shape[0]):
            paths[i] = paths[i].split('\t')[1]
        if set == 'train':
            self.utt_label = torch.tensor(np.load(f'{args.feat_dir}/tr_label_utt.npy'), dtype=torch.float)
        elif set == 'test':
            self.utt_label = torch.tensor(np.load(f'{args.feat_dir}/te_label_utt.npy'), dtype=torch.float)
        self.paths = paths

    def __len__(self):
        return self.utt_label.size(0)

    def __getitem__(self, idx):
        # audio, utt_label
        return self.paths[idx], self.utt_label[idx, :]

batch_size = 1

tr_dataset = fluDataset('train')
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
te_dataset = fluDataset('test')
te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

max_iter, num_clusters, bs, n_init = 100, 50, 10000, 20
warnings.simplefilter(action='ignore', category=FutureWarning)

cluster = MiniBatchKMeans(n_clusters=num_clusters,
                        max_iter=max_iter,
                        batch_size=bs,
                        n_init=n_init,
                        max_no_improvement=100,
                        random_state=0,
                        reassignment_ratio=0.0,
                        )

extract_feat_tensor, saved_tensor_dict = load_feature(tr_dataloader, 'tr')

print('Start k-means fit...')
cluster.fit(extract_feat_tensor.numpy())
print('\033[1;34mMission Completed: cluster_fit X.\033[0m')

model_output_path = args.output_dir
if not os.path.isdir(model_output_path):
    os.mkdir(model_output_path)
joblib.dump(cluster, f'{model_output_path}/kmeans_model.joblib')

cluster_index_dict = {}
for i in range(len(cluster.cluster_centers_)):
    cluster_index_dict[i] = cluster.cluster_centers_[i]
with open(f'{args.feat_dir}/cluster_centers.pkl', 'wb') as file:
    pickle.dump(cluster_index_dict, file)
print("Saved KMeans model.")

print('Start k-means prediction...')
cluster_pred(tr_dataloader, saved_tensor_dict, 'tr')
extract_feat_tensor, saved_tensor_dict = load_feature(te_dataloader, 'te')
cluster_pred(te_dataloader, saved_tensor_dict, 'te')
print('\033[1;34mMission Completed: cluster prediction.\033[0m')
