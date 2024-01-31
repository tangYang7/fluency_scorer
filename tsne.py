# -*- coding: utf-8 -*-
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from tqdm import tqdm
from pprint import pprint
import pickle
import joblib


def load_feature(dataLoader, dataset_type):
    extract_feat_list = []
    file_path = f'data/{dataset_type}_feats.pkl'

    with open(file_path, 'rb') as file:
        saved_tensor_dict = pickle.load(file)

    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            feats = saved_tensor_dict[path]
            extract_feat_list.append(feats)
        if j == 2:
            break

    extract_feat_tensor = torch.concat(extract_feat_list, dim=0)
    print(extract_feat_tensor.shape)
    return extract_feat_tensor, saved_tensor_dict

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

class fluDataset(Dataset):
    def __init__(self, set):
        paths = load_file(f'speechocean762/{set}/wav.scp')
        for i in range(paths.shape[0]):
            paths[i] = paths[i].split('\t')[1]
        if set == 'train':
            self.utt_label = torch.tensor(np.load('data/tr_label_utt.npy'), dtype=torch.float)
        elif set == 'test':
            self.utt_label = torch.tensor(np.load('data/te_label_utt.npy'), dtype=torch.float)
        self.paths = paths

    def __len__(self):
        return self.utt_label.size(0)

    def __getitem__(self, idx):
        # audio, utt_label
        return self.paths[idx], self.utt_label[idx, :]
    
def cluster_pred(dataLoader, saved_tensor_dict, dataset_type, fig_path):
    cluster_pred_list = []
    # with open(f'{fig_path}/{dataset_type}_clust_index.txt', 'w') as file:
    #     pass
    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            feat_tensor = saved_tensor_dict[path]
            cluster_pred = cluster.predict(feat_tensor.numpy())
            # with open(f'{fig_path}/{dataset_type}_clust_index.txt', 'a') as file:
            #     pprint(cluster_pred, stream=file)
            pred_tensor = torch.tensor(cluster_pred)
            cluster_pred_list.append(pred_tensor)
        if j == 2:
            break

    cluster_pred_tensor = torch.concat(cluster_pred_list, dim=0)
    cluster_pred_tensor = cluster_pred_tensor.view(-1)

    print(cluster_pred_tensor.shape)
    return cluster_pred_tensor

def draw_fig(X_tsne, y, dataset_type, fig_path):    
    plt.figure(figsize=(12,10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.5,
                cmap=plt.cm.Spectral)
    # 加入圖例
    legend1 = plt.legend(*scatter.legend_elements(), title="Digits",
                        bbox_to_anchor=(1.03, 0.8), loc='upper left', 
                        edgecolors='k', s=50,
                        )
    plt.gca().add_artist(legend1)
    plt.savefig(f"{fig_path}/{dataset_type}_tsne.jpg")

def get_label(dataLoader):
    label_array = np.array([])

    for j, (paths, utt_label) in enumerate(dataLoader):
        for i, path in enumerate(paths):
            label_array = np.append(label_array, utt_label[i][2])

    return label_array

if __name__ == '__main__':

    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
    print('Preparing datasets')

    batch_size = 8
    tr_dataset = fluDataset('train')
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
    te_dataset = fluDataset('test')
    te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)
    print('Done.')

    fig_path = f'exp/tsne_fig'
    if os.path.exists(fig_path) == False:
        os.mkdir(fig_path)

    model_output_path = 'exp/kmeans'
    cluster = joblib.load(f'{model_output_path}/kmeans_model.joblib')


    extract_feat_tensor, saved_tensor_dict = load_feature(tr_dataloader, 'tr')    
    # y = get_label(tr_dataloader)
    y = cluster_pred(tr_dataloader, saved_tensor_dict, 'tr', fig_path)
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5,
                           perplexity= 30,
                           verbose=1).fit_transform(extract_feat_tensor.numpy())
    draw_fig(X_tsne, y, 'tr', fig_path)


    extract_feat_tensor, saved_tensor_dict = load_feature(te_dataloader, 'te')
    # y = get_label(te_dataloader)
    y = cluster_pred(te_dataloader, saved_tensor_dict, 'te', fig_path)
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, 
                            perplexity= 30,
                           verbose=1).fit_transform(extract_feat_tensor.numpy())
    draw_fig(X_tsne, y, 'te', fig_path)

    print('Mission Completed.')
