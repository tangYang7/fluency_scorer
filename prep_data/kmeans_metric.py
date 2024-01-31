import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def load_feature(dataLoader, dataset_type):
    extract_feat_list = []
    file_path = f'../data/{dataset_type}_feats.pkl'

    with open(file_path, 'rb') as file:
        saved_tensor_dict = pickle.load(file)

    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            feats = saved_tensor_dict[path]
        extract_feat_list.append(feats)

    extract_feat_tensor = torch.concat(extract_feat_list, dim=0)
    print(extract_feat_tensor.shape)
    return extract_feat_tensor, saved_tensor_dict

def cluster_pred(dataLoader, saved_tensor_dict, dataset_type):
    cluster_pred_dict = {}
    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            feat_tensor = saved_tensor_dict[path]
            # print(feat_tensor.shape)
            cluster_pred = cluster.predict(feat_tensor.numpy())
            cluster_pred_tensor = torch.tensor(cluster_pred)
            # print(cluster_pred_tensor)
            if path not in cluster_pred_dict:
                cluster_pred_dict[path] = cluster_pred_tensor

    with open(f'../data/{dataset_type}_cluster_index.pkl', 'wb') as file:
        pickle.dump(cluster_pred_dict, file)
    if dataset_type == 'te':
        return 

def load_file(path):
    file = np.loadtxt(path, delimiter=',', dtype=str)
    return file

def convert_bin(input):
    # Convert each number to its binary representation
    binary_representations = [list(map(int, bin(num)[2:].zfill(6))) for num in input]
    
    # Convert to a PyTorch tensor
    tensor_2d = torch.tensor(binary_representations)
    return tensor_2d

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
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
te_dataset = fluDataset('test')
te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

model_path = '../exp/kmeans'
cluster = joblib.load(f'{model_path}/kmeans_model.joblib')


extract_feat_tensor, saved_tensor_dict = load_feature(tr_dataloader, 'tr')
print('<train>')
feat_numpy = extract_feat_tensor.cpu().numpy()
labels = cluster.predict(feat_numpy)

db_score = davies_bouldin_score(feat_numpy, labels)
print(f'Davies-Bouldin Score: {db_score :.3f} (⬇)')
ch_score = calinski_harabasz_score(feat_numpy, labels)
print(f'Calinski-Harabasz Score: {ch_score :.3f} (⬆)')
# si_score = silhouette_score(feat_numpy, cluster.labels_)
# print(f'Silhouette_score: {si_score}'(⬆))


print('<test>')
extract_feat_tensor, saved_tensor_dict = load_feature(te_dataloader, 'te')
feat_numpy = extract_feat_tensor.cpu().numpy()
labels = cluster.predict(feat_numpy)

db_score = davies_bouldin_score(feat_numpy, labels)
print(f'Davies-Bouldin Score: {db_score :.3f} (⬇)')
ch_score = calinski_harabasz_score(feat_numpy, labels)
print(f'Calinski-Harabasz Score: {ch_score :.3f} (⬆)')

print('Mission Completed.')
