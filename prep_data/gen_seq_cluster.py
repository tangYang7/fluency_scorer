import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
import warnings
import pickle

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

num_clusters = 50
warnings.simplefilter(action='ignore', category=FutureWarning)
cluster = KMeans(n_clusters=num_clusters)

def extract_feature(dataLoader, dataset_type):
    extract_feat_list = []
    # 指定保存文件的路径
    file_path = f'../data/{dataset_type}_feats.pkl'

    # 使用 pickle.load() 从文件中读取张量
    with open(file_path, 'rb') as file:
        saved_tensor_dict = pickle.load(file)

    for j, (paths, utt_label) in enumerate(dataLoader):
        feats = saved_tensor_dict[paths].view(saved_tensor_dict[paths].size(1), -1)
        if j == 1:
            print(feats.shape)
        extract_feat_list.append(feats)
    extract_feat_tensor = torch.concat(extract_feat_list, dim=0)
    print(extract_feat_tensor.shape)
    return extract_feat_tensor, saved_tensor_dict

def cluster_pred(dataLoader, extract_feat_tensor, saved_tensor_dict, dataset_type):
    print('Start k-means prediction...')

    cluster_pred_dict = {}
    cluster_index_dict = {}
    cluster.fit(extract_feat_tensor.cpu().detach().numpy())
    print('done!')

    for j, (paths, utt_label) in enumerate(dataLoader):
        feat_tensor = saved_tensor_dict[paths].view(saved_tensor_dict[paths].size(1), -1)
        # print(feat_tensor.shape)
        cluster_pred = cluster.predict(feat_tensor.numpy())
        # print(cluster_pred)
        cluster_pred_bin = convert_bin(cluster_pred)
        if paths not in cluster_pred_dict:
            cluster_pred_dict[paths] = cluster_pred_bin

    for i in range(len(cluster.cluster_centers_)):
        cluster_index_dict[i] = cluster.cluster_centers_[i]

    with open(f'../data/{dataset_type}_cluster_index.pkl', 'wb') as file:
        pickle.dump(cluster_pred_dict, file)
    with open(f'../data/cluster_centers.pkl', 'wb') as file:
        pickle.dump(cluster_index_dict, file)


extract_feat_tensor, saved_tensor_dict = extract_feature(tr_dataloader, 'tr')
cluster_pred(tr_dataloader, extract_feat_tensor, saved_tensor_dict, 'tr')

extract_feat_tensor, saved_tensor_dict = extract_feature(te_dataloader, 'te')
cluster_pred_dict = {}
cluster_index_dict = {}
for j, (paths, utt_label) in enumerate(te_dataloader):
    feat_tensor = saved_tensor_dict[paths].view(saved_tensor_dict[paths].size(1), -1)
    # print(feat_tensor.shape)
    cluster_pred = cluster.predict(feat_tensor.numpy())
    # print(cluster_pred)
    cluster_pred_bin = convert_bin(cluster_pred)
    if paths not in cluster_pred_dict:
        cluster_pred_dict[paths] = cluster_pred_bin
    # if j == 7:
    #     break

with open(f'../data/te_cluster_index.pkl', 'wb') as file:
    pickle.dump(cluster_pred_dict, file)

print('Gen_seq_cluster: done.')
