import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle

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
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
te_dataset = fluDataset('test')
te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
wav2vec2 = torchaudio.pipelines.WAV2VEC2_LARGE.get_model()
wav2vec2 = wav2vec2.to(device)


def extract_feature(dataLoader, dataset_type):
    extract_feat_list = []

    for j, (paths, utt_label) in tqdm(enumerate(dataLoader), total=len(dataLoader)):
        # load waveform
        audio_list = []
        for path in paths:
            waveform, sample_rate = torchaudio.load(f"../speechocean762/{path}")
            audio_list.append(waveform)

        max_length = max(waveform.size(1) for waveform in audio_list)
        padded_audio_list = [torch.nn.functional.pad(waveform.squeeze(0), (0, max_length - waveform.size(1)), mode='constant').unsqueeze(0) for waveform in audio_list]
        audio = torch.stack(padded_audio_list, dim=0)
        audio = audio.to(device)
        audio = audio.view(audio.size(0), -1)

        with torch.inference_mode():
            audio_embedding, _ = wav2vec2.extract_features(audio)
        # print(audio_embedding.shape)

        for i, feats in enumerate(audio_embedding):
            if i == 14:
                my_feature = feats
        # print(my_feature.size())
        # print(my_feature)
        extract_feat_list.append(my_feature)

    print('====================')

    saved_tensor_dict = {}
    for j, (paths, utt_label) in enumerate(dataLoader):
        for path in paths:
            if path not in saved_tensor_dict:
                saved_tensor_dict[path] = extract_feat_list[j][0]

    with open(f'../data/{dataset_type}_feats.pkl', 'wb') as file:
        pickle.dump(saved_tensor_dict, file)

    extract_feat_tensor = torch.cat(extract_feat_list, dim=1)  # cat all frames in 1024 dim
    print(extract_feat_tensor.shape)
    extract_feat_tensor = extract_feat_tensor.view(extract_feat_tensor.size(1), -1)
    print(extract_feat_tensor.shape)

    return extract_feat_tensor, saved_tensor_dict


extract_feat_tensor, saved_tensor_dict = extract_feature(tr_dataloader, 'tr')
extract_feat_tensor, saved_tensor_dict = extract_feature(te_dataloader, 'te')
print('Gen_seq_acoustic_feature: done.')
