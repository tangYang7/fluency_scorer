import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.GELU(),
        )

    def forward(self, x, attn, mask):
        w = self.attention(attn).float()
        w[mask==0] = float('-inf')
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x

def mean_pooling(feature_tensor, mask):
    mean = torch.sum(feature_tensor * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    return mean

def create_mask(feature_embedding, seq_lengths):
    device = feature_embedding.device
    B, T, D = feature_embedding.shape
    range_tensor_for_mask = torch.arange(T).expand(B, T).to(device)
    mask = range_tensor_for_mask < seq_lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand(B, T, D) 

    return mask

# adapt: tanh -> GELU
class BiLSTMScorer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        '''
        BiLSTM(input_size, hidden_size, num_layers=num_layers)
        '''
        super().__init__()
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        #self.fc = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, 1))

        self.activations = nn.ModuleDict([
                ['tanh', nn.Tanh()],
                ['GELU', nn.GELU()],
        ])

    def forward(self, x, act=None, seq_lengths=None):
        x_nopadded = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True)
        output, hidden = self.blstm(x_nopadded)
        BiLSTM_embedding, out_len = pad_packed_sequence(output, batch_first=True)        
        mask = create_mask(BiLSTM_embedding, seq_lengths)

        output = mean_pooling(BiLSTM_embedding, mask)

        score = self.fc(output)
        #score = self.activations['GELU'](score)
        
        return score

class FluencyScorerNoclu(nn.Module):
    ''' 
        A model for fluency score prediction without using cluster.
    '''
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )
        self.scorer = BiLSTMScorer(embed_dim, embed_dim, 2)

    def forward(self, x):
        ''' 
        x: extract audio features
        return: a pred score
        '''
        device = x.device
        # step 1: audio features preprocessing
        nonzero_mask = x.abs().sum(dim=2) != 0
        seq_lengths = nonzero_mask.sum(dim=1).to(device)
        new_audio_embedding_tensor = self.preprocessing(x)

        # create mask
        mask = create_mask(new_audio_embedding_tensor, seq_lengths)

        new_audio_embedding_tensor = new_audio_embedding_tensor * mask

        # step 2: make a score directly
        pred = self.scorer(x=new_audio_embedding_tensor, seq_lengths=seq_lengths)

        return pred

class FluencyScorer(nn.Module):
    ''' 
        The main model for fluency score prediction with using cluster.
    '''
    def __init__(self, input_dim, embed_dim, clustering_dim=6):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )
        self.cluster_embed = nn.Embedding(50 + 1, clustering_dim, padding_idx=0)
        self.scorer = BiLSTMScorer(embed_dim+clustering_dim, embed_dim, 2)

    def forward(self, x, cluster_id):
        ''' 
        x: extract audio features
        return: a pred score
        '''
        device = x.device
        # step 1: audio features preprocessing
        nonzero_mask = x.abs().sum(dim=2) != 0
        seq_lengths = nonzero_mask.sum(dim=1).to(device)
        new_audio_embedding_tensor = self.preprocessing(x)
        cluster_embed = self.cluster_embed(cluster_id).float()

        # step 2: concat audio and cluster embedding
        audio_features = torch.concat((new_audio_embedding_tensor, cluster_embed), dim=2)
        # create mask
        mask = create_mask(audio_features, seq_lengths)

        audio_features = audio_features * mask

        # step 3: make a score
        pred = self.scorer(x=audio_features, seq_lengths=seq_lengths)
        return pred
