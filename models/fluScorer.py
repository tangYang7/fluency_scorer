import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_pooling(feature_tensor: torch.Tensor, padding_value: float = 0.0):
    mask = feature_tensor != padding_value
    count = torch.sum(mask, axis=1)
    feature_tensor = torch.where(mask, feature_tensor, 0)
    mean = torch.sum(feature_tensor, axis=1) / count

    return mean

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x) -> torch.Tensor:
        '''
        input x: tensor of shape (batch size, seqence_length, input_size) when batch_first=True
        output: tensor of shape (batch size, seqence_length, D * hidden_size) when batch_first=True
        D = 2 if bidirectional else 1
        '''

        ''' set initial hidden and cell states
        h0: tensor of shape (D∗num_layers, batch_size, hidden_size)
        c0: tensor of shape (D∗num_layers, batch_size, hidden_size)
        D = 2 if bidirectional else 1
        '''
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.blstm(x, (h0, c0))
        return out
    
# adapt: tanh -> GELU
class BiLSTMScorer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        '''
        BiLSTM(input_size, hidden_size, num_layers=num_layers)
        '''
        super().__init__()
        self.blstm = BiLSTM(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.activations = nn.ModuleDict([
                ['tanh', nn.Tanh()],
                ['GELU', nn.GELU()],
        ])
    
    def forward(self, x, act=None):
        BiLSTM_embedding = self.blstm(x)

        output = mean_pooling(BiLSTM_embedding)

        score = self.fc(output)

        if act != None:
            score = self.activations[act](score)
        else:
            score = self.activations['GELU'](score)
        return score

def _preprocessing(audio_embedding, preprocessing_module):
    audio_embedding_list = []
    for i in range(audio_embedding.size(0)):
        new_audio_embedding = preprocessing_module(audio_embedding[i])
        audio_embedding_list.append(new_audio_embedding)
    new_audio_embedding_tensor = torch.stack(audio_embedding_list, dim=0)
    return new_audio_embedding_tensor

class FluencyScorerNoclu(nn.Module):
    ''' 
        A model for fluency score prediction without using cluster.
    '''
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p=0.5),
            # nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )
        self.scorer = BiLSTMScorer(embed_dim+clustering_dim, embed_dim, 2)

    def forward(self, x):
        ''' 
        x: extract audio features
        return: a pred score
        '''
        # step 1: audio features preprocessing
        new_audio_embedding_tensor = _preprocessing(x, self.preprocessing)

        # step 2: make a score directly
        pred = self.scorer(new_audio_embedding_tensor)
        return pred

class FluencyScorer(nn.Module):
    ''' 
        The main model for fluency score prediction with using cluster.
    '''
    def __init__(self, input_dim, embed_dim, clustering_dim=6):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Tanh()
        )
        self.scorer = BiLSTMScorer(input_dim+clustering_dim, embed_dim, 2)

    def forward(self, x, cluster_id):
        ''' 
        x: extract audio features
        return: a pred score
        '''
        # step 1: audio features preprocessing
        new_audio_embedding_tensor = _preprocessing(x, self.preprocessing)

        # step 2: concat audio and cluster embedding
        audio_features = torch.concat((new_audio_embedding_tensor, cluster_id), dim=2)

        # step 3: make a score
        pred = self.scorer(audio_features)
        return pred
