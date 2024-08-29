import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score
import os
import scipy.stats
import matplotlib.pyplot as plt

class Flu_TFR(nn.Module):
    def __init__(self, num_heads, depth=3, input_dim=84, dropout_prob=0.1,
                 activation='gelu', 
                 norm_first=True,
                 clustering_dim=6,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_prob = dropout_prob
        self.hidden_dim = 32
        self.projLayer = nn.Linear(input_dim, self.hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim+clustering_dim, nhead=num_heads, 
                                                        dim_feedforward=(self.hidden_dim+clustering_dim)*4, 
                                                        dropout=dropout_prob, 
                                                        activation=activation, batch_first=True,
                                                        norm_first=norm_first,
                            )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)

        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))

    def forward(self, x, cluster_idx):
        x = self.projLayer(x)
        x = torch.cat([x, cluster_idx], dim=2)
        mask = self.create_mask(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.mean_pooling(x, padding_value=0.0)
        x = self.mlp_head_utt1(x)
        return x
    
    def mean_pooling(self, feature_tensor: torch.Tensor, padding_value: float = 0.0):
        mask = feature_tensor != padding_value
        count = torch.sum(mask, axis=1)
        count = torch.clamp(count, min=1e-9)  # Avoid division by zero
        feature_tensor = torch.where(mask, feature_tensor, 0)
        mean = torch.sum(feature_tensor, axis=1) / count
        return mean
    
    def create_mask(self, x: torch.Tensor, padding_value: float = 0.0):
        # return mask [batch_size, seq_len]
        mask_2d = torch.all(x==padding_value, dim=-1)
        return mask_2d
