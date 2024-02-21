# Fluency Scorer

## Introduction
It's my implementation for speech fluency assessment model. 
The idea for this model is from the paper [An ASR-Free Fluency Scoring Approach with Self-Supervised Learning](<https://arxiv.org/abs/2302.09928>) (Wei Liu, Kaiqi Fu, Xiaohai Tian, Shuju Shi, Wei Li, Zejun Ma, Tan Lee) proposed in the ICASSP 2023.

## Data
The SpeechOcean762 dataset used in my work is an open dataset licenced with CC BY 4.0. 
It can be downloaded from [this link](<https://www.openslr.org/101>).

### Input Features and Labels
The input generation program are in `prep_data` and you need to download SpeechOcean762 first.
```
cd prep/data
python3 gen_seq_data_utt.py
python3 gen_seq_acoustic_feat.py
python3 train_kmeans.py
python3 kmeans_metric.py
```
- The labels are utterance-level scores, which the **fluency score is `utt_label[: 2]`**.
- The acoustic features are extracted by **Wav2vec_large**, where the dim is the value of 1024.
- The cluster also can be prepare in `gen_seq_acoustic_feat.py`, but my program got killed somehow maybe out of memory, so I handle them in seperated programs. 
- The feats and labels files are collected in `data`.
